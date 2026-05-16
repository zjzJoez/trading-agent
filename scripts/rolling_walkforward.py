"""Rolling walk-forward backtest — the gold-standard OOS evaluation.

DESIGN
------
For each test year Y in [first_year, last_year]:
    1. Train a fresh HMM on ALL feature snapshots with as_of < Y-01-01.
    2. Auto-calibrate state→label mapping using TRAINING-WINDOW data only
       (no future leakage): rank states by mean training-window spy_ret_20.
       Highest = BULL_TREND, lowest = BEAR_TREND, middle two = VOLATILE_TRANSITION.
    3. Classify each day in year Y using that HMM.
    4. Append predictions to a global series.

Then compute the same per-regime / headline metrics as the single-split
walkforward_backtest.py — but on the much larger OOS sample (~2500 days
for 2017-2026 vs 575 days for the 2024-2026 single split).

WHY AUTO-CALIBRATE
------------------
A single human review of 10 HMM fingerprints is impractical. The
spy_ret_20-rank rule is:
  - Deterministic (reproducible)
  - Uses ONLY training-window features (no leakage)
  - Forces the same calibration semantics across years so cross-year
    comparison is meaningful

The rule will sometimes label a "high-vol mixed-direction" state as
VOLATILE_TRANSITION when a human might have picked something else.
That's the trade-off — the trade-off is documented and the rule is
visible in the code.

WHAT THIS DOES NOT VALIDATE
---------------------------
Same caveat as walkforward_backtest.py: this tests Layer 0 + Layer 1 +
gate sizing on a SPY-long proxy. The LLM trader, debate, risk council,
and actual options pricing are not tested here.

USAGE
-----
    python -m scripts.rolling_walkforward \\
        --first-year 2017 \\
        --last-year 2026 \\
        --out-dir /tmp/rolling_walkforward
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from trading_agent.regime.classifier import HMMModel, classify
from trading_agent.regime.features import HMM_FEATURE_ORDER, FeatureSnapshot
from trading_agent.regime.gates import SIZE_MULTIPLIERS, regime_size_multiplier
from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)


def load_snapshots_range(start: str | None, end: str | None) -> list[tuple[datetime, FeatureSnapshot]]:
    """Load snapshots with DATE(as_of) BETWEEN [start, end). End is exclusive."""
    with cursor() as cur:
        if start and end:
            cur.execute(
                """
                SELECT as_of, features, source_freshness, data_quality
                FROM regime_feature_snapshots
                WHERE DATE(as_of) >= %s AND DATE(as_of) < %s
                ORDER BY as_of ASC
                """,
                (start, end),
            )
        elif end:
            cur.execute(
                """
                SELECT as_of, features, source_freshness, data_quality
                FROM regime_feature_snapshots
                WHERE DATE(as_of) < %s
                ORDER BY as_of ASC
                """,
                (end,),
            )
        else:
            raise ValueError("at least one of start/end required")
        rows = cur.fetchall()
    out = []
    for as_of, feats, freshness, dq in rows:
        out.append((as_of, FeatureSnapshot(
            as_of=as_of,
            features=dict(feats or {}),
            source_freshness=dict(freshness or {}),
            data_quality=dict(dq or {}),
        )))
    return out


def train_hmm_for_year(training_snaps: list[tuple[datetime, FeatureSnapshot]],
                       n_states: int, seed: int) -> tuple[HMMModel, dict]:
    """Fit a fresh HMM on `training_snaps` and auto-calibrate state labels.

    Returns (HMMModel, calibration_audit_dict). The audit dict records
    per-state mean spy_ret_20 + assigned label so we can verify the
    calibration was sane.
    """
    from hmmlearn.hmm import GaussianHMM
    # Build feature matrix
    valid = [(t, s) for (t, s) in training_snaps if s.degradation_level < 2]
    if len(valid) < 200:
        raise RuntimeError(f"Too few training snapshots: {len(valid)}")
    X = np.array([s.as_vector() for _, s in valid])
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    Z = (X - means) / stds

    model = GaussianHMM(
        n_components=n_states, covariance_type="full",
        n_iter=200, random_state=seed, tol=1e-4,
    )
    model.fit(Z)
    states = model.predict(Z)

    # Auto-calibrate: rank states by mean spy_ret_20 ON TRAINING SET
    spy_ret_20_idx = HMM_FEATURE_ORDER.index("spy_ret_20")
    state_metrics = {}
    for s in range(n_states):
        mask = (states == s)
        if mask.sum() == 0:
            state_metrics[s] = {"mean_spy_ret_20": 0.0, "n": 0}
        else:
            state_metrics[s] = {
                "mean_spy_ret_20": float(X[mask, spy_ret_20_idx].mean()),
                "n": int(mask.sum()),
            }

    # Sort states by mean spy_ret_20 (ascending). Lowest = BEAR, highest = BULL.
    ranked = sorted(range(n_states), key=lambda s: state_metrics[s]["mean_spy_ret_20"])
    state_to_label = {}
    for rank, s in enumerate(ranked):
        if rank == 0:
            state_to_label[s] = "BEAR_TREND"
        elif rank == n_states - 1:
            state_to_label[s] = "BULL_TREND"
        else:
            state_to_label[s] = "VOLATILE_TRANSITION"

    hmm = HMMModel(
        n_states=int(model.n_components),
        feature_order=list(HMM_FEATURE_ORDER),
        means=model.means_.tolist(),
        covars=model.covars_.tolist(),
        transmat=model.transmat_.tolist(),
        startprob=model.startprob_.tolist(),
        feature_means=means.tolist(),
        feature_stds=stds.tolist(),
        state_to_label=state_to_label,
        train_meta={"n_obs": len(valid)},
    )
    audit = {
        "state_metrics": state_metrics,
        "state_to_label": state_to_label,
        "ranked_states_low_to_high_spy_ret": ranked,
    }
    return hmm, audit


def yfinance_spy_returns(start_dt, end_dt) -> pd.Series:
    import yfinance as yf
    pad_start = (start_dt - timedelta(days=10)).isoformat()
    pad_end = (end_dt + timedelta(days=40)).isoformat()
    df = yf.download("SPY", start=pad_start, end=pad_end, progress=False, auto_adjust=True)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    rets = close.pct_change()
    rets.index = pd.to_datetime(rets.index).date
    return rets


def fwd_return(rets: pd.Series, day, horizon: int) -> float:
    idx = rets.index.searchsorted(day, side="right")
    take = rets.iloc[idx : idx + horizon]
    if len(take) < horizon:
        return float("nan")
    return float((1.0 + take).prod() - 1.0)


def main():
    p = argparse.ArgumentParser("rolling_walkforward")
    p.add_argument("--first-year", type=int, default=2017,
                   help="First test year (training uses everything before "
                        "Jan 1 of this year). Earlier than 2017 means too "
                        "few training samples (HMM convergence issues).")
    p.add_argument("--last-year", type=int, default=2026)
    p.add_argument("--n-states", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    all_preds = []
    audits = []
    prior_label_by_year_end = None

    for year in range(args.first_year, args.last_year + 1):
        train_end = f"{year}-01-01"
        test_start = f"{year}-01-01"
        test_end = f"{year + 1}-01-01"

        log.info("--- Year %d ---", year)
        log.info("Training on < %s", train_end)
        train_snaps = load_snapshots_range(None, train_end)
        if len(train_snaps) < 200:
            log.warning("Skipping year %d: only %d training snapshots", year, len(train_snaps))
            continue
        log.info("Training snapshots: %d", len(train_snaps))

        try:
            hmm, audit = train_hmm_for_year(train_snaps, args.n_states, args.seed)
        except Exception as e:
            log.error("Failed to train HMM for year %d: %s", year, e)
            continue

        audit["year"] = year
        audit["n_train"] = len(train_snaps)
        audits.append(audit)
        log.info("  State→label: %s", audit["state_to_label"])

        log.info("Classifying year %d [%s, %s)", year, test_start, test_end)
        test_snaps = load_snapshots_range(test_start, test_end)
        log.info("Test snapshots: %d", len(test_snaps))
        if not test_snaps:
            continue

        prior_label = prior_label_by_year_end
        for as_of, snap in test_snaps:
            d = as_of.date() if hasattr(as_of, "date") else as_of
            pred = classify(snap, hmm, prior_label=prior_label)
            mult = regime_size_multiplier(pred.label, pred.confidence)
            all_preds.append({
                "as_of": d.isoformat(),
                "test_year": year,
                "cutoff_used": train_end,
                "label": pred.label,
                "confidence": round(float(pred.confidence), 4),
                "size_mult": round(mult, 4),
                "degradation_level": pred.degradation_level,
                "crisis": int(bool(pred.crisis_flags)),
            })
            prior_label = pred.label
        prior_label_by_year_end = prior_label

    if not all_preds:
        log.error("No predictions produced — check first_year/last_year and snapshots")
        sys.exit(1)

    df = pd.DataFrame(all_preds)
    df["as_of"] = pd.to_datetime(df["as_of"]).dt.date

    # Forward returns across the whole OOS range
    log.info("Fetching SPY forward returns…")
    rets = yfinance_spy_returns(df["as_of"].min(), df["as_of"].max())
    df["fwd_1d_ret"] = df["as_of"].apply(lambda d: fwd_return(rets, d, 1))
    df["fwd_5d_ret"] = df["as_of"].apply(lambda d: fwd_return(rets, d, 5))
    df["fwd_20d_ret"] = df["as_of"].apply(lambda d: fwd_return(rets, d, 20))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rolling_per_day.csv", index=False)
    log.info("Wrote %s (%d rows across %d years)",
             out / "rolling_per_day.csv", len(df), df["test_year"].nunique())

    # Per-regime aggregates (across the whole OOS series)
    by_regime = []
    valid = df.dropna(subset=["fwd_20d_ret"])
    uncond_mean = float(valid["fwd_20d_ret"].mean())
    uncond_neg = float((valid["fwd_20d_ret"] < 0).mean())
    for label, g in df.groupby("label"):
        gv = g.dropna(subset=["fwd_20d_ret"])
        n = len(g)
        if label == "BULL_TREND":
            hits = int((g["fwd_20d_ret"] > 0).sum())
            hr = hits / n if n else float("nan")
        elif label == "BEAR_TREND":
            hits = int((g["fwd_20d_ret"] < 0).sum())
            hr = hits / n if n else float("nan")
        else:
            hr = float("nan")
        m20 = float(gv["fwd_20d_ret"].mean()) if len(gv) else float("nan")
        s20 = float(gv["fwd_20d_ret"].std()) if len(gv) > 1 else float("nan")
        sharpe = m20 / s20 * math.sqrt(252 / 20) if s20 and s20 > 0 else float("nan")
        by_regime.append({
            "label": label,
            "n_days": n,
            "mean_fwd_20d": round(m20, 6),
            "neg_rate_fwd_20d": round(float((gv["fwd_20d_ret"] < 0).mean()), 4) if len(gv) else None,
            "uncond_mean_20d": round(uncond_mean, 6),
            "uncond_neg_rate": round(uncond_neg, 4),
            "hit_rate_sign_match": round(hr, 4) if not math.isnan(hr) else None,
            "conditional_sharpe": round(sharpe, 4) if not math.isnan(sharpe) else None,
            "size_mult": SIZE_MULTIPLIERS.get(label, 0.0),
        })
    by_regime_df = pd.DataFrame(by_regime).sort_values("label")
    by_regime_df.to_csv(out / "rolling_by_regime.csv", index=False)
    log.info("\n%s", by_regime_df.to_string(index=False))

    # Per-year breakdown
    by_year = []
    for year, g in df.groupby("test_year"):
        gv = g.dropna(subset=["fwd_20d_ret"])
        yr_mean = float(gv["fwd_20d_ret"].mean()) if len(gv) else float("nan")
        labels_in_yr = g["label"].value_counts().to_dict()
        # strategy cumulative ret for the year
        valid_g = g.dropna(subset=["fwd_1d_ret"])
        cum = float((1.0 + valid_g["size_mult"] * valid_g["fwd_1d_ret"]).prod() - 1.0)
        spy_cum = float((1.0 + valid_g["fwd_1d_ret"]).prod() - 1.0)
        by_year.append({
            "year": int(year),
            "n_days": int(len(g)),
            "n_bull": int(labels_in_yr.get("BULL_TREND", 0)),
            "n_bear": int(labels_in_yr.get("BEAR_TREND", 0)),
            "n_transition": int(labels_in_yr.get("VOLATILE_TRANSITION", 0)),
            "n_crisis": int(labels_in_yr.get("CRISIS", 0)),
            "uncond_mean_20d": round(yr_mean, 6),
            "strategy_year_ret": round(cum, 6),
            "spy_year_ret": round(spy_cum, 6),
            "delta_pp": round((cum - spy_cum) * 100, 2),
        })
    by_year_df = pd.DataFrame(by_year)
    by_year_df.to_csv(out / "rolling_by_year.csv", index=False)
    log.info("\n%s", by_year_df.to_string(index=False))

    # Headline
    valid_h = df.dropna(subset=["fwd_1d_ret"])
    strat = valid_h["size_mult"] * valid_h["fwd_1d_ret"]
    spy = valid_h["fwd_1d_ret"]
    cum_strat = float((1.0 + strat).prod() - 1.0)
    cum_spy = float((1.0 + spy).prod() - 1.0)
    sr = strat.mean() / strat.std() * math.sqrt(252) if strat.std() > 0 else float("nan")
    ssr = spy.mean() / spy.std() * math.sqrt(252) if spy.std() > 0 else float("nan")
    cum_curve = (1.0 + strat).cumprod()
    mdd = float((cum_curve / cum_curve.cummax() - 1.0).min())
    headline = {
        "first_year": args.first_year,
        "last_year": args.last_year,
        "n_test_days": len(df),
        "n_with_full_forward": len(valid_h),
        "cum_strategy_ret": round(cum_strat, 6),
        "cum_spy_ret": round(cum_spy, 6),
        "strategy_sharpe_annualized": round(sr, 4),
        "spy_sharpe_annualized": round(ssr, 4),
        "sharpe_se_rough": round(1.0 / math.sqrt(len(valid_h)) * math.sqrt(252), 3),
        "strategy_max_drawdown": round(mdd, 6),
        "avg_size_mult": round(float(df["size_mult"].mean()), 4),
        "audits_per_year": audits,
    }
    (out / "rolling_headline.json").write_text(json.dumps(headline, indent=2, default=str))

    print("\n" + "=" * 72)
    print(f"ROLLING WALK-FORWARD — {args.first_year} to {args.last_year}")
    print(f"OOS days: {len(df)} across {df['test_year'].nunique()} year(s)")
    print(f"Strategy cum return: {cum_strat:+.2%}  SPY: {cum_spy:+.2%}")
    print(f"Strategy Sharpe:     {sr:.2f}          SPY: {ssr:.2f}")
    print(f"Max DD:              {mdd:.2%}")
    print(f"Avg size_mult:       {df['size_mult'].mean():.3f}")
    print("=" * 72)
    print(f"\nArtifacts in: {out}")


if __name__ == "__main__":
    main()
