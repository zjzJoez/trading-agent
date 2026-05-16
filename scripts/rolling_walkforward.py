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
                       n_states: int, seed: int,
                       cov_type: str = "diag",
                       min_covar: float = 1e-3) -> tuple[HMMModel, dict]:
    """Fit a fresh HMM on `training_snaps` and auto-calibrate state labels.

    Returns (HMMModel, calibration_audit_dict). The audit dict records
    per-state mean spy_ret_20 + assigned label so we can verify the
    calibration was sane.

    cov_type: 'diag' is now the default (was 'full' in earlier runs).
    'full' covariance + 17 features + ~250-2700 obs/year caused hmmlearn
    to abort EM at iter 3-4 (log-likelihood decreasing) in every annual
    refit. 'diag' is standard in finance regime-HMM literature and is
    numerically stable.
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
        n_components=n_states, covariance_type=cov_type,
        n_iter=200, random_state=seed, tol=1e-4,
        min_covar=min_covar,
    )
    model.fit(Z)
    states = model.predict(Z)

    # Auto-calibrate: rank states by mean spy_ret_20 ON TRAINING SET (no leakage).
    # Compute mean spy_ret_20 AND mean spy_rv_20 per state — RV is used to
    # differentiate the middle two states between RANGE_LOW_VOL (low-vol bull
    # grind, 0.75x sizing in production) and VOLATILE_TRANSITION (high-vol
    # transition, 0.5x).
    #
    # WHY THIS 4-REGIME RULE: matches the manual calibration semantics used
    # by the production HMM (regime_model_versions.id=4). Earlier versions of
    # this script used only 3 labels (BULL/BEAR/TRANSITION) — the two middle
    # states were both labeled VOLATILE_TRANSITION (0.5x), which systematically
    # under-sized the calm-bull state that production treats as RANGE_LOW_VOL.
    spy_ret_20_idx = HMM_FEATURE_ORDER.index("spy_ret_20")
    spy_rv_20_idx = HMM_FEATURE_ORDER.index("spy_rv_20")
    state_metrics = {}
    for s in range(n_states):
        mask = (states == s)
        if mask.sum() == 0:
            state_metrics[s] = {"mean_spy_ret_20": 0.0, "mean_spy_rv_20": 0.0, "n": 0}
        else:
            state_metrics[s] = {
                "mean_spy_ret_20": float(X[mask, spy_ret_20_idx].mean()),
                "mean_spy_rv_20": float(X[mask, spy_rv_20_idx].mean()),
                "n": int(mask.sum()),
            }

    # Sort states by mean spy_ret_20 (ascending). Lowest = BEAR, highest = BULL.
    ranked = sorted(range(n_states), key=lambda s: state_metrics[s]["mean_spy_ret_20"])
    state_to_label = {}
    if n_states == 4:
        # 4-regime calibration matching production:
        #   rank 0 (lowest ret)  → BEAR_TREND   (0.5x sizing)
        #   rank 3 (highest ret) → BULL_TREND   (1.0x sizing)
        #   middle 2: lower RV → RANGE_LOW_VOL (0.75x)
        #             higher RV → VOLATILE_TRANSITION (0.5x)
        state_to_label[ranked[0]] = "BEAR_TREND"
        state_to_label[ranked[-1]] = "BULL_TREND"
        mid_a, mid_b = ranked[1], ranked[2]
        rv_a = state_metrics[mid_a]["mean_spy_rv_20"]
        rv_b = state_metrics[mid_b]["mean_spy_rv_20"]
        if rv_a <= rv_b:
            state_to_label[mid_a] = "RANGE_LOW_VOL"
            state_to_label[mid_b] = "VOLATILE_TRANSITION"
        else:
            state_to_label[mid_a] = "VOLATILE_TRANSITION"
            state_to_label[mid_b] = "RANGE_LOW_VOL"
    else:
        # Fallback to 3-regime rule for n_states != 4 (kept for compat)
        for rank, s in enumerate(ranked):
            if rank == 0:
                state_to_label[s] = "BEAR_TREND"
            elif rank == n_states - 1:
                state_to_label[s] = "BULL_TREND"
            else:
                state_to_label[s] = "VOLATILE_TRANSITION"

    # Normalize covars to (n_states, d, d) full-matrix form regardless of
    # cov_type, so the inference path doesn't need to know how we fit it.
    covars_arr = np.asarray(model.covars_)
    if covars_arr.ndim == 2:  # diag
        n_st = covars_arr.shape[0]
        d_ = covars_arr.shape[1]
        covars_full = np.zeros((n_st, d_, d_), dtype=float)
        for i in range(n_st):
            np.fill_diagonal(covars_full[i], covars_arr[i])
    else:
        covars_full = covars_arr

    hmm = HMMModel(
        n_states=int(model.n_components),
        feature_order=list(HMM_FEATURE_ORDER),
        means=model.means_.tolist(),
        covars=covars_full.tolist(),
        transmat=model.transmat_.tolist(),
        startprob=model.startprob_.tolist(),
        feature_means=means.tolist(),
        feature_stds=stds.tolist(),
        state_to_label=state_to_label,
        train_meta={"n_obs": len(valid), "cov_type": cov_type, "seed": seed},
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


def run_one_seed(args, seed: int, out_dir: Path, spy_rets_cache: pd.Series | None) -> dict:
    """Run a full rolling walk-forward for a single seed, writing artifacts
    into `out_dir`. Returns a headline-metrics dict suitable for aggregation.

    `spy_rets_cache` lets the caller pre-fetch yfinance once and reuse across
    seeds — saves ~5s × n_seeds.
    """
    all_preds = []
    audits = []
    prior_label_by_year_end = None

    for year in range(args.first_year, args.last_year + 1):
        train_end = f"{year}-01-01"
        test_start = f"{year}-01-01"
        test_end = f"{year + 1}-01-01"

        log.info("[seed=%d] --- Year %d ---", seed, year)
        train_snaps = load_snapshots_range(None, train_end)
        if len(train_snaps) < 200:
            log.warning("Skipping year %d: only %d training snapshots", year, len(train_snaps))
            continue

        try:
            hmm, audit = train_hmm_for_year(
                train_snaps, args.n_states, seed,
                cov_type=args.cov_type, min_covar=args.min_covar,
            )
        except Exception as e:
            log.error("Failed to train HMM for year %d (seed=%d): %s", year, seed, e)
            continue

        audit["year"] = year
        audit["n_train"] = len(train_snaps)
        audit["seed"] = seed
        audits.append(audit)
        log.info("  State→label: %s", audit["state_to_label"])

        test_snaps = load_snapshots_range(test_start, test_end)
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
        raise RuntimeError(f"No predictions for seed={seed}")

    df = pd.DataFrame(all_preds)
    df["as_of"] = pd.to_datetime(df["as_of"]).dt.date

    rets = spy_rets_cache if spy_rets_cache is not None else \
        yfinance_spy_returns(df["as_of"].min(), df["as_of"].max())
    df["fwd_1d_ret"] = df["as_of"].apply(lambda d: fwd_return(rets, d, 1))
    df["fwd_5d_ret"] = df["as_of"].apply(lambda d: fwd_return(rets, d, 5))
    df["fwd_20d_ret"] = df["as_of"].apply(lambda d: fwd_return(rets, d, 20))

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "rolling_per_day.csv", index=False)

    return _compute_headline_metrics(df, audits, args, out_dir, seed)


def _compute_headline_metrics(df: pd.DataFrame, audits: list[dict],
                              args, out_dir: Path, seed: int) -> dict:
    """Compute and write per-regime + per-year + headline JSON for one seed."""

    out = out_dir  # alias for the rest of the function
    log.info("[seed=%d] Wrote %s (%d rows across %d years)",
             seed, out / "rolling_per_day.csv", len(df), df["test_year"].nunique())

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
        "seed": seed,
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
        "cov_type": args.cov_type,
        "min_covar": args.min_covar,
        "audits_per_year": audits,
    }
    (out / "rolling_headline.json").write_text(json.dumps(headline, indent=2, default=str))

    print(f"[seed={seed}] strat_cum={cum_strat:+.2%}  spy_cum={cum_spy:+.2%}  "
          f"strat_sharpe={sr:.2f}  spy_sharpe={ssr:.2f}  mdd={mdd:.2%}")
    return headline


def main():
    """Single-seed or multi-seed rolling walk-forward dispatcher."""
    p = argparse.ArgumentParser("rolling_walkforward")
    p.add_argument("--first-year", type=int, default=2017)
    p.add_argument("--last-year", type=int, default=2026)
    p.add_argument("--n-states", type=int, default=4)
    p.add_argument("--seed", type=int, default=42,
                   help="Single seed (used unless --seeds is given)")
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated list (e.g. '42,7,99,123,2026') for "
                        "multi-seed stability test. Overrides --seed.")
    p.add_argument("--cov-type", default="diag", choices=["diag", "full", "spherical", "tied"])
    p.add_argument("--min-covar", type=float, default=1e-3)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    # Pre-fetch SPY returns ONCE across all seeds — same data, no point re-fetching
    log.info("Pre-fetching SPY forward returns (shared across seeds)…")
    # We don't know the exact test range until after first seed runs, so use a
    # generous date window covering all expected test years
    from datetime import date
    spy_rets = yfinance_spy_returns(
        date(args.first_year, 1, 1),
        date(min(args.last_year + 1, 2099), 12, 31),
    )

    headlines = []
    for seed in seeds:
        seed_dir = base_out / f"seed_{seed}" if len(seeds) > 1 else base_out
        hl = run_one_seed(args, seed, seed_dir, spy_rets)
        headlines.append(hl)

    if len(seeds) > 1:
        # Multi-seed aggregate
        summary_rows = []
        for hl in headlines:
            summary_rows.append({
                "seed": hl["seed"],
                "cum_strategy_ret": hl["cum_strategy_ret"],
                "cum_spy_ret": hl["cum_spy_ret"],
                "strategy_sharpe": hl["strategy_sharpe_annualized"],
                "spy_sharpe": hl["spy_sharpe_annualized"],
                "max_drawdown": hl["strategy_max_drawdown"],
                "avg_size_mult": hl["avg_size_mult"],
            })
        summary_df = pd.DataFrame(summary_rows)
        # mean ± std row
        stats_row = {
            "seed": "mean ± std",
            "cum_strategy_ret": f"{summary_df['cum_strategy_ret'].mean():.4f} ± {summary_df['cum_strategy_ret'].std():.4f}",
            "cum_spy_ret": f"{summary_df['cum_spy_ret'].mean():.4f}",  # same across seeds
            "strategy_sharpe": f"{summary_df['strategy_sharpe'].mean():.4f} ± {summary_df['strategy_sharpe'].std():.4f}",
            "spy_sharpe": f"{summary_df['spy_sharpe'].mean():.4f}",
            "max_drawdown": f"{summary_df['max_drawdown'].mean():.4f} ± {summary_df['max_drawdown'].std():.4f}",
            "avg_size_mult": f"{summary_df['avg_size_mult'].mean():.4f} ± {summary_df['avg_size_mult'].std():.4f}",
        }
        summary_df.to_csv(base_out / "rolling_seed_summary.csv", index=False)
        with open(base_out / "rolling_seed_aggregate.json", "w") as f:
            json.dump({
                "seeds": seeds,
                "n_seeds": len(seeds),
                "cum_strategy_ret_mean": float(summary_df["cum_strategy_ret"].mean()),
                "cum_strategy_ret_std": float(summary_df["cum_strategy_ret"].std()),
                "strategy_sharpe_mean": float(summary_df["strategy_sharpe"].mean()),
                "strategy_sharpe_std": float(summary_df["strategy_sharpe"].std()),
                "spy_sharpe": float(summary_df["spy_sharpe"].mean()),
                "max_drawdown_mean": float(summary_df["max_drawdown"].mean()),
                "max_drawdown_std": float(summary_df["max_drawdown"].std()),
                "avg_size_mult_mean": float(summary_df["avg_size_mult"].mean()),
                "per_seed": summary_rows,
            }, f, indent=2)
        print("\n" + "=" * 72)
        print(f"MULTI-SEED STABILITY — {len(seeds)} seeds: {seeds}")
        print("=" * 72)
        print(summary_df.to_string(index=False))
        print(f"\nStrategy Sharpe across seeds: "
              f"mean={summary_df['strategy_sharpe'].mean():.3f}  "
              f"std={summary_df['strategy_sharpe'].std():.3f}")
        print(f"SPY Sharpe (baseline, same for all seeds): "
              f"{summary_df['spy_sharpe'].mean():.3f}")
        print(f"\nArtifacts: {base_out}")


if __name__ == "__main__":
    main()
