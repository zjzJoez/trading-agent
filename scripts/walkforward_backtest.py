"""Out-of-sample walk-forward backtest for the regime classifier + gate.

WHAT THIS VALIDATES
-------------------
Layer 0 (crisis overlay) + Layer 1 (HMM) + gate sizing logic — i.e. the
*deterministic* spine of the regime pipeline. The script replays
`classify()` on each held-out day with a backtest-only HMM that was
trained on data BEFORE the test window, then checks whether the labels
produced have predictive value for forward SPY returns.

WHAT THIS DOES NOT VALIDATE
---------------------------
The LLM trader, debate, and risk council are not replayable here — each
historical day would need fresh LLM calls. This backtest answers
"does the deterministic regime classifier separate forward returns?",
not "does the system make money." That distinction matters; do not
oversell the result.

DESIGN
------
1. Load an HMM by `--hmm-id` (typically a status='shadow' model whose
   `train_end_date` predates `--test-start`). Refuse to run if the model's
   training window overlaps the test window — that would be in-sample
   evaluation and is the bias this script exists to avoid.
2. Pull every snapshot from `regime_feature_snapshots` in
   [test_start, test_end].
3. For each snapshot:
   - Reconstruct a `FeatureSnapshot` from the persisted JSON.
   - Call `classify()` with the shadow HMM.
   - Record the label, confidence, size multiplier, and crisis flags.
4. Pull SPY daily closes (from regime_feature_snapshots — we have spy_ret_5
   etc. but we need forward returns, so we use the persisted SPY price if
   present; else fetch from yfinance).
5. For each day, compute SPY forward returns at horizons 1d / 5d / 20d.
6. Aggregate per-regime and write three artifacts:
   - `walkforward_per_day.csv`  (one row per test day)
   - `walkforward_by_regime.csv` (per-label aggregates: n_days,
     mean_fwd_5d, mean_fwd_20d, std_fwd_20d, hit_rate_sign, sharpe)
   - `walkforward_headline.json` (the regime-weighted strategy return
     vs passive SPY)

METRICS ARE PRE-REGISTERED
--------------------------
All three metrics (directional, hit rate, regime Sharpe) are computed
and written. Do not cherry-pick post-hoc.

USAGE
-----
    python -m scripts.walkforward_backtest \\
        --hmm-id 2 \\
        --test-start 2024-01-01 \\
        --test-end 2026-05-15 \\
        --out-dir /tmp/walkforward_v1
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

from trading_agent.regime.classifier import classify
from trading_agent.regime.features import FeatureSnapshot
from trading_agent.regime.gates import SIZE_MULTIPLIERS, regime_size_multiplier
from trading_agent.regime.persist import get_model_by_id
from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)


def load_test_snapshots(test_start: str, test_end: str) -> list[tuple[datetime, FeatureSnapshot]]:
    """Pull snapshots in [test_start, test_end] (inclusive) from Postgres.

    Returns a list of (as_of, FeatureSnapshot) tuples in chronological order.
    Snapshots with degradation_level>=2 are kept; classify() will route them
    to safe-mode TRANSITION, which is also the production behavior — the
    backtest should reflect that.
    """
    with cursor() as cur:
        cur.execute(
            """
            SELECT as_of, features, source_freshness, data_quality
            FROM regime_feature_snapshots
            WHERE DATE(as_of) BETWEEN %s AND %s
            ORDER BY as_of ASC
            """,
            (test_start, test_end),
        )
        rows = cur.fetchall()

    out = []
    for as_of, feats, freshness, dq in rows:
        snap = FeatureSnapshot(
            as_of=as_of,
            features=dict(feats or {}),
            source_freshness=dict(freshness or {}),
            data_quality=dict(dq or {}),
        )
        out.append((as_of, snap))
    return out


def build_spy_price_series(snaps: list[tuple[datetime, FeatureSnapshot]]) -> pd.Series:
    """Reconstruct SPY close-to-close return path from successive snapshots.

    Strategy: from each snapshot we KNOW spy_ret_5. So forward returns can be
    computed from the *snapshot sequence itself*: if snapshot at day d has
    spy_ret_5, that's already SPY's 5-day return ending d. For forward
    returns we shift: forward_5d at day d = spy_ret_5 measured at day d+5.

    More robust: derive 1-day SPY returns from successive spy_ret_5 differences,
    but that compounds drift. Cleanest: hit yfinance once and cache. To keep
    this script self-contained, we fall back to using the next snapshot's
    spy_ret_5 / spy_ret_20 values as the realized forward return.

    Returns a pd.Series indexed by DATE (not datetime) of "approximate next-
    day forward log return", reconstructed from the snapshot diff.
    """
    # We approximate: forward_1d_ret(d) ≈ spy_ret_5(d+1) - spy_ret_5(d) * 0.8
    # This is fragile. Better: just pull yfinance SPY directly.
    return _yfinance_spy_returns(snaps[0][0].date(), snaps[-1][0].date())


def _yfinance_spy_returns(start_dt, end_dt) -> pd.Series:
    """Daily SPY simple returns from yfinance. Returns pd.Series indexed by date."""
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not available; cannot compute forward returns")
        sys.exit(1)

    # Pad by 30 days each side to ensure we have data for forward windows
    pad_start = (start_dt - timedelta(days=10)).isoformat()
    pad_end = (end_dt + timedelta(days=40)).isoformat()

    df = yf.download("SPY", start=pad_start, end=pad_end, progress=False, auto_adjust=True)
    if df.empty:
        log.error("yfinance returned no SPY data for %s → %s", pad_start, pad_end)
        sys.exit(1)
    close = df["Close"]
    # When yfinance returns a MultiIndex (Ticker, Field), squeeze it.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    rets = close.pct_change()
    rets.index = pd.to_datetime(rets.index).date
    return rets


def forward_returns(spy_rets: pd.Series, as_of_date, horizon_days: int) -> float:
    """Compound SPY simple returns over [d+1, d+horizon_days], inclusive.

    Returns NaN if any required day is missing (e.g. test window ends inside
    the forward horizon and we don't have the future data).
    """
    target_dates = []
    cur = as_of_date
    # Skip non-business days: advance until we have `horizon_days` entries
    needed = horizon_days
    idx = spy_rets.index.searchsorted(as_of_date, side="right")
    take = spy_rets.iloc[idx : idx + horizon_days]
    if len(take) < horizon_days:
        return float("nan")
    return float((1.0 + take).prod() - 1.0)


def main():
    p = argparse.ArgumentParser("walkforward_backtest")
    p.add_argument("--hmm-id", type=int, required=True,
                   help="regime_model_versions.id of the backtest HMM. "
                        "MUST have train_end_date < --test-start.")
    p.add_argument("--test-start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--test-end", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--out-dir", required=True,
                   help="Directory to write per_day.csv + by_regime.csv + headline.json")
    p.add_argument("--allow-overlap", action="store_true",
                   help="Override the train/test overlap safety check. Use only "
                        "when you know what you're doing (sanity-check runs).")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ----- Load HMM + sanity check train/test gap -----
    loaded = get_model_by_id(args.hmm_id)
    if loaded is None:
        log.error("HMM id=%d not found in regime_model_versions", args.hmm_id)
        sys.exit(2)
    hmm, info = loaded
    log.info("Loaded HMM id=%d (status=%s, train=%s → %s)",
             info["id"], info["status"], info["train_start"], info["train_end"])

    train_end = info.get("train_end")
    if train_end and not args.allow_overlap:
        if str(train_end) >= args.test_start:
            log.error(
                "REFUSING TO RUN: HMM train_end=%s >= test_start=%s. "
                "This would be in-sample evaluation. Train a separate HMM "
                "with --train-end-date strictly before %s, or pass "
                "--allow-overlap if you're intentionally running a sanity check.",
                train_end, args.test_start, args.test_start,
            )
            sys.exit(3)

    # ----- Pull test-window snapshots -----
    snaps = load_test_snapshots(args.test_start, args.test_end)
    log.info("Loaded %d test snapshots in [%s, %s]",
             len(snaps), args.test_start, args.test_end)
    if not snaps:
        log.error("No test snapshots found")
        sys.exit(4)

    # ----- Forward SPY returns -----
    log.info("Fetching SPY price series from yfinance for forward-return calc...")
    spy_rets = _yfinance_spy_returns(snaps[0][0].date(), snaps[-1][0].date())
    log.info("SPY return series: %d days from %s to %s",
             len(spy_rets), spy_rets.index[0], spy_rets.index[-1])

    # ----- Classify each test day -----
    per_day = []
    prior_label = None
    for as_of, snap in snaps:
        d = as_of.date() if hasattr(as_of, "date") else as_of
        pred = classify(snap, hmm, prior_label=prior_label)
        size_mult = regime_size_multiplier(pred.label, pred.confidence)
        per_day.append({
            "as_of": d.isoformat(),
            "label": pred.label,
            "confidence": round(float(pred.confidence), 4),
            "size_mult": round(size_mult, 4),
            "degradation_level": pred.degradation_level,
            "crisis": int(bool(pred.crisis_flags)),
            "needs_llm_review": int(pred.needs_llm_review),
            "fwd_1d_ret": forward_returns(spy_rets, d, 1),
            "fwd_5d_ret": forward_returns(spy_rets, d, 5),
            "fwd_20d_ret": forward_returns(spy_rets, d, 20),
        })
        prior_label = pred.label

    df = pd.DataFrame(per_day)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_day_path = out_dir / "walkforward_per_day.csv"
    df.to_csv(per_day_path, index=False)
    log.info("Wrote %s (%d rows)", per_day_path, len(df))

    # ----- Per-regime aggregates -----
    by_regime_rows = []
    for label, g in df.groupby("label"):
        valid = g.dropna(subset=["fwd_20d_ret"])
        n = int(len(g))
        # Hit rate: BULL → fwd_20d > 0, BEAR → fwd_20d < 0, others n/a
        if label == "BULL_TREND":
            hits = int((g["fwd_20d_ret"] > 0).sum())
            hit_rate = hits / n if n else float("nan")
        elif label == "BEAR_TREND":
            hits = int((g["fwd_20d_ret"] < 0).sum())
            hit_rate = hits / n if n else float("nan")
        else:
            hit_rate = float("nan")
        mean_20d = float(valid["fwd_20d_ret"].mean()) if len(valid) else float("nan")
        std_20d = float(valid["fwd_20d_ret"].std()) if len(valid) > 1 else float("nan")
        # Conditional Sharpe: annualized using ~12.6 = sqrt(252/20-day windows / year)
        if std_20d and std_20d > 0:
            cond_sharpe = mean_20d / std_20d * math.sqrt(252.0 / 20.0)
        else:
            cond_sharpe = float("nan")
        by_regime_rows.append({
            "label": label,
            "n_days": n,
            "mean_fwd_5d": round(float(g["fwd_5d_ret"].mean()) if len(g) else float("nan"), 6),
            "mean_fwd_20d": round(mean_20d, 6),
            "std_fwd_20d": round(std_20d, 6),
            "hit_rate_sign_match": round(hit_rate, 4) if not math.isnan(hit_rate) else None,
            "conditional_sharpe": round(cond_sharpe, 4) if not math.isnan(cond_sharpe) else None,
            "size_mult": round(SIZE_MULTIPLIERS.get(label, 0.0), 4),
        })
    by_regime = pd.DataFrame(by_regime_rows).sort_values("label")
    by_regime_path = out_dir / "walkforward_by_regime.csv"
    by_regime.to_csv(by_regime_path, index=False)
    log.info("Wrote %s\n%s", by_regime_path, by_regime.to_string(index=False))

    # ----- Headline: regime-weighted strategy vs passive SPY -----
    # "Strategy": each day d, position size = size_mult(label(d)) * 1.0,
    #   return = size_mult * SPY_next_1d_ret. Compounded daily, this is what
    #   a hypothetical "use regime gate to scale a SPY long" would have done.
    # NOT a tradeable strategy — no costs, slippage, or position sizing.
    df_valid = df.dropna(subset=["fwd_1d_ret"]).copy()
    df_valid["strategy_ret"] = df_valid["size_mult"] * df_valid["fwd_1d_ret"]
    df_valid["spy_ret"] = df_valid["fwd_1d_ret"]
    cum_strategy = (1.0 + df_valid["strategy_ret"]).prod() - 1.0
    cum_spy = (1.0 + df_valid["spy_ret"]).prod() - 1.0
    daily_strategy_mean = df_valid["strategy_ret"].mean()
    daily_strategy_std = df_valid["strategy_ret"].std()
    daily_spy_mean = df_valid["spy_ret"].mean()
    daily_spy_std = df_valid["spy_ret"].std()
    strategy_sharpe = (
        daily_strategy_mean / daily_strategy_std * math.sqrt(252.0)
        if daily_strategy_std and daily_strategy_std > 0 else float("nan")
    )
    spy_sharpe = (
        daily_spy_mean / daily_spy_std * math.sqrt(252.0)
        if daily_spy_std and daily_spy_std > 0 else float("nan")
    )
    # Max drawdown of strategy cumulative curve
    cum_curve = (1.0 + df_valid["strategy_ret"]).cumprod()
    running_max = cum_curve.cummax()
    max_dd = float((cum_curve / running_max - 1.0).min())

    headline = {
        "hmm_id": args.hmm_id,
        "hmm_train_window": [info.get("train_start"), info.get("train_end")],
        "test_start": args.test_start,
        "test_end": args.test_end,
        "n_test_days": len(df),
        "n_days_with_full_forward": len(df_valid),
        "regime_count": {row["label"]: row["n_days"] for _, row in by_regime.iterrows()},
        "cum_strategy_ret": round(cum_strategy, 6),
        "cum_spy_ret": round(cum_spy, 6),
        "strategy_sharpe_annualized": round(strategy_sharpe, 4) if not math.isnan(strategy_sharpe) else None,
        "spy_sharpe_annualized": round(spy_sharpe, 4) if not math.isnan(spy_sharpe) else None,
        "strategy_max_drawdown": round(max_dd, 6),
        "interpretation_caveats": [
            "Strategy = size_mult(label(d)) * SPY_next_1d_ret, compounded daily.",
            "This is NOT a tradeable PnL — no costs, slippage, options pricing, or position sizing.",
            "Only validates Layer 0 + Layer 1 + gate sizing. LLM trader/debate/council NOT tested.",
            "Single-asset proxy (SPY long). Real system trades options on individual tickers.",
        ],
    }
    headline_path = out_dir / "walkforward_headline.json"
    headline_path.write_text(json.dumps(headline, indent=2, default=str))
    log.info("Wrote %s", headline_path)

    # ----- Console summary -----
    print("\n" + "=" * 72)
    print(f"WALK-FORWARD BACKTEST — HMM id={args.hmm_id}")
    print(f"Train: {info.get('train_start')} → {info.get('train_end')}")
    print(f"Test:  {args.test_start} → {args.test_end} ({len(df)} days)")
    print("=" * 72)
    print("\nPer-regime forward 20-day SPY return:")
    print(by_regime.to_string(index=False))
    print(f"\nHeadline: regime-gated strategy cum return = {cum_strategy:+.2%}")
    print(f"          passive SPY cum return         = {cum_spy:+.2%}")
    print(f"          strategy Sharpe (annualized)   = {strategy_sharpe:.2f}")
    print(f"          passive SPY Sharpe (annualized) = {spy_sharpe:.2f}")
    print(f"          strategy max drawdown          = {max_dd:.2%}")
    print(f"\nArtifacts in: {out_dir}")
    print("\nINTERPRETATION CAVEAT:")
    print("This validates Layer 0+1+gate sizing only, on a SPY-long proxy.")
    print("It does NOT validate the LLM trader, debate, or risk council.")


if __name__ == "__main__":
    main()
