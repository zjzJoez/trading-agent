"""Validate the DEPLOYED production HMM — the artifact, not the architecture.

WHY THIS SCRIPT EXISTS
----------------------
rolling_walkforward.py validates an AUTO-calibrated architecture: each
annual refit ranks hidden states by training-window mean spy_ret_20 and
assigns labels mechanically. But the production model
(regime_model_versions, status='active' — currently id=5) carries a
HUMAN-assigned state→label mapping chosen via scripts/promote_hmm.py CLI
flags after fingerprint review. The thing the walk-forward numbers
describe was never the thing deployed. This script closes that gap: it
loads the ACTIVE row with its human mapping and replays classification
over a historical window using the exact same next-day-forward-return
methodology (helpers imported from trading_agent.regime.evaluation —
shared with rolling_walkforward, not copy-pasted).

WHAT IT REPORTS
---------------
  - per-regime occupancy (days, share of window)
  - regime-conditional forward return / vol (1d + 20d horizons)
  - the size-multiplier overlay's Sharpe, max drawdown, cumulative return
    vs SPY buy-and-hold
  - IR vs SPY with overlap-aware Newey-West (lag ~20) standard errors —
    the t-stat is the honest "is this distinguishable from zero" number

INTERPRETATION
--------------
This is a drawdown-control diagnostic, NOT an alpha claim. The replay
window (anything before 2026-06-13) is the burned 2017-2026 selection
window — see regime/holdout.py. Numbers from it describe the deployed
model's risk behavior; they cannot anoint a successor model. Replaying
into the holdout requires --break-glass and burns it.

CAVEAT: the active model's training window typically OVERLAPS any
historical replay window (id=5 trained 2007→2026-05). For the deployed-
artifact question that's acceptable — we're asking "what does the thing
in production actually do over history", not "is this architecture
better than another" — but it means conditional returns are partially
in-sample. The output JSON records the overlap so nobody mistakes this
for OOS evidence.

RUNTIME
-------
Needs Postgres (regime_model_versions + regime_feature_snapshots) and
yfinance — run on EC2. Locally it degrades with a clear message instead
of a stack trace.

USAGE
-----
    python -m scripts.validate_production_hmm \\
        --start 2017-01-01 --end 2026-06-12 \\
        --out-dir reports/production_model_validation
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from trading_agent.regime.classifier import classify
from trading_agent.regime.evaluation import (
    fwd_return,
    overlay_headline,
    yfinance_spy_returns,
)
from trading_agent.regime.gates import SIZE_MULTIPLIERS, regime_size_multiplier
from trading_agent.regime.holdout import (
    HOLDOUT_START,
    HoldoutViolation,
    require_holdout_intact,
)

log = logging.getLogger(__name__)


def _load_active_model_or_exit():
    """Load (HMMModel, info) for the ACTIVE production row, or exit cleanly.

    All Postgres access is funneled through here + _load_snapshots_or_exit
    so a laptop without POSTGRES_DSN gets one actionable message (exit 2)
    instead of a psycopg traceback.
    """
    try:
        from trading_agent.regime.persist import get_active_model_id, get_model_by_id

        active_id = get_active_model_id()
        if active_id is None:
            log.error("No ACTIVE row in regime_model_versions — nothing to validate. "
                      "Promote a model first (scripts/promote_hmm.py).")
            sys.exit(3)
        loaded = get_model_by_id(active_id)
        if loaded is None:
            log.error("Active model id=%d vanished between lookups", active_id)
            sys.exit(3)
        return loaded
    except SystemExit:
        raise
    except Exception as e:
        log.error(
            "Postgres unavailable (%s: %s). This script reads "
            "regime_model_versions + regime_feature_snapshots and is meant to "
            "run on EC2. Locally: export POSTGRES_DSN=... or run it there.",
            type(e).__name__, e,
        )
        sys.exit(2)


def _load_snapshots_or_exit(start: str, end: str):
    """Snapshot range loader with the same local-degrade contract."""
    # Reuse rolling_walkforward's loader ([start, end) exclusive end) — same
    # rows the walk-forward consumed, no second query dialect. `scripts/` is
    # not a package, so resolve the sibling file by path: a bare
    # `from scripts.rolling_walkforward import ...` only works when the repo
    # root happens to be on sys.path, which `python scripts/foo.py` does NOT
    # provide (first EC2 run failed exactly there, misreported as a Postgres
    # outage by the broad except below).
    try:
        import importlib.util
        rw_path = Path(__file__).resolve().parent / "rolling_walkforward.py"
        spec = importlib.util.spec_from_file_location("rolling_walkforward", rw_path)
        rw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rw)
        load_snapshots_range = rw.load_snapshots_range
    except Exception as e:
        log.error("cannot load scripts/rolling_walkforward.py (%s: %s)",
                  type(e).__name__, e)
        sys.exit(2)
    try:
        end_exclusive = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
        return load_snapshots_range(start, end_exclusive)
    except SystemExit:
        raise
    except Exception as e:
        log.error(
            "Postgres unavailable while loading snapshots (%s: %s). "
            "Run on EC2 or export POSTGRES_DSN=...",
            type(e).__name__, e,
        )
        sys.exit(2)


def main():
    p = argparse.ArgumentParser("validate_production_hmm")
    p.add_argument("--start", default="2017-01-01", help="YYYY-MM-DD inclusive")
    p.add_argument("--end", default=(HOLDOUT_START - timedelta(days=1)).isoformat(),
                   help="YYYY-MM-DD inclusive (default: last pre-holdout day)")
    p.add_argument("--out-dir", default="reports/production_model_validation")
    p.add_argument("--nw-lag", type=int, default=20,
                   help="Newey-West lag for the IR standard error (~20 matches "
                        "regime persistence + 20d forward-window overlap)")
    p.add_argument("--break-glass", action="store_true",
                   help="Replay into the frozen v6+ holdout window "
                        "(>= holdout.HOLDOUT_START). BURNS the holdout for "
                        "model selection — see regime/holdout.py.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ----- Holdout freeze: same guard as the walk-forward scripts -----
    try:
        require_holdout_intact(
            args.end,
            break_glass=args.break_glass,
            context=f"validate_production_hmm --end {args.end}",
        )
    except HoldoutViolation as e:
        log.error("%s", e)
        sys.exit(5)

    # ----- Load the deployed artifact: active model + human label mapping -----
    hmm, info = _load_active_model_or_exit()
    mapping = {int(k): v for k, v in dict(hmm.state_to_label).items()}
    rationale = (info.get("metrics") or {}).get("calibration_rationale")
    log.info("Active model id=%d (status=%s, train=%s → %s)",
             info["id"], info["status"], info["train_start"], info["train_end"])
    log.info("Human state→label mapping: %s", mapping)

    train_end = info.get("train_end")
    train_overlaps_replay = bool(train_end and str(train_end) >= args.start)
    if train_overlaps_replay:
        log.warning(
            "Training window (ends %s) OVERLAPS the replay window (starts %s). "
            "Fine for a deployed-artifact diagnostic; NOT OOS evidence.",
            train_end, args.start,
        )

    # ----- Replay classification day by day -----
    snaps = _load_snapshots_or_exit(args.start, args.end)
    if not snaps:
        log.error("No feature snapshots in [%s, %s]", args.start, args.end)
        sys.exit(4)
    log.info("Replaying %d snapshots in [%s, %s]", len(snaps), args.start, args.end)

    try:
        spy_rets = yfinance_spy_returns(snaps[0][0].date(), snaps[-1][0].date())
    except (ImportError, RuntimeError) as e:
        log.error("Cannot compute forward returns: %s", e)
        sys.exit(1)

    per_day = []
    prior_label = None
    for as_of, snap in snaps:
        d = as_of.date() if hasattr(as_of, "date") else as_of
        pred = classify(snap, hmm, prior_label=prior_label)
        mult = regime_size_multiplier(pred.label, pred.confidence)
        per_day.append({
            "as_of": d.isoformat(),
            "label": pred.label,
            "confidence": round(float(pred.confidence), 4),
            "size_mult": round(mult, 4),
            "degradation_level": pred.degradation_level,
            "crisis": int(bool(pred.crisis_flags)),
            "fwd_1d_ret": fwd_return(spy_rets, d, 1),
            "fwd_20d_ret": fwd_return(spy_rets, d, 20),
        })
        prior_label = pred.label

    df = pd.DataFrame(per_day)

    # ----- Per-regime occupancy + conditional forward return / vol -----
    by_regime = []
    n_total = len(df)
    for label, g in df.groupby("label"):
        g1 = g.dropna(subset=["fwd_1d_ret"])
        g20 = g.dropna(subset=["fwd_20d_ret"])
        mean_1d = float(g1["fwd_1d_ret"].mean()) if len(g1) else float("nan")
        vol_1d = float(g1["fwd_1d_ret"].std()) if len(g1) > 1 else float("nan")
        mean_20d = float(g20["fwd_20d_ret"].mean()) if len(g20) else float("nan")
        vol_20d = float(g20["fwd_20d_ret"].std()) if len(g20) > 1 else float("nan")
        cond_sharpe = (mean_20d / vol_20d * math.sqrt(252.0 / 20.0)
                       if vol_20d and vol_20d > 0 else float("nan"))
        by_regime.append({
            "label": label,
            "n_days": int(len(g)),
            "occupancy_pct": round(100.0 * len(g) / n_total, 2),
            "mean_fwd_1d": round(mean_1d, 6),
            "vol_fwd_1d": round(vol_1d, 6),
            "mean_fwd_20d": round(mean_20d, 6),
            "vol_fwd_20d": round(vol_20d, 6),
            "conditional_sharpe_20d": round(cond_sharpe, 4) if cond_sharpe == cond_sharpe else None,
            "size_mult": SIZE_MULTIPLIERS.get(label, 0.0),
        })
    by_regime_df = pd.DataFrame(by_regime).sort_values("label")

    # ----- Overlay headline vs SPY buy-and-hold (shared helper) -----
    df_valid = df.dropna(subset=["fwd_1d_ret"])
    headline = overlay_headline(
        (df_valid["size_mult"] * df_valid["fwd_1d_ret"]).to_numpy(),
        df_valid["fwd_1d_ret"].to_numpy(),
        nw_lag=args.nw_lag,
    )

    # ----- Artifacts -----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "production_per_day.csv", index=False)
    result = {
        "generated_at": datetime.utcnow().isoformat(),
        "model": {
            "id": info["id"],
            "status": info["status"],
            "train_start": info["train_start"],
            "train_end": info["train_end"],
            "state_to_label": mapping,
            "calibration_rationale": rationale,
        },
        "window": {
            "start": args.start,
            "end": args.end,
            "n_days": n_total,
            "n_days_with_full_forward": int(len(df_valid)),
            "train_overlaps_replay": train_overlaps_replay,
            "holdout_start": HOLDOUT_START.isoformat(),
            "break_glass_used": bool(args.break_glass),
        },
        "by_regime": by_regime,
        "overlay_headline": headline,
        "interpretation_caveats": [
            "Deployed-artifact diagnostic of the ACTIVE human-calibrated model; "
            "training window may overlap the replay window (see train_overlaps_replay).",
            "Overlay = size_mult(label(d)) * SPY_next_1d_ret, compounded. Not a "
            "tradeable PnL — no costs, slippage, options pricing, or position sizing.",
            "Treat as drawdown control, not alpha: quote the NW t-stat, not the raw IR.",
            "Replay window predating 2026-06-13 is the burned v1-v5 selection window "
            "(regime/holdout.py) — it cannot adjudicate v6+ candidates.",
        ],
    }
    (out_dir / "production_validation.json").write_text(
        json.dumps(result, indent=2, default=str)
    )

    # ----- Console summary -----
    a = headline["active_vs_spy"]
    print("\n" + "=" * 72)
    print(f"PRODUCTION HMM VALIDATION — regime_model_versions id={info['id']} "
          f"(human-calibrated)")
    print(f"Replay window: {args.start} → {args.end} ({n_total} days)"
          + ("  [TRAIN OVERLAPS — not OOS]" if train_overlaps_replay else ""))
    print("=" * 72)
    print("\nPer-regime occupancy + conditional forward returns:")
    print(by_regime_df.to_string(index=False))
    print(f"\nOverlay cum return: {headline['cum_strategy_ret']:+.2%}  "
          f"vs SPY buy-and-hold: {headline['cum_spy_ret']:+.2%}  "
          f"(gap {headline['cum_gap_pp']:+.1f}pp)")
    print(f"Overlay Sharpe: {headline['strategy_sharpe_annualized']:.3f}  "
          f"SPY Sharpe: {headline['spy_sharpe_annualized']:.3f}")
    print(f"Overlay max drawdown: {headline['strategy_max_drawdown']:.2%}  "
          f"SPY max drawdown: {headline['spy_max_drawdown']:.2%}")
    print(f"IR vs SPY (ann.): {a['ir_annualized']:.3f}   "
          f"NW(lag={a['nw_lag']}) t-stat: {a['t_stat_nw']:.2f}   "
          f"(naive SE would be {a['naive_se_mean_daily']:.2e}, "
          f"NW SE {a['nw_se_mean_daily']:.2e})")
    print(f"\nArtifacts: {out_dir}")
    print("\nINTERPRETATION: drawdown-control diagnostic of the deployed model.")
    print("Not an alpha claim; not admissible for v6+ selection (window burned).")


if __name__ == "__main__":
    main()
