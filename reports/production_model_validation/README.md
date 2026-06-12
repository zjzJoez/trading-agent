# Production model validation artifacts

Output directory for `scripts/validate_production_hmm.py` — the script
that measures the DEPLOYED human-calibrated HMM (regime_model_versions
status='active'), as opposed to the auto-calibrated architecture that
`scripts/rolling_walkforward.py` validates.

Artifacts per run:

- `production_validation.json` — model identity + human state→label
  mapping, per-regime occupancy and conditional forward return/vol,
  overlay Sharpe / max drawdown / cumulative return vs SPY buy-and-hold,
  IR vs SPY with Newey-West (lag ~20) standard errors, and the
  interpretation caveats.
- `production_per_day.csv` — per-day replayed classification + forward
  returns.

Runs on EC2 (needs Postgres snapshots + yfinance). Replaying past
2026-06-13 requires `--break-glass` and burns the v6+ holdout — see
`src/trading_agent/regime/holdout.py`.

Interpretation rule: these numbers describe the deployed model's
**drawdown-control** behavior. They are not OOS alpha evidence (the
active model's training window overlaps any historical replay window)
and cannot adjudicate v6+ candidates (the pre-holdout window is burned).
