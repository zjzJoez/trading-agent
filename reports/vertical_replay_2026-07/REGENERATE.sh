#!/usr/bin/env bash
# Regenerate every artifact in this directory from the committed inputs.
#
#   bash reports/vertical_replay_2026-07/REGENERATE.sh [CALIBRATION_JSON]
#
# Run from the repository root. All four runs are deterministic per seed, so a
# clean checkout must reproduce the committed files BYTE FOR BYTE — that is the
# check, not a vague "should match".
#
# The one input that is NOT committed is the friction calibration file
# (data/execution_costs.json is gitignored). Pass its path as $1; without it
# the engine falls back to the compiled-in DEFAULT_SPREAD_PCT, shouts about it,
# and the summary's `friction_under_calibration_file` section is omitted. The
# sha256 of the file used for the committed artifacts is recorded in
# summary.json .config.spread_pct_provenance.sha256, so the version is pinned
# even though the bytes are not in the repo.
set -euo pipefail

DIR="reports/vertical_replay_2026-07"
IN="$DIR/inputs"
PLAN="$IN/batch_plan.json"
CALIB="${1:-data/execution_costs.json}"
PY="${PYTHON:-python3}"
SPREADS="SPY=0.0049,QQQ=0.0094"   # M1-0.3 calibration RUN, not the file

# 1. PRIMARY — smile marks, prior-session liquidity screen, spec floor 0.25.
#    Also writes sensitivity/entries_*.jsonl, one full replay per convention.
"$PY" scripts/replay_vertical_gates.py \
  --data-dir "$IN" --batch-plan "$PLAN" --out-dir "$DIR" \
  --spread-pct "$SPREADS" --calibration "$CALIB"

# 2. DISCLOSED UPPER BOUND — close marks, same-day liquidity screen.
"$PY" scripts/replay_vertical_gates.py \
  --data-dir "$IN" --batch-plan "$PLAN" --out-dir "$DIR/as_reported" \
  --spread-pct "$SPREADS" --calibration "$CALIB" \
  --mark close --liquidity-lag same-day --no-sensitivity

# 3. Adversarial diagnostics for both samples. ORDER MATTERS only for the key
#    order inside diagnostics.json; the upper bound is emitted first.
"$PY" scripts/replay_vertical_diagnostics.py \
  --data-dir "$IN" --batch-plan "$PLAN" \
  --entries "$DIR/as_reported/entries.jsonl" --label as_reported_close_sameday \
  --entries "$DIR/entries.jsonl" --label primary_smile_prior \
  --out "$DIR/diagnostics.json"

# 4. Pin every quantitative claim in REPORT.md to the artifact row it came
#    from. Per-claim tuple match, not global value membership.
"$PY" scripts/verify_replay_report.py --report-dir "$DIR"
