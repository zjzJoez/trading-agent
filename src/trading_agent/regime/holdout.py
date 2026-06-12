"""Holdout freeze for regime-model selection — the 2017-2026 window is burned.

WHY THIS MODULE EXISTS
----------------------
Every walk-forward evaluation v1-v5 scored itself on the SAME
2017 → 2026-06-12 OOS window. Each run was individually honest (no
training-window leakage), but the *design iteration loop* — change the
feature schema, the covariance type, the calibration rule, the inference
prior, then re-read the same OOS Sharpe — is implicit selection on that
window. After ~7 evaluation passes the headline (+0.071 Sharpe vs SPY,
~0.2 SE) cannot be read as unbiased evidence for any FUTURE design
choice made by looking at it.

The fix is the standard one: freeze a never-touched window. Data from
HOLDOUT_START forward is reserved for ONE final evaluation of whatever
v6+ candidate wins on the burned window. Until that evaluation, no
model-selection run may consume post-HOLDOUT_START snapshots.

The guard is code, not convention, because convention already failed:
v3's "+0.18 edge" was celebrated and cited before v4 showed it was a
broken-model artifact — on the same window we then kept reusing.

USAGE
-----
    from trading_agent.regime.holdout import require_holdout_intact
    require_holdout_intact(test_end)                     # raises if burned
    require_holdout_intact(test_end, break_glass=True)   # loud warning, proceeds
"""
from __future__ import annotations

import sys
from datetime import date, datetime

# Data from this date forward is the untouched evaluation window for v6+.
# Chosen as the day after the last evaluation run on the burned window
# (2026-06-12). Do not move this date backward, ever; moving it forward
# only wastes holdout data.
HOLDOUT_START = date(2026, 6, 13)

# Every model-selection evaluation that consumed the 2017 → 2026-06-12
# window. One line per run; dates and numbers from reports/. The count is
# the point: this window has been read at least seven times across design
# iterations, so any number computed on it is in-sample FOR SELECTION
# PURPOSES even though each individual run avoided train/test overlap.
EVALUATIONS_BURNED: tuple[str, ...] = (
    "v1   2026-05-16  single split (train ≤2023-12, test 2024-01→2026-05-15); "
    "broken EM (binary feats × full cov) — reports/walkforward_v1",
    "v2   2026-05-16  single split (train ≤2020-12, test 2021-01→2026-05-15); "
    "same broken setup — reports/walkforward_v2",
    "v3   2026-05-16  rolling annual 2017→2026; '+0.18 Sharpe edge' was a "
    "non-converged-model artifact — reports/walkforward_v3_rolling",
    "v4   2026-05-17  rolling 2017→2026, converged diag-cov, 5 seeds; "
    "Sharpe 0.78±0.015 vs SPY 0.84 — reports/walkforward_v4_converged",
    "v4.1 2026-05-17  4-regime auto-calibration; Sharpe 0.80±0.011 — "
    "reports/walkforward_v4_1_4regime",
    "v4.2 2026-05-19  hmm_predict stationary-prior fix; Sharpe 0.905±0.026 "
    "vs SPY 0.837 — reports/walkforward_v4_2_priorfix",
    "v5   2026-05-19  2007-2026 training history; Sharpe 0.908±0.029 vs "
    "SPY 0.837; promoted as id=5 — reports/walkforward_v5_longhistory",
)


class HoldoutViolation(RuntimeError):
    """A model-selection evaluation tried to consume post-HOLDOUT_START data."""


def _as_date(d: date | datetime | str) -> date:
    """Normalize date / datetime / ISO 'YYYY-MM-DD' to a date.

    Scripts pass --test-end as a string and computed window ends as
    date/datetime; the guard must not silently mis-compare types.
    """
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return date.fromisoformat(d.strip())
    raise TypeError(f"test_end_date must be date/datetime/ISO string, got {type(d)!r}")


def require_holdout_intact(
    test_end_date: date | datetime | str,
    break_glass: bool = False,
    *,
    context: str = "",
) -> None:
    """Refuse model-selection evaluations that reach into the holdout window.

    Call this BEFORE loading any snapshot data. `test_end_date` is the last
    day the evaluation will consume (forward-return horizons past it are
    tolerated — they leak ~20 trading days, which is unavoidable for any
    evaluation ending near the boundary and is why HOLDOUT_START is a hard
    date rather than a soft buffer).

    break_glass=True proceeds but prints a loud warning: the holdout is then
    BURNED for v6+ selection and a new freeze date must be chosen. There is
    deliberately no env-var or config override — the flag must appear in the
    command line so it shows up in shell history and run logs.
    """
    end = _as_date(test_end_date)
    if end < HOLDOUT_START:
        return
    where = f" [{context}]" if context else ""
    if break_glass:
        print(
            "\n" + "!" * 72 + "\n"
            f"!! BREAK GLASS{where}: evaluating through {end.isoformat()}, which\n"
            f"!! reaches into the frozen holdout window (>= {HOLDOUT_START.isoformat()}).\n"
            "!! The holdout is now BURNED for v6+ model selection. Any model\n"
            "!! chosen after reading this result is selected in-sample on this\n"
            "!! window. Pick a new HOLDOUT_START in regime/holdout.py and append\n"
            "!! this run to EVALUATIONS_BURNED.\n"
            + "!" * 72 + "\n",
            file=sys.stderr,
        )
        return
    raise HoldoutViolation(
        f"Evaluation through {end.isoformat()} would consume holdout data "
        f"(frozen from {HOLDOUT_START.isoformat()}){where}. The 2017 -> "
        f"2026-06-12 window is already burned by {len(EVALUATIONS_BURNED)} "
        "evaluation passes (see regime/holdout.py EVALUATIONS_BURNED); the "
        "holdout exists so v6+ gets ONE untouched test. End the evaluation "
        f"before {HOLDOUT_START.isoformat()}, or pass --break-glass to "
        "knowingly burn the holdout."
    )
