"""Phase 2.7 — promotion gate math (Wilson LB + aggregate stats)."""
from __future__ import annotations

import math
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from trading_agent.learning.promote import (
    MDD_FRAC,
    PROFIT_FACTOR_FRAC,
    _aggregate,
    evaluate_canary,
    wilson_lb,
)


# -- Wilson interval ---------------------------------------------------------

def test_wilson_lb_zero_n():
    assert wilson_lb(0, 0) == 0.0


def test_wilson_lb_all_wins_low_n():
    # 5/5 wins: Wilson 95% LB ≈ 0.566 (textbook value)
    lb = wilson_lb(5, 5)
    assert 0.55 < lb < 0.60


def test_wilson_lb_all_wins_high_n():
    # 100/100: LB ≈ 0.964
    lb = wilson_lb(100, 100)
    assert 0.96 < lb < 0.97


def test_wilson_lb_50_50():
    # 50/100 wins: LB ≈ 0.402
    lb = wilson_lb(50, 100)
    assert 0.39 < lb < 0.42


def test_wilson_lb_monotone_in_n():
    """Same proportion 0.6 — Wilson LB rises with n (more confidence)."""
    a = wilson_lb(6, 10)
    b = wilson_lb(60, 100)
    c = wilson_lb(600, 1000)
    assert a < b < c


def test_wilson_lb_monotone_in_p():
    """Same n — higher proportion gives higher LB."""
    n = 50
    assert wilson_lb(20, n) < wilson_lb(30, n) < wilson_lb(40, n)


# -- Aggregate stats ---------------------------------------------------------

def test_aggregate_empty():
    s = _aggregate([])
    assert s.n == 0 and s.win_rate == 0 and s.profit_factor == 0


def test_aggregate_pure_winner():
    s = _aggregate([1.0, 1.5, 2.0])
    assert s.n == 3 and s.wins == 3 and s.losses == 0
    assert s.win_rate == 1.0
    assert math.isinf(s.profit_factor)
    assert s.cum_R == 4.5
    assert s.max_drawdown_R == 0


def test_aggregate_mixed():
    s = _aggregate([1.0, -0.5, 1.0, -0.5, 2.0])
    assert s.n == 5 and s.wins == 3 and s.losses == 2
    # gross_win = 4.0, gross_loss = 1.0 → profit factor = 4.0
    assert math.isclose(s.profit_factor, 4.0, rel_tol=1e-6)
    # cum: 1, 0.5, 1.5, 1.0, 3.0 → peak 3 then 0 dip; mdd = 0.5 (peak 1 then 0.5)
    assert s.max_drawdown_R == 0.5


def test_aggregate_drawdown_after_recovery():
    # cum: 1, 0.5, 0.0, 1.5 → peak 1 → mdd 1.0 (1 → 0)
    s = _aggregate([1.0, -0.5, -0.5, 1.5])
    assert s.max_drawdown_R == 1.0


# -- Gate constants ----------------------------------------------------------

def test_promotion_thresholds_documented():
    assert PROFIT_FACTOR_FRAC == 0.95
    assert MDD_FRAC == 1.10


# -- evaluate_canary fail-safe evidence gate ----------------------------------

class _FakeLearningDB:
    """Scripted stand-in for the Postgres cursor used by evaluate_canary.

    Dispatches on the SQL of the last execute() so one object can serve every
    query the promotion path issues: canary status, rollback triggers,
    routed-trade count, and per-version outcome series.
    """

    def __init__(self, *, status_row=("CANARY", 1), n_routed=25,
                 outcomes_by_version=None, daily_rows=None):
        self.status_row = status_row
        self.n_routed = n_routed
        self.outcomes_by_version = outcomes_by_version or {}
        self.daily_rows = daily_rows or []
        self._sql = ""
        self._params = ()

    def execute(self, sql, params=()):
        self._sql = sql
        self._params = params or ()

    def fetchone(self):
        # Counterfactual dispatch row (ITEM 11) — None means "row not
        # found" so maybe_evaluate_qty_only falls through to the
        # traffic-split path these tests exercise.
        if "params, created_at" in self._sql:
            return None
        if "SELECT status, parent_version_id" in self._sql:
            return self.status_row
        if "count(*)" in self._sql.lower():
            return (self.n_routed,)
        return None

    def fetchall(self):
        if "GROUP BY DATE" in self._sql:
            return list(self.daily_rows)
        if "trade_outcome_features" in self._sql:
            rs = [(r,) for r in self.outcomes_by_version.get(self._params[0], [])]
            if "LIMIT" in self._sql:
                return rs[: int(self._params[1])]
            return rs
        return []


@contextmanager
def _patched_db(db: _FakeLearningDB):
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=db)
    ctx.__exit__ = MagicMock(return_value=False)
    with patch("trading_agent.learning.promote.cursor", return_value=ctx), \
         patch("trading_agent.learning.canary.cursor", return_value=ctx), \
         patch("trading_agent.learning.counterfactual.cursor", return_value=ctx):
        yield


def test_evaluate_canary_empty_outcomes_retains_never_promotes():
    """Regression: 20+ routed journal rows but an EMPTY trade_outcome_features
    table must RETAIN.  Before the evidence gate, empty aggregates passed all
    three comparisons vacuously (0 >= 0, 0 >= 0×0.95, 0 <= 0×1.1) and the
    first canary to accumulate 20 routed rows auto-promoted on zero data."""
    db = _FakeLearningDB(status_row=("CANARY", 1), n_routed=25,
                         outcomes_by_version={})
    with _patched_db(db):
        d = evaluate_canary(42)
    assert d.decision == "RETAIN"
    assert "insufficient closed-outcome evidence" in d.rationale[0]
    assert d.canary_stats is not None and d.canary_stats.n == 0
    assert d.control_stats is not None and d.control_stats.n == 0


def test_evaluate_canary_thin_control_retains():
    """Canary has 20 closed outcomes but control only 3 — still RETAIN.
    Both sides need >= N_CANARY_MIN before any gate comparison runs."""
    db = _FakeLearningDB(
        status_row=("CANARY", 1), n_routed=25,
        outcomes_by_version={
            42: [1.0] * 16 + [-0.5] * 4,
            1: [1.0, -1.0, 0.5],
        },
    )
    with _patched_db(db):
        d = evaluate_canary(42)
    assert d.decision == "RETAIN"
    assert "insufficient closed-outcome evidence" in d.rationale[0]


def test_evaluate_canary_promotes_with_full_evidence_on_both_sides():
    """The evidence gate must not block a genuinely better canary: 20 closed
    outcomes each side, canary clearly stronger on all three gates."""
    db = _FakeLearningDB(
        status_row=("CANARY", 1), n_routed=25,
        outcomes_by_version={
            42: [1.0, 1.0, 1.0, 1.0, -0.5] * 4,   # wr .8, pf 8, mdd .5
            1: [1.0, -1.0] * 10,                  # wr .5, pf 1, mdd 1
        },
    )
    with _patched_db(db):
        d = evaluate_canary(42)
    assert d.decision == "PROMOTE"


def test_evaluate_canary_non_canary_status_retains():
    db = _FakeLearningDB(status_row=("ACTIVE", None))
    with _patched_db(db):
        d = evaluate_canary(42)
    assert d.decision == "RETAIN"
