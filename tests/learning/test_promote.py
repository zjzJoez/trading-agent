"""Phase 2.7 — promotion gate math (Wilson LB + aggregate stats)."""
from __future__ import annotations

import math

from trading_agent.learning.promote import (
    MDD_FRAC,
    PROFIT_FACTOR_FRAC,
    _aggregate,
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
