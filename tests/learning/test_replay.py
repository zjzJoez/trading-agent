"""Phase 2.6 — replay aggregate metrics."""
from __future__ import annotations

import math

from trading_agent.learning.replay import _aggregate, _safe_div


def test_safe_div_default_when_denom_zero():
    assert _safe_div(1.0, 0.0, default=99.0) == 99.0
    assert _safe_div(1.0, 2.0) == 0.5


def test_aggregate_empty():
    m = _aggregate([])
    assert m.n_trades == 0
    assert m.composite_score == 0


def test_aggregate_known_series():
    """Hand-checked: [+1, -0.5, +0.5, -1, +2] → 5 trades."""
    rs = [1.0, -0.5, 0.5, -1.0, 2.0]
    m = _aggregate(rs)
    assert m.n_trades == 5
    assert math.isclose(m.win_rate, 0.6)
    assert math.isclose(m.mean_R, 0.4)
    # gross_win=3.5, gross_loss=1.5 → PF≈2.333
    assert math.isclose(m.profit_factor, 3.5 / 1.5, rel_tol=1e-3)
    # cum: 1, 0.5, 1.0, 0.0, 2.0 → peak=1.0 then dips to 0 → mdd=1.0
    assert math.isclose(m.max_drawdown_R, 1.0)


def test_aggregate_pure_winner():
    rs = [1.0, 1.5, 2.0]
    m = _aggregate(rs)
    assert m.win_rate == 1.0
    # No losses → profit_factor returns gross_win as default
    assert m.profit_factor > 0


def test_information_ratio_zero_when_baseline_matches():
    rs = [1.0, -0.5, 0.5]
    m = _aggregate(rs, baseline_rs=list(rs))
    # diff identically zero → IR ≈ 0
    assert m.composite_score == m.sharpe + 0 - 0.5 * m.max_drawdown_R


def test_composite_penalizes_drawdown():
    """Holding mean fixed and variance comparable, the series with a real
    drawdown should score lower — drawdown carries a -λ penalty."""
    # Both series: mean=1.0, n=3
    # a — cum 2.0, 2.5, 3.0 → MDD = 0
    a = _aggregate([2.0, 0.5, 0.5])
    # b — cum 2.0, 0.5, 3.0 → MDD = 1.5
    b = _aggregate([2.0, -1.5, 2.5])
    assert a.max_drawdown_R == 0
    assert b.max_drawdown_R > 0
    assert a.composite_score > b.composite_score
