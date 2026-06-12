"""Phase 2.7 follow-up — drawdown helpers extracted from archive/promote/replay.

Plus the item-10 expectancy statistic ``lcb_mean`` (one-sided Student-t
lower bound on the mean) the promotion gate is keyed on.
"""
from __future__ import annotations

import math

import pytest

from trading_agent.learning._stats import (
    lcb_mean,
    running_drawdown_step,
    series_max_drawdown,
)


def test_running_step_first_winner_initializes_peak():
    peak, mdd = running_drawdown_step(0.0, 0.0, 1.0)
    assert peak == 1.0 and mdd == 0.0


def test_running_step_dip_increases_mdd():
    peak, mdd = running_drawdown_step(0.0, 0.0, 1.0)
    peak, mdd = running_drawdown_step(peak, mdd, 0.5)
    assert peak == 1.0
    assert math.isclose(mdd, 0.5)


def test_running_step_recovery_keeps_mdd():
    peak, mdd = running_drawdown_step(0.0, 0.0, 1.0)
    peak, mdd = running_drawdown_step(peak, mdd, 0.5)
    peak, mdd = running_drawdown_step(peak, mdd, 0.8)
    assert peak == 1.0
    assert math.isclose(mdd, 0.5)


def test_running_step_new_peak_does_not_reset_mdd():
    peak, mdd = running_drawdown_step(0.0, 0.0, 1.0)
    peak, mdd = running_drawdown_step(peak, mdd, 0.5)
    peak, mdd = running_drawdown_step(peak, mdd, 2.0)
    assert peak == 2.0
    assert math.isclose(mdd, 0.5)  # historical drawdown remains


def test_series_empty_is_zero():
    assert series_max_drawdown([]) == (0.0, 0.0)


def test_series_known_curve():
    # rs = [1.0, -0.5, -0.5, 1.5] → cum {1, 0.5, 0, 1.5}; peak=1, mdd=1.0
    cum, mdd = series_max_drawdown([1.0, -0.5, -0.5, 1.5])
    assert math.isclose(cum, 1.5)
    assert math.isclose(mdd, 1.0)


def test_series_pure_winner_no_drawdown():
    cum, mdd = series_max_drawdown([0.5, 1.0, 0.5])
    assert math.isclose(cum, 2.0)
    assert mdd == 0.0


def test_series_matches_running_step_walk():
    """Reference: applying running_drawdown_step in a loop must match
    series_max_drawdown's output exactly."""
    rs = [1.5, -0.5, -1.0, 0.8, 2.0, -1.5]
    expected_cum, expected_mdd = series_max_drawdown(rs)
    cum = 0.0
    peak = 0.0
    mdd = 0.0
    for r in rs:
        cum += r
        peak, mdd = running_drawdown_step(peak, mdd, cum)
    assert math.isclose(cum, expected_cum)
    assert math.isclose(mdd, expected_mdd)


# -- lcb_mean (item 10 — expectancy lower bound) ------------------------------


def test_lcb_mean_constant_series_returns_mean():
    """Zero spread → zero margin: the bound IS the mean."""
    assert lcb_mean([0.5] * 10) == 0.5
    assert lcb_mean([-1.0, -1.0, -1.0]) == -1.0


def test_lcb_mean_no_evidence_is_none_not_zero():
    """n < 2 → None: a single sample has df=0 and NO defensible bound.
    Returning 0 here would let a one-trade canary look exactly breakeven."""
    assert lcb_mean([]) is None
    assert lcb_mean([3.0]) is None


def test_lcb_mean_known_textbook_value():
    # [1..5]: mean 3, s = sqrt(2.5), n=5, one-sided t(df=4, 95%) = 2.132
    expected = 3.0 - 2.132 * math.sqrt(2.5) / math.sqrt(5)
    assert lcb_mean([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(
        expected, abs=1e-9)


def test_lcb_mean_symmetric_series_sanity():
    """Symmetric around 0: mean 0, bound strictly negative by exactly the
    t-margin (same spread as the [1..5] series, shifted)."""
    vs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    lb = lcb_mean(vs)
    assert lb == pytest.approx(-2.132 * math.sqrt(2.5) / math.sqrt(5),
                               abs=1e-9)
    assert lb < 0


def test_lcb_mean_tightens_with_n():
    """Same distribution, more samples → bound closes toward the mean."""
    base = [1.0, -1.0]
    a = lcb_mean(base * 5)     # n=10
    b = lcb_mean(base * 50)    # n=100 (z branch)
    assert a < b < 0           # mean is 0; both bounds below it


def test_lcb_mean_df_above_30_uses_z():
    # n=100 alternating ±1: mean 0, s = sqrt(100/99); z(95%) = 1.6449
    vs = [1.0, -1.0] * 50
    expected = -1.6449 * math.sqrt(100.0 / 99.0) / 10.0
    assert lcb_mean(vs) == pytest.approx(expected, abs=1e-9)


def test_lcb_mean_99_is_stricter_than_95():
    vs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert lcb_mean(vs, confidence=0.99) < lcb_mean(vs, confidence=0.95)


def test_lcb_mean_unsupported_confidence_raises():
    """No silent interpolation between tabled confidences."""
    with pytest.raises(ValueError):
        lcb_mean([1.0, 2.0, 3.0], confidence=0.90)
