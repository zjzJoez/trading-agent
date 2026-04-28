"""Phase 2.7 follow-up — drawdown helpers extracted from archive/promote/replay."""
from __future__ import annotations

import math

from trading_agent.learning._stats import (
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
