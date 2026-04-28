"""Phase 2.6.5 — diversity archive: cell math + composite score."""
from __future__ import annotations

import math

from trading_agent.learning.archive import (
    LAMBDA_MDD,
    NA,
    CellKey,
    _composite_from_running,
    _next_drawdown,
    bucket_dte,
    bucket_iv,
    cell_for_trade,
)


# -- Cell taxonomy -----------------------------------------------------------

def test_cell_key_round_trip():
    c = CellKey("BULL_TREND", "trend_continuation_atm_call", "7-14", "mid")
    assert CellKey.from_text(c.to_text()) == c


def test_dte_bucket_boundaries():
    assert bucket_dte(0) == "0-7"
    assert bucket_dte(6) == "0-7"
    assert bucket_dte(7) == "7-14"
    assert bucket_dte(13) == "7-14"
    assert bucket_dte(14) == "14-30"
    assert bucket_dte(29) == "14-30"
    assert bucket_dte(30) == "30-60"
    assert bucket_dte(59) == "30-60"
    assert bucket_dte(60) == "60+"
    assert bucket_dte(180) == "60+"


def test_dte_bucket_handles_missing():
    assert bucket_dte(None) == NA
    assert bucket_dte("not-a-number") == NA
    assert bucket_dte(-1) == NA


def test_iv_bucket_boundaries():
    assert bucket_iv(0.10) == "low"
    assert bucket_iv(0.24) == "low"
    assert bucket_iv(0.25) == "mid"
    assert bucket_iv(0.39) == "mid"
    assert bucket_iv(0.40) == "high"
    assert bucket_iv(2.0) == "high"


def test_iv_bucket_handles_missing():
    assert bucket_iv(None) == NA
    assert bucket_iv(-0.05) == NA
    assert bucket_iv(float("nan")) == NA


def test_cell_for_trade_with_full_data():
    c = cell_for_trade(
        regime_label="BULL_TREND",
        strategy_label="trend_continuation_atm_call",
        option_dte=10,
        option_iv=0.32,
    )
    assert c == CellKey("BULL_TREND", "trend_continuation_atm_call", "7-14", "mid")


def test_cell_for_trade_with_stk_collapses_options_dims_to_na():
    c = cell_for_trade(
        regime_label="RANGE_LOW_VOL",
        strategy_label="mean_reversion",
        option_dte=None,
        option_iv=None,
    )
    assert c.dte_bucket == NA
    assert c.iv_bucket == NA
    assert c.regime == "RANGE_LOW_VOL"


def test_cell_for_trade_missing_regime_falls_back_to_na():
    c = cell_for_trade(
        regime_label=None, strategy_label="anything",
        option_dte=10, option_iv=0.30,
    )
    assert c.regime == NA


# -- Drawdown & composite ----------------------------------------------------

def test_drawdown_increments_on_dip():
    # cumulative path: 1.0, 0.5, 0.8 — peak=1.0, mdd=0.5 then 0.5 (still 0.5)
    peak, mdd = _next_drawdown(0.0, 0.0, 1.0)
    assert (peak, mdd) == (1.0, 0.0)
    peak, mdd = _next_drawdown(peak, mdd, 0.5)
    assert peak == 1.0 and mdd == 0.5
    peak, mdd = _next_drawdown(peak, mdd, 0.8)
    assert peak == 1.0 and abs(mdd - 0.5) < 1e-9


def test_drawdown_resets_on_new_peak():
    peak, mdd = _next_drawdown(0.0, 0.0, 1.0)
    peak, mdd = _next_drawdown(peak, mdd, 2.0)
    assert peak == 2.0 and mdd == 0.0


def test_composite_none_when_under_two_trades():
    assert _composite_from_running(0, 0, 0, 0) is None
    assert _composite_from_running(1, 1.0, 1.0, 0) is None


def test_composite_known_series():
    # R = [+1, +1] → mean=1, var=0 → degraded to mean − λ·MDD
    s = _composite_from_running(2, 2.0, 2.0, 0.0)
    assert s == 1.0  # mean=1, mdd=0


def test_composite_penalizes_drawdown():
    # Same mean, different MDD → smaller composite
    a = _composite_from_running(3, 3.0, 3.0, 0.0)   # mean=1, var=0
    b = _composite_from_running(3, 3.0, 3.0, 1.5)   # mean=1, var=0, MDD=1.5
    assert a is not None and b is not None
    assert a > b
    assert math.isclose(a - b, LAMBDA_MDD * 1.5, rel_tol=1e-9)
