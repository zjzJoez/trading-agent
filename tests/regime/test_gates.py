"""Tests for regime gates — direction allowance + size multipliers."""
from __future__ import annotations

from trading_agent.regime.gates import (
    SIZE_MULTIPLIERS,
    gate_trade_for_regime,
    regime_size_multiplier,
)


def test_size_multiplier_full_at_high_confidence():
    assert regime_size_multiplier("BULL_TREND", confidence=0.9) == 1.0
    assert regime_size_multiplier("RANGE_LOW_VOL", confidence=0.8) == 0.75
    assert regime_size_multiplier("VOLATILE_TRANSITION", confidence=0.7) == 0.5
    assert regime_size_multiplier("BEAR_TREND", confidence=0.65) == 0.5
    assert regime_size_multiplier("CRISIS", confidence=1.0) == 0.0


def test_size_multiplier_tapers_below_065():
    """At confidence=0.55, multiplier should be half of base; smooth between."""
    assert regime_size_multiplier("BULL_TREND", confidence=0.55) == 0.5
    # Halfway in the taper range (0.6) → 0.75 of base
    assert abs(regime_size_multiplier("BULL_TREND", confidence=0.60) - 0.75) < 0.01


def test_crisis_blocks_all_new_entries():
    res = gate_trade_for_regime("LONG_CALL", "CRISIS", confidence=1.0)
    assert not res.allowed
    assert res.size_multiplier == 0.0
    assert res.block_reason == "regime_crisis"


def test_bear_blocks_long_calls_allows_long_puts():
    """Per §6.5 — BEAR_TREND blocks new long calls; allows long puts."""
    call = gate_trade_for_regime("LONG_CALL", "BEAR_TREND", confidence=0.8)
    assert not call.allowed
    assert call.block_reason == "direction_blocked_by_regime"

    put = gate_trade_for_regime("LONG_PUT", "BEAR_TREND", confidence=0.8)
    assert put.allowed
    assert put.size_multiplier == 0.5


def test_volatile_transition_requires_a_quality():
    """VOLATILE_TRANSITION blocks non-A-quality + always asks for LLM review."""
    blocked = gate_trade_for_regime(
        "LONG_CALL", "VOLATILE_TRANSITION", confidence=0.65, is_a_quality=False
    )
    assert not blocked.allowed
    assert blocked.block_reason == "quality_below_transition_bar"

    allowed = gate_trade_for_regime(
        "LONG_CALL", "VOLATILE_TRANSITION", confidence=0.65, is_a_quality=True
    )
    assert allowed.allowed
    assert allowed.require_llm_risk_review
    assert allowed.size_multiplier == 0.5


def test_range_low_vol_requires_catalyst_for_long_premium():
    no_cat = gate_trade_for_regime(
        "LONG_CALL", "RANGE_LOW_VOL", confidence=0.8, has_catalyst=False
    )
    assert not no_cat.allowed
    assert no_cat.block_reason == "no_catalyst_in_range_regime"

    with_cat = gate_trade_for_regime(
        "LONG_CALL",
        "RANGE_LOW_VOL",
        confidence=0.8,
        has_catalyst=True,
        iv_percentile=30,
    )
    assert with_cat.allowed


def test_range_low_vol_blocks_high_iv():
    """Long premium in RANGE_LOW_VOL also blocked when IV percentile > 50."""
    res = gate_trade_for_regime(
        "LONG_CALL",
        "RANGE_LOW_VOL",
        confidence=0.8,
        has_catalyst=True,
        iv_percentile=70,
    )
    assert not res.allowed
    assert res.block_reason == "iv_too_high_in_range"


def test_bull_trend_full_size_no_review():
    res = gate_trade_for_regime("LONG_CALL", "BULL_TREND", confidence=0.9)
    assert res.allowed
    assert res.size_multiplier == 1.0
    # BULL_TREND high confidence shouldn't force LLM review
    assert not res.require_llm_risk_review


def test_all_size_multipliers_in_valid_range():
    """Belt-and-braces: every regime multiplier ∈ [0, 1]."""
    for label, m in SIZE_MULTIPLIERS.items():
        assert 0.0 <= m <= 1.0, f"{label} multiplier {m} out of range"
