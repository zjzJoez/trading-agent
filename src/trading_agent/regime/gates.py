"""Regime gates — translate a regime label into trade-flow rules.

Per §6.5 of plan v3:
    BULL_TREND          1.00x size, full long-call/long-put allowed
    RANGE_LOW_VOL       0.75x, mean-reversion only, IV penalty for long premium
    VOLATILE_TRANSITION 0.50x, A-quality only, mandatory LLM risk review
    BEAR_TREND          0.50x, no new long calls (except RS earnings drift)
    CRISIS              0.00x, no new risk-on entries; exits + hedges only

These multipliers are applied BEFORE R1–R6. R1–R6 remain absolute caps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from trading_agent.regime.classifier import RegimeLabel

Direction = Literal["LONG", "LONG_CALL", "LONG_PUT", "SHORT_CALL", "SHORT_PUT"]


@dataclass
class GateResult:
    """Output of `gate_trade_for_regime`."""

    allowed: bool
    size_multiplier: float
    require_llm_risk_review: bool
    reason: str
    block_reason: str | None = None


# Default per-regime multipliers. Online-learning §8 will tune these within
# bounded ranges.
SIZE_MULTIPLIERS: dict[RegimeLabel, float] = {
    "BULL_TREND": 1.00,
    "RANGE_LOW_VOL": 0.75,
    "VOLATILE_TRANSITION": 0.50,
    "BEAR_TREND": 0.50,
    "CRISIS": 0.00,
}

# Per-regime allowed directions for new entries.
ALLOWED_DIRECTIONS: dict[RegimeLabel, set[Direction]] = {
    "BULL_TREND": {"LONG", "LONG_CALL", "LONG_PUT"},
    "RANGE_LOW_VOL": {"LONG", "LONG_CALL"},     # Long premium needs catalyst — separate gate
    "VOLATILE_TRANSITION": {"LONG_CALL", "LONG_PUT"},  # only A-quality + LLM mandatory
    "BEAR_TREND": {"LONG_PUT"},                 # block new calls; allow puts
    "CRISIS": set(),                            # nothing new; exits + hedges only
}


def regime_size_multiplier(
    label: RegimeLabel,
    confidence: float = 1.0,
) -> float:
    """Get the size multiplier for a regime, with confidence-aware blending.

    Confidence-aware behavior: if confidence is borderline (0.55-0.65), blend
    halfway toward 0 (more conservative). At confidence ≥ 0.65 use full mult.
    """
    base = SIZE_MULTIPLIERS.get(label, 0.0)
    if confidence >= 0.65:
        return base
    # linear taper from 0.55 → 0.65 down to half of base
    frac = max(0.0, (confidence - 0.55) / 0.10)
    return base * (0.5 + 0.5 * frac)


def gate_trade_for_regime(
    proposal_direction: Direction,
    label: RegimeLabel,
    confidence: float,
    *,
    is_a_quality: bool = True,
    has_catalyst: bool = False,
    iv_percentile: float | None = None,
) -> GateResult:
    """Decide whether a proposed direction is allowed under the current regime.

    Inputs:
      proposal_direction: what the trader wants to do (LONG_CALL etc.)
      label, confidence:  current regime
      is_a_quality:       caller's flag — does the proposal pass quality bars?
                          (Trader sets True; Risk Agent can override.)
      has_catalyst:       earnings, filing, news — relevant for RANGE_LOW_VOL
                          long-premium gate.
      iv_percentile:      0..100 — used for RANGE_LOW_VOL long premium veto.
    """
    allowed_dirs = ALLOWED_DIRECTIONS.get(label, set())

    # CRISIS: blanket block on new entries
    if label == "CRISIS":
        return GateResult(
            allowed=False,
            size_multiplier=0.0,
            require_llm_risk_review=False,
            reason="regime=CRISIS — no new risk-on entries",
            block_reason="regime_crisis",
        )

    # Direction not allowed in this regime
    if proposal_direction not in allowed_dirs:
        return GateResult(
            allowed=False,
            size_multiplier=0.0,
            require_llm_risk_review=True,
            reason=f"direction={proposal_direction} blocked under regime={label}",
            block_reason="direction_blocked_by_regime",
        )

    # VOLATILE_TRANSITION: A-quality + mandatory LLM
    if label == "VOLATILE_TRANSITION":
        if not is_a_quality:
            return GateResult(
                allowed=False,
                size_multiplier=0.0,
                require_llm_risk_review=True,
                reason="regime=VOLATILE_TRANSITION requires A-quality setup",
                block_reason="quality_below_transition_bar",
            )
        return GateResult(
            allowed=True,
            size_multiplier=regime_size_multiplier(label, confidence),
            require_llm_risk_review=True,
            reason="VOLATILE_TRANSITION A-quality, half size, LLM risk required",
        )

    # RANGE_LOW_VOL: long premium needs catalyst + low IV
    if label == "RANGE_LOW_VOL" and proposal_direction in {"LONG_CALL", "LONG_PUT"}:
        if not has_catalyst:
            return GateResult(
                allowed=False,
                size_multiplier=0.0,
                require_llm_risk_review=False,
                reason="RANGE_LOW_VOL: long premium needs explicit catalyst",
                block_reason="no_catalyst_in_range_regime",
            )
        if iv_percentile is not None and iv_percentile > 50:
            return GateResult(
                allowed=False,
                size_multiplier=0.0,
                require_llm_risk_review=False,
                reason=f"RANGE_LOW_VOL: IV percentile {iv_percentile:.0f} > 50 cap",
                block_reason="iv_too_high_in_range",
            )

    # BEAR_TREND long puts are explicitly allowed; LLM review still smart
    require_llm = label in {"BEAR_TREND", "VOLATILE_TRANSITION"}

    return GateResult(
        allowed=True,
        size_multiplier=regime_size_multiplier(label, confidence),
        require_llm_risk_review=require_llm,
        reason=f"allowed under regime={label} at confidence={confidence:.2f}",
    )
