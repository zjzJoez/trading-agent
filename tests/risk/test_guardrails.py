"""Synthetic-portfolio tests for Active Risk Agent guardrails.

Each test fixture is a minimal portfolio + proposal scenario. No Postgres
or moomoo dependency — pure Python.
"""
from __future__ import annotations

import pytest

from trading_agent.risk.guardrails import (
    REGIME_GUARDRAILS,
    check_guardrails,
)


def _greeks(**kwargs):
    """Make a minimal aggregate_greeks dict (all 0s except overrides)."""
    base = {
        "net_delta": 0,
        "net_gamma": 0,
        "net_vega": 0,
        "net_theta": 0,
        "delta_dollar": 0,
        "vega_10vol_pnl": 0,
        "gamma_2sigma_pnl": 0,
        "vega_10vol_pct_equity": 0,
        "gamma_2sigma_pct_equity": 0,
        "delta_dollar_pct_equity": 0,
    }
    base.update(kwargs)
    return base


def _heat(**kwargs):
    base = {"heat_pct": 0.0, "max_per_underlying_pct": 0.0}
    base.update(kwargs)
    return base


def _dq(level=0):
    return {"missing": [], "stale": [], "outlier": [], "degradation_level": level}


# ---------------------------------------------------------------------------
# Real-money marker → VETO (always wins)
# ---------------------------------------------------------------------------


def test_real_trading_marker_always_vetos():
    res = check_guardrails(
        regime="BULL_TREND",
        proposal_after_sizing={},
        portfolio_aggregate_greeks=_greeks(),
        portfolio_factor_exposures={},
        portfolio_correlations={},
        heat_metrics=_heat(),
        data_quality=_dq(),
        real_trading_marker=True,
    )
    assert res.decision == "VETO"
    assert "real_trading_marker_blocked" in res.reasons


# ---------------------------------------------------------------------------
# Data quality critical → DEFER
# ---------------------------------------------------------------------------


def test_critical_data_missing_defers():
    res = check_guardrails(
        regime="BULL_TREND",
        proposal_after_sizing={},
        portfolio_aggregate_greeks=_greeks(),
        portfolio_factor_exposures={},
        portfolio_correlations={},
        heat_metrics=_heat(),
        data_quality=_dq(level=2),
    )
    assert res.decision == "DEFER"


# ---------------------------------------------------------------------------
# Regime CRISIS → VETO new entries
# ---------------------------------------------------------------------------


def test_crisis_regime_vetos_new_entries():
    res = check_guardrails(
        regime="CRISIS",
        proposal_after_sizing={},
        portfolio_aggregate_greeks=_greeks(),
        portfolio_factor_exposures={},
        portfolio_correlations={},
        heat_metrics=_heat(),
        data_quality=_dq(),
    )
    assert res.decision == "VETO"
    assert "regime_blocks_new_entries" in res.reasons


# ---------------------------------------------------------------------------
# Heat over cap → DOWNSIZE if mild, VETO if extreme
# ---------------------------------------------------------------------------


def test_heat_mild_breach_downsizes():
    """heat 7% in BULL (cap 6%) → DOWNSIZE (within 1.5×)."""
    res = check_guardrails(
        regime="BULL_TREND",
        proposal_after_sizing={},
        portfolio_aggregate_greeks=_greeks(),
        portfolio_factor_exposures={},
        portfolio_correlations={},
        heat_metrics=_heat(heat_pct=0.07),
        data_quality=_dq(),
    )
    assert res.decision == "DOWNSIZE"
    assert any("heat_pct_max" in v.rule for v in res.violations)
    assert res.suggested_qty_factor < 1.0


def test_heat_extreme_breach_vetos():
    """heat 12% in BULL (cap 6%) → VETO (>1.5× = 9%)."""
    res = check_guardrails(
        regime="BULL_TREND",
        proposal_after_sizing={},
        portfolio_aggregate_greeks=_greeks(),
        portfolio_factor_exposures={},
        portfolio_correlations={},
        heat_metrics=_heat(heat_pct=0.12),
        data_quality=_dq(),
    )
    assert res.decision == "VETO"


# ---------------------------------------------------------------------------
# Single-underlying delta-$ over cap
# ---------------------------------------------------------------------------


def test_single_underlying_breach_downsizes():
    res = check_guardrails(
        regime="BULL_TREND",
        proposal_after_sizing={},
        portfolio_aggregate_greeks=_greeks(),
        portfolio_factor_exposures={},
        portfolio_correlations={},
        heat_metrics=_heat(max_per_underlying_pct=0.18),  # cap 0.15
        data_quality=_dq(),
    )
    assert res.decision == "DOWNSIZE"


# ---------------------------------------------------------------------------
# Net delta + vega + gamma stress
# ---------------------------------------------------------------------------


def test_net_delta_breach_downsizes():
    res = check_guardrails(
        regime="BULL_TREND",
        proposal_after_sizing={},
        portfolio_aggregate_greeks=_greeks(delta_dollar_pct_equity=0.40),  # cap 0.35
        portfolio_factor_exposures={},
        portfolio_correlations={},
        heat_metrics=_heat(),
        data_quality=_dq(),
    )
    assert res.decision == "DOWNSIZE"


def test_vega_shock_extreme_vetos():
    """Vega 10vol at 2% (cap 1%) → 2× = boundary → VETO."""
    res = check_guardrails(
        regime="BULL_TREND",
        proposal_after_sizing={},
        portfolio_aggregate_greeks=_greeks(vega_10vol_pct_equity=0.02),
        portfolio_factor_exposures={},
        portfolio_correlations={},
        heat_metrics=_heat(),
        data_quality=_dq(),
    )
    assert res.decision == "VETO"


# ---------------------------------------------------------------------------
# Tiered limits: TRANSITION/BEAR have stricter caps
# ---------------------------------------------------------------------------


def test_transition_regime_stricter_heat_cap():
    """heat 4% passes BULL (cap 6%) but breaches TRANSITION (cap 3%)."""
    bull = check_guardrails(
        regime="BULL_TREND",
        proposal_after_sizing={},
        portfolio_aggregate_greeks=_greeks(),
        portfolio_factor_exposures={},
        portfolio_correlations={},
        heat_metrics=_heat(heat_pct=0.04),
        data_quality=_dq(),
    )
    assert bull.decision == "APPROVE"

    transition = check_guardrails(
        regime="VOLATILE_TRANSITION",
        proposal_after_sizing={},
        portfolio_aggregate_greeks=_greeks(),
        portfolio_factor_exposures={},
        portfolio_correlations={},
        heat_metrics=_heat(heat_pct=0.04),
        data_quality=_dq(),
    )
    assert transition.decision == "DOWNSIZE"


# ---------------------------------------------------------------------------
# Factor exposure caps
# ---------------------------------------------------------------------------


def test_factor_exposure_breach():
    res = check_guardrails(
        regime="BULL_TREND",
        proposal_after_sizing={},
        portfolio_aggregate_greeks=_greeks(),
        portfolio_factor_exposures={"tech": 0.40},  # cap 0.30
        portfolio_correlations={},
        heat_metrics=_heat(),
        data_quality=_dq(),
    )
    assert res.decision == "DOWNSIZE"
    assert any("tech" in v.rule for v in res.violations)


# ---------------------------------------------------------------------------
# Same-expiry premium concentration (event clustering)
# ---------------------------------------------------------------------------


def test_same_expiry_premium_mild_breach_downsizes():
    """2.5% in one expiry vs BULL cap 2% → 1.25× → DOWNSIZE."""
    res = check_guardrails(
        regime="BULL_TREND",
        proposal_after_sizing={},
        portfolio_aggregate_greeks=_greeks(),
        portfolio_factor_exposures={},
        portfolio_correlations={},
        heat_metrics=_heat(max_per_expiry_pct=0.025),  # cap 0.02
        data_quality=_dq(),
    )
    assert res.decision == "DOWNSIZE"
    assert any(v.rule == "same_expiry_premium_max" for v in res.violations)
    assert res.suggested_qty_factor < 1.0


def test_same_expiry_premium_extreme_vetos():
    """4% in one expiry vs BULL cap 2% → 2× > 1.5 → VETO."""
    res = check_guardrails(
        regime="BULL_TREND",
        proposal_after_sizing={},
        portfolio_aggregate_greeks=_greeks(),
        portfolio_factor_exposures={},
        portfolio_correlations={},
        heat_metrics=_heat(max_per_expiry_pct=0.04),  # cap 0.02
        data_quality=_dq(),
    )
    assert res.decision == "VETO"
    assert any(v.rule == "same_expiry_premium_max" for v in res.violations)


def test_same_expiry_premium_under_cap_approves():
    """1.5% in one expiry is under the BULL cap 2% → no same-expiry violation."""
    res = check_guardrails(
        regime="BULL_TREND",
        proposal_after_sizing={},
        portfolio_aggregate_greeks=_greeks(),
        portfolio_factor_exposures={},
        portfolio_correlations={},
        heat_metrics=_heat(max_per_expiry_pct=0.015),  # cap 0.02
        data_quality=_dq(),
    )
    assert res.decision == "APPROVE"
    assert not any(v.rule == "same_expiry_premium_max" for v in res.violations)


def test_same_expiry_premium_stricter_in_bear():
    """1.5% passes BULL (cap 2%) but breaches BEAR (cap 1%)."""
    bull = check_guardrails(
        regime="BULL_TREND", proposal_after_sizing={},
        portfolio_aggregate_greeks=_greeks(), portfolio_factor_exposures={},
        portfolio_correlations={}, heat_metrics=_heat(max_per_expiry_pct=0.015),
        data_quality=_dq(),
    )
    assert bull.decision == "APPROVE"
    bear = check_guardrails(
        regime="BEAR_TREND", proposal_after_sizing={},
        portfolio_aggregate_greeks=_greeks(), portfolio_factor_exposures={},
        portfolio_correlations={}, heat_metrics=_heat(max_per_expiry_pct=0.015),
        data_quality=_dq(),
    )
    assert bear.decision == "DOWNSIZE"
    assert any(v.rule == "same_expiry_premium_max" for v in bear.violations)


# ---------------------------------------------------------------------------
# All-clear → APPROVE
# ---------------------------------------------------------------------------


def test_all_clear_approves():
    res = check_guardrails(
        regime="BULL_TREND",
        proposal_after_sizing={},
        portfolio_aggregate_greeks=_greeks(
            delta_dollar_pct_equity=0.20,
            vega_10vol_pct_equity=0.005,
            gamma_2sigma_pct_equity=0.005,
        ),
        portfolio_factor_exposures={"tech": 0.20},
        portfolio_correlations={"A": {"B": 0.4}, "B": {"A": 0.4}},
        heat_metrics=_heat(heat_pct=0.04, max_per_underlying_pct=0.10),
        data_quality=_dq(),
    )
    assert res.decision == "APPROVE"
    assert res.violations == []
    assert res.suggested_qty_factor == 1.0


# ---------------------------------------------------------------------------
# All 5 regimes have valid guardrail tables
# ---------------------------------------------------------------------------


def test_all_regimes_have_guardrails():
    for regime in (
        "BULL_TREND",
        "RANGE_LOW_VOL",
        "VOLATILE_TRANSITION",
        "BEAR_TREND",
        "CRISIS",
    ):
        assert regime in REGIME_GUARDRAILS
        g = REGIME_GUARDRAILS[regime]
        assert g.heat_pct_max >= 0
        assert g.heat_pct_max <= 0.10
