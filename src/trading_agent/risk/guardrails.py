"""Per-regime tiered hard guardrails for Active Risk Agent.

These limits CANNOT be overridden by any LLM call. Every borderline trade
proposal must pass these deterministic gates before the LLM council even
sees it. (LLM can only choose APPROVE/DOWNSIZE/VETO/DEFER inside the
envelope these guardrails define — never outside.)

§7.3 of plan v3. Tunable via Phase 2.7 online learning within bounded
ranges (see learning/params.py). Defaults are intentionally conservative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RegimeLabel = Literal[
    "BULL_TREND",
    "RANGE_LOW_VOL",
    "VOLATILE_TRANSITION",
    "BEAR_TREND",
    "CRISIS",
]


@dataclass
class Guardrails:
    """Per-regime tiered limits. All thresholds are fractions of equity unless noted."""

    # Heat-to-stop (sum of $-at-risk / equity)
    heat_pct_max: float
    # Single-underlying delta-dollar exposure
    single_underlying_delta_dollar_max: float
    # Beta-adjusted net portfolio delta
    portfolio_net_delta_max: float
    # Vega ±10vol shock $-impact
    vega_10vol_max: float
    # Gamma 2σ underlying shock $-impact
    gamma_2sigma_max: float
    # Avg pairwise correlation across open risk assets (raw fraction, not equity)
    avg_pairwise_corr_max: float
    # Premium-at-risk in same expiry bucket
    same_expiry_premium_max: float
    # Per-factor exposure (tech, rates_sensitive, commodity, etc.)
    factor_exposure_max: float
    # Whether new risk-on entries are allowed at all
    allow_new_risk_on: bool


# Per-regime defaults — strictly tier downward as regime worsens.
# These match the plan v3 §7.3 table.
REGIME_GUARDRAILS: dict[RegimeLabel, Guardrails] = {
    "BULL_TREND": Guardrails(
        heat_pct_max=0.06,
        single_underlying_delta_dollar_max=0.15,
        portfolio_net_delta_max=0.35,
        vega_10vol_max=0.01,
        gamma_2sigma_max=0.015,
        avg_pairwise_corr_max=0.65,
        same_expiry_premium_max=0.02,
        factor_exposure_max=0.30,
        allow_new_risk_on=True,
    ),
    "RANGE_LOW_VOL": Guardrails(
        heat_pct_max=0.06,
        single_underlying_delta_dollar_max=0.15,
        portfolio_net_delta_max=0.35,
        vega_10vol_max=0.01,
        gamma_2sigma_max=0.015,
        avg_pairwise_corr_max=0.65,
        same_expiry_premium_max=0.02,
        factor_exposure_max=0.30,
        allow_new_risk_on=True,
    ),
    "VOLATILE_TRANSITION": Guardrails(
        heat_pct_max=0.03,
        single_underlying_delta_dollar_max=0.075,
        portfolio_net_delta_max=0.15,
        vega_10vol_max=0.005,
        gamma_2sigma_max=0.0075,
        avg_pairwise_corr_max=0.55,
        same_expiry_premium_max=0.01,
        factor_exposure_max=0.20,
        allow_new_risk_on=True,
    ),
    "BEAR_TREND": Guardrails(
        heat_pct_max=0.03,
        single_underlying_delta_dollar_max=0.075,
        portfolio_net_delta_max=0.15,  # negative-bias acceptable in BEAR
        vega_10vol_max=0.005,
        gamma_2sigma_max=0.0075,
        avg_pairwise_corr_max=0.55,
        same_expiry_premium_max=0.01,
        factor_exposure_max=0.20,
        allow_new_risk_on=True,  # but only LONG_PUT direction (gates.py blocks calls)
    ),
    "CRISIS": Guardrails(
        heat_pct_max=0.015,
        single_underlying_delta_dollar_max=0.0,
        portfolio_net_delta_max=0.0,
        vega_10vol_max=0.0025,
        gamma_2sigma_max=0.0025,
        avg_pairwise_corr_max=1.0,  # n/a in crisis (we're exiting, not opening)
        same_expiry_premium_max=0.0,
        factor_exposure_max=0.05,
        allow_new_risk_on=False,
    ),
}


@dataclass
class GuardrailViolation:
    """One specific guardrail breach. Multiple may stack."""

    rule: str  # e.g. "heat_pct_max", "single_underlying_delta_dollar_max"
    severity: Literal["DOWNSIZE", "VETO", "DEFER"]
    detail: str
    measured: float
    threshold: float


@dataclass
class GuardrailsCheck:
    """Output of check_guardrails. `violations` is empty for a clean APPROVE."""

    decision: Literal["APPROVE", "DOWNSIZE", "VETO", "DEFER"]
    violations: list[GuardrailViolation] = field(default_factory=list)
    suggested_qty_factor: float = 1.0  # multiply original qty by this on DOWNSIZE
    reasons: list[str] = field(default_factory=list)


# Severity thresholds — how far past a cap before VETO instead of DOWNSIZE
DOWNSIZE_TO_VETO_RATIO = 1.5


def _classify_breach(measured: float, threshold: float) -> Literal["DOWNSIZE", "VETO"]:
    """How bad is the breach? Within 1.5× → DOWNSIZE; beyond → VETO."""
    if threshold <= 0:
        return "VETO" if measured != 0 else "DOWNSIZE"
    ratio = abs(measured) / abs(threshold)
    return "VETO" if ratio > DOWNSIZE_TO_VETO_RATIO else "DOWNSIZE"


def check_guardrails(
    *,
    regime: RegimeLabel,
    proposal_after_sizing: dict,
    portfolio_aggregate_greeks: dict,
    portfolio_factor_exposures: dict,
    portfolio_correlations: dict,
    heat_metrics: dict,
    data_quality: dict,
    real_trading_marker: bool = False,
) -> GuardrailsCheck:
    """Run every hard guardrail. Returns the worst decision + all violations.

    Inputs come from PortfolioSnapshot + the proposed trade. The function is
    pure (no I/O) so it's trivially testable.
    """
    g = REGIME_GUARDRAILS[regime]
    violations: list[GuardrailViolation] = []
    suggested_factor = 1.0

    # -------- VETO: real-trading marker (always wins) --------
    if real_trading_marker:
        return GuardrailsCheck(
            decision="VETO",
            violations=[
                GuardrailViolation(
                    rule="real_trading_marker",
                    severity="VETO",
                    detail="real-trading marker detected — paper-only system",
                    measured=1.0,
                    threshold=0.0,
                )
            ],
            reasons=["real_trading_marker_blocked"],
        )

    # -------- DEFER: data quality / regime missing --------
    if data_quality.get("degradation_level", 0) >= 2:
        return GuardrailsCheck(
            decision="DEFER",
            violations=[
                GuardrailViolation(
                    rule="data_quality",
                    severity="DEFER",
                    detail=f"missing critical data: {data_quality.get('missing', [])}",
                    measured=2.0,
                    threshold=2.0,
                )
            ],
            reasons=["data_quality_critical"],
        )

    # -------- VETO: regime forbids new risk-on (CRISIS) --------
    if not g.allow_new_risk_on:
        return GuardrailsCheck(
            decision="VETO",
            violations=[
                GuardrailViolation(
                    rule="allow_new_risk_on",
                    severity="VETO",
                    detail=f"regime={regime} blocks new risk-on entries",
                    measured=1.0,
                    threshold=0.0,
                )
            ],
            reasons=["regime_blocks_new_entries"],
        )

    # -------- Aggregate greek caps (after this trade) --------
    delta_dollar_pct = portfolio_aggregate_greeks.get("delta_dollar_pct_equity", 0)
    if abs(delta_dollar_pct) > g.portfolio_net_delta_max:
        sev = _classify_breach(delta_dollar_pct, g.portfolio_net_delta_max)
        violations.append(
            GuardrailViolation(
                rule="portfolio_net_delta_max",
                severity=sev,
                detail=f"net delta {delta_dollar_pct:.1%} > cap {g.portfolio_net_delta_max:.1%}",
                measured=abs(delta_dollar_pct),
                threshold=g.portfolio_net_delta_max,
            )
        )
        if sev == "DOWNSIZE":
            suggested_factor = min(
                suggested_factor, g.portfolio_net_delta_max / abs(delta_dollar_pct)
            )

    vega_pct = portfolio_aggregate_greeks.get("vega_10vol_pct_equity", 0)
    if vega_pct > g.vega_10vol_max:
        sev = _classify_breach(vega_pct, g.vega_10vol_max)
        violations.append(
            GuardrailViolation(
                rule="vega_10vol_max",
                severity=sev,
                detail=f"vega 10vol shock {vega_pct:.2%} > cap {g.vega_10vol_max:.2%}",
                measured=vega_pct,
                threshold=g.vega_10vol_max,
            )
        )
        if sev == "DOWNSIZE":
            suggested_factor = min(suggested_factor, g.vega_10vol_max / vega_pct)

    gamma_pct = portfolio_aggregate_greeks.get("gamma_2sigma_pct_equity", 0)
    if gamma_pct > g.gamma_2sigma_max:
        sev = _classify_breach(gamma_pct, g.gamma_2sigma_max)
        violations.append(
            GuardrailViolation(
                rule="gamma_2sigma_max",
                severity=sev,
                detail=f"gamma 2σ {gamma_pct:.2%} > cap {g.gamma_2sigma_max:.2%}",
                measured=gamma_pct,
                threshold=g.gamma_2sigma_max,
            )
        )
        if sev == "DOWNSIZE":
            suggested_factor = min(suggested_factor, g.gamma_2sigma_max / gamma_pct)

    # -------- Heat-to-stop --------
    heat_pct = heat_metrics.get("heat_pct", 0)
    if heat_pct > g.heat_pct_max:
        sev = _classify_breach(heat_pct, g.heat_pct_max)
        violations.append(
            GuardrailViolation(
                rule="heat_pct_max",
                severity=sev,
                detail=f"portfolio heat {heat_pct:.2%} > cap {g.heat_pct_max:.2%}",
                measured=heat_pct,
                threshold=g.heat_pct_max,
            )
        )
        if sev == "DOWNSIZE":
            suggested_factor = min(suggested_factor, g.heat_pct_max / heat_pct)

    # -------- Single-underlying delta-$ --------
    max_per_und = heat_metrics.get("max_per_underlying_pct", 0)
    if max_per_und > g.single_underlying_delta_dollar_max:
        sev = _classify_breach(max_per_und, g.single_underlying_delta_dollar_max)
        violations.append(
            GuardrailViolation(
                rule="single_underlying_delta_dollar_max",
                severity=sev,
                detail=(
                    f"max per-underlying {max_per_und:.2%} > "
                    f"cap {g.single_underlying_delta_dollar_max:.2%}"
                ),
                measured=max_per_und,
                threshold=g.single_underlying_delta_dollar_max,
            )
        )
        if sev == "DOWNSIZE":
            suggested_factor = min(
                suggested_factor,
                g.single_underlying_delta_dollar_max / max_per_und,
            )

    # -------- Factor exposure --------
    for factor, exposure in (portfolio_factor_exposures or {}).items():
        if abs(exposure) > g.factor_exposure_max:
            sev = _classify_breach(exposure, g.factor_exposure_max)
            violations.append(
                GuardrailViolation(
                    rule=f"factor_exposure_max[{factor}]",
                    severity=sev,
                    detail=f"factor={factor} exposure {exposure:.1%} > cap {g.factor_exposure_max:.1%}",
                    measured=abs(exposure),
                    threshold=g.factor_exposure_max,
                )
            )
            if sev == "DOWNSIZE":
                suggested_factor = min(
                    suggested_factor, g.factor_exposure_max / abs(exposure)
                )

    # -------- Avg pairwise correlation --------
    if portfolio_correlations:
        max_corr = _max_off_diagonal(portfolio_correlations)
        if max_corr > g.avg_pairwise_corr_max:
            sev = _classify_breach(max_corr, g.avg_pairwise_corr_max)
            violations.append(
                GuardrailViolation(
                    rule="avg_pairwise_corr_max",
                    severity=sev,
                    detail=(
                        f"max pairwise corr {max_corr:.2f} > "
                        f"cap {g.avg_pairwise_corr_max:.2f}"
                    ),
                    measured=max_corr,
                    threshold=g.avg_pairwise_corr_max,
                )
            )
            # Correlation breach can't be fixed by downsizing — VETO if strong
            if sev == "DOWNSIZE":
                # Mild over-corr → DOWNSIZE 50%
                suggested_factor = min(suggested_factor, 0.5)

    # -------- Aggregate decision --------
    if not violations:
        return GuardrailsCheck(decision="APPROVE", reasons=["all_guardrails_passed"])

    if any(v.severity == "VETO" for v in violations):
        return GuardrailsCheck(
            decision="VETO",
            violations=violations,
            reasons=[v.rule for v in violations if v.severity == "VETO"],
        )

    return GuardrailsCheck(
        decision="DOWNSIZE",
        violations=violations,
        suggested_qty_factor=max(0.1, suggested_factor),  # floor at 10% to avoid pin-prick orders
        reasons=[v.rule for v in violations],
    )


def _max_off_diagonal(corr: dict[str, dict[str, float]]) -> float:
    """Max correlation strictly off-diagonal."""
    best = 0.0
    syms = list(corr.keys())
    for i, a in enumerate(syms):
        for b in syms[i + 1:]:
            v = corr.get(a, {}).get(b, 0.0)
            if abs(v) > best:
                best = abs(v)
    return best
