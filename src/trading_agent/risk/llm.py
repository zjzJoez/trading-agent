"""LLM Risk Council — Phase 2.4 wiring placeholder.

The real council is a 3-step prompt chain:
    1. Risk Conservative Reviewer (Claude Sonnet 4.6)
    2. Risk Opportunity Reviewer  (Codex GPT-5.5)
    3. Risk Arbiter              (Claude Opus 4.7)

Phase 2.3 ships only the deterministic guardrails. The LLM council is
invoked only for borderline cases (DOWNSIZE / corner-case APPROVE) per
§7.4 of plan v3. Until oauth_router.py is wired (Phase 2.4), we return
a neutral stub that just confirms the deterministic decision.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

log = logging.getLogger(__name__)

CouncilDecision = Literal["APPROVE", "DOWNSIZE", "VETO", "DEFER"]


@dataclass
class CouncilReview:
    """One reviewer's output. Aggregated by the arbiter."""

    role: Literal["conservative", "opportunity"]
    decision: CouncilDecision
    primary_risks: list[str]
    reason: str
    cost_usd: float = 0.0


@dataclass
class CouncilDecisionResult:
    """Final output of the 3-step council."""

    decision: CouncilDecision
    approved_qty_factor: float = 1.0
    reviewers: list[CouncilReview] = field(default_factory=list)
    arbiter_reason: str = ""
    total_cost_usd: float = 0.0


def stub_council_review(
    *,
    deterministic_decision: CouncilDecision,
    deterministic_factor: float,
    deterministic_reasons: list[str],
) -> CouncilDecisionResult:
    """Stub for unit tests / no-LLM mode. Production uses `real_council_review`."""
    log.info(
        "[risk_llm:stub] passing through deterministic decision=%s factor=%.2f",
        deterministic_decision, deterministic_factor,
    )
    return CouncilDecisionResult(
        decision=deterministic_decision,
        approved_qty_factor=deterministic_factor,
        reviewers=[
            CouncilReview(
                role="conservative", decision=deterministic_decision,
                primary_risks=deterministic_reasons, reason="(stub)",
            ),
            CouncilReview(
                role="opportunity", decision=deterministic_decision,
                primary_risks=deterministic_reasons, reason="(stub)",
            ),
        ],
        arbiter_reason="stub_no_llm",
        total_cost_usd=0.0,
    )


def real_council_review(
    *,
    deterministic_decision: CouncilDecision,
    deterministic_factor: float,
    deterministic_reasons: list[str],
    proposal: dict,
    portfolio_summary: dict,
    regime_state: dict,
) -> CouncilDecisionResult:
    """Phase 2.4: 3-step LLM council via OAuth router.

    Conservative (Claude Sonnet) + Opportunity (Codex GPT-5.5) → Arbiter (Claude Opus).
    On any router failure → fall back to stub.

    Hard rules (enforced after the LLM returns; LLM cannot violate):
      - Council CANNOT exceed deterministic_factor (downsize cap)
      - Council CANNOT change deterministic VETO/DEFER to APPROVE/DOWNSIZE
    """
    try:
        from trading_agent.llm import get_router
        from trading_agent.llm.schemas import (
            RiskArbiterOutput,
            RiskConservativeOutput,
            RiskOpportunityOutput,
        )

        router = get_router()
        ctx = _format_council_context(
            deterministic_decision=deterministic_decision,
            deterministic_factor=deterministic_factor,
            deterministic_reasons=deterministic_reasons,
            proposal=proposal,
            portfolio_summary=portfolio_summary,
            regime_state=regime_state,
        )

        # Step 1: Conservative
        cons = router.call("risk_conservative", ctx, schema=RiskConservativeOutput)
        cons_parsed: RiskConservativeOutput | None = (
            cons.parsed if isinstance(cons.parsed, RiskConservativeOutput) else None
        )

        # Step 2: Opportunity (Codex)
        opp_ctx = ctx + "\n\nConservative Reviewer output:\n" + (
            cons_parsed.model_dump_json() if cons_parsed else "(unavailable)"
        )
        opp = router.call("risk_opportunity", opp_ctx, schema=RiskOpportunityOutput)
        opp_parsed: RiskOpportunityOutput | None = (
            opp.parsed if isinstance(opp.parsed, RiskOpportunityOutput) else None
        )

        # Step 3: Arbiter (Claude Opus)
        arb_ctx = ctx + "\n\nConservative:\n" + (
            cons_parsed.model_dump_json() if cons_parsed else "(unavailable)"
        ) + "\n\nOpportunity:\n" + (
            opp_parsed.model_dump_json() if opp_parsed else "(unavailable)"
        )
        arb = router.call("risk_arbiter", arb_ctx, schema=RiskArbiterOutput)
        arb_parsed: RiskArbiterOutput | None = (
            arb.parsed if isinstance(arb.parsed, RiskArbiterOutput) else None
        )

        if arb_parsed is None:
            log.warning("[risk_llm:real] arbiter returned no parsed output → fallback to stub")
            return stub_council_review(
                deterministic_decision=deterministic_decision,
                deterministic_factor=deterministic_factor,
                deterministic_reasons=deterministic_reasons,
            )

        # ----- Hard guardrails on the arbiter output -----
        decision = arb_parsed.decision
        # Cannot upgrade past deterministic
        if deterministic_decision == "VETO":
            decision = "VETO"
        elif deterministic_decision == "DEFER":
            decision = "DEFER" if decision != "VETO" else "VETO"
        elif deterministic_decision == "DOWNSIZE" and decision == "APPROVE":
            decision = "DOWNSIZE"

        # qty factor = arbiter approved_qty / requested_qty, clipped by deterministic_factor
        requested_qty = float(proposal.get("qty", 0)) or 1.0
        arb_factor = min(1.0, max(0.0, arb_parsed.approved_qty / requested_qty))
        final_factor = min(deterministic_factor, arb_factor)

        cost = (cons.cost_usd or 0.0) + (opp.cost_usd or 0.0) + (arb.cost_usd or 0.0)
        return CouncilDecisionResult(
            decision=decision,
            approved_qty_factor=final_factor,
            reviewers=[
                CouncilReview(
                    role="conservative",
                    decision=cons_parsed.decision if cons_parsed else deterministic_decision,
                    primary_risks=list(cons_parsed.primary_risks) if cons_parsed else [],
                    reason=(cons_parsed.concerns_md if cons_parsed else "(unavailable)")[:1000],
                    cost_usd=cons.cost_usd,
                ),
                CouncilReview(
                    role="opportunity",
                    decision=opp_parsed.decision if opp_parsed else deterministic_decision,
                    primary_risks=list(opp_parsed.primary_opportunities) if opp_parsed else [],
                    reason=(opp_parsed.rationale_md if opp_parsed else "(unavailable)")[:1000],
                    cost_usd=opp.cost_usd,
                ),
            ],
            arbiter_reason=arb_parsed.reason[:2000],
            total_cost_usd=cost,
        )
    except Exception as e:
        log.error("[risk_llm:real] failed → stub fallback: %s", e)
        return stub_council_review(
            deterministic_decision=deterministic_decision,
            deterministic_factor=deterministic_factor,
            deterministic_reasons=deterministic_reasons,
        )


def _format_council_context(
    *,
    deterministic_decision: CouncilDecision,
    deterministic_factor: float,
    deterministic_reasons: list[str],
    proposal: dict,
    portfolio_summary: dict,
    regime_state: dict,
) -> str:
    return (
        f"deterministic_decision: {deterministic_decision}\n"
        f"deterministic_qty_factor: {deterministic_factor:.2f}\n"
        f"deterministic_reasons: {deterministic_reasons}\n\n"
        f"regime: {regime_state.get('label', 'UNKNOWN')} "
        f"(confidence {regime_state.get('confidence', 0):.2f})\n\n"
        f"proposal_after_sizing:\n"
        f"  ticker: {proposal.get('ticker', '?')}\n"
        f"  symbol: {proposal.get('symbol', '?')}\n"
        f"  asset_type: {proposal.get('asset_type', 'OPT')}\n"
        f"  direction: {proposal.get('direction', '?')}\n"
        f"  qty: {proposal.get('qty', 0)}\n"
        f"  entry_price: {proposal.get('entry_price', 0)}\n"
        f"  stop: {proposal.get('stop', 'none')}\n"
        f"  target: {proposal.get('target', 'none')}\n"
        f"  strategy_label: {proposal.get('strategy_label', '?')}\n\n"
        f"risk_snapshot:\n"
        f"  equity: {portfolio_summary.get('equity', 0):.0f}\n"
        f"  n_positions: {portfolio_summary.get('n_positions', 0)}\n"
        f"  heat_pct: {portfolio_summary.get('heat_metrics', {}).get('heat_pct', 0):.2%}\n"
        f"  net_delta_pct: {portfolio_summary.get('aggregate_greeks', {}).get('delta_dollar_pct_equity', 0):.2%}\n"
        f"  vega_10vol_pct: {portfolio_summary.get('aggregate_greeks', {}).get('vega_10vol_pct_equity', 0):.2%}\n"
        f"  factor_exposures: {portfolio_summary.get('factor_exposures', {})}\n\n"
        f"Respond per your role's output schema."
    )


def should_invoke_council(
    *,
    deterministic_decision: CouncilDecision,
    regime_label: str,
    is_earnings_play: bool = False,
    canary_param_version: bool = False,
    new_portfolio_high_notional: bool = False,
) -> tuple[bool, list[str]]:
    """Decide if the LLM council should run for this proposal.

    From §7.4 of plan v3. Returns (should_invoke, reasons).
    """
    reasons = []
    if deterministic_decision != "APPROVE":
        reasons.append(f"deterministic={deterministic_decision}")
    if regime_label in {"VOLATILE_TRANSITION", "BEAR_TREND", "CRISIS"}:
        reasons.append(f"regime={regime_label}")
    if is_earnings_play:
        reasons.append("earnings_play")
    if canary_param_version:
        reasons.append("canary_param_version")
    if new_portfolio_high_notional:
        reasons.append("new_portfolio_high")

    return (len(reasons) > 0, reasons)
