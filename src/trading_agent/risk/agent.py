"""Active Risk Agent — top-level decision orchestrator.

Inputs:  proposal + portfolio snapshot + regime
Outputs: APPROVE / DOWNSIZE / VETO / DEFER (with approved_qty + reasons)

Flow:
    1. Compute aggregates if a "after-this-trade" snapshot is needed
    2. Run deterministic guardrails (Python-enforced, hard caps)
    3. If borderline → invoke the LLM council (here: the deterministic stub)
    4. Return final RiskOutput

NOTE ON THE COUNCIL: `decide()` is the deterministic, no-LLM orchestrator —
it deliberately uses `stub_council_review` and is retained for unit tests and
as a library re-export (`trading_agent.risk.decide`). The PRODUCTION risk path
does NOT call `decide()`; it runs through `graph/nodes/risk_nodes.py`, which
calls the real LLM council (`real_council_review`, with its own stub fallback
when the router is unavailable). Keep this path LLM-free so tests stay
hermetic and `decide()` has no router dependency.

The agent never directly calls OpenD. It just makes a decision; the
executor is what places the order if approved.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from trading_agent.risk.guardrails import (
    GuardrailsCheck,
    REGIME_GUARDRAILS,
    check_guardrails,
)
from trading_agent.risk.llm import (
    CouncilDecisionResult,
    should_invoke_council,
    stub_council_review,
)
from trading_agent.risk.portfolio import PortfolioSnapshot

log = logging.getLogger(__name__)


@dataclass
class RiskInput:
    """Everything the risk agent reads to decide on one proposal."""

    proposal: dict[str, Any]  # TradeProposal-shaped dict
    portfolio: PortfolioSnapshot
    regime: dict[str, Any]  # {label, confidence, gate, ...}
    real_trading_marker_detected: bool = False
    is_earnings_play: bool = False
    canary_param_version: bool = False


@dataclass
class RiskOutput:
    """Final decision with full audit trail."""

    risk_decision_id: str  # ULID-like
    decision: str  # APPROVE | DOWNSIZE | VETO | DEFER
    approved_qty: float  # 0.0 on VETO/DEFER
    max_entry_price: float
    reasons: list[str] = field(default_factory=list)
    hard_violations: list[str] = field(default_factory=list)
    guardrails_check: GuardrailsCheck | None = None
    council_result: CouncilDecisionResult | None = None
    expires_at: str = ""  # ISO 8601
    cost_usd: float = 0.0


# Risk decisions auto-expire after 5 minutes by default — prevents
# replay of stale APPROVES if the executor lags.
DECISION_TTL_SECONDS = 300


def decide(input: RiskInput) -> RiskOutput:
    """Top-level entry: deterministic guardrails first, council on borderline."""
    proposal = input.proposal
    portfolio = input.portfolio
    regime = input.regime
    regime_label = regime.get("label", "VOLATILE_TRANSITION")

    decision_id = f"risk_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=DECISION_TTL_SECONDS)
    ).isoformat()

    # -------- Step 1: deterministic guardrails --------
    gc = check_guardrails(
        regime=regime_label,  # type: ignore[arg-type]
        proposal_after_sizing=proposal,
        portfolio_aggregate_greeks=portfolio.aggregate_greeks,
        portfolio_factor_exposures=portfolio.factor_exposures,
        portfolio_correlations=portfolio.correlations,
        heat_metrics=portfolio.heat_metrics,
        data_quality=portfolio.data_quality,
        real_trading_marker=input.real_trading_marker_detected,
    )

    log.info(
        "[risk_agent] deterministic decision=%s violations=%d factor=%.2f",
        gc.decision, len(gc.violations), gc.suggested_qty_factor,
    )

    # -------- Step 2: should we invoke the LLM council? --------
    invoke, council_reasons = should_invoke_council(
        deterministic_decision=gc.decision,
        regime_label=regime_label,
        is_earnings_play=input.is_earnings_play,
        canary_param_version=input.canary_param_version,
        new_portfolio_high_notional=False,  # Phase 2.4 wires this
    )

    if invoke:
        # decide() is the deterministic/test path: it ALWAYS uses the stub.
        # The real LLM council lives in graph/nodes/risk_nodes.py (production).
        council = stub_council_review(
            deterministic_decision=gc.decision,
            deterministic_factor=gc.suggested_qty_factor,
            deterministic_reasons=gc.reasons,
        )
    else:
        council = None

    # -------- Step 3: combine deterministic + council into final output --------
    final_decision = gc.decision
    final_factor = gc.suggested_qty_factor

    if council is not None:
        # Council can only make decisions stricter, never more permissive.
        if council.decision in ("VETO", "DEFER") and final_decision != "VETO":
            final_decision = council.decision
        elif final_decision == "DOWNSIZE" and council.decision == "DOWNSIZE":
            # Take the smaller factor (more conservative)
            final_factor = min(final_factor, council.approved_qty_factor)

    # Compute final approved_qty + entry price ceiling
    requested_qty = float(proposal.get("qty", 0))
    if final_decision in ("VETO", "DEFER"):
        approved_qty = 0.0
        max_entry_price = 0.0
    else:
        approved_qty = max(0.0, requested_qty * final_factor)
        # Round-half-up for option contracts, NOT int() truncation.
        # int(0.6) = 0 would convert "DOWNSIZE to 60% of 1 contract" into
        # a forced VETO, which contradicts the Council's intent. Round-half-up
        # keeps a single-contract proposal alive at light downsizes (factor>=0.5)
        # while still floor-to-zero / VETO when the downsize is severe (<0.5).
        # The fix lives identically in trade_nodes.regime_execution_gate and
        # risk_nodes.finalize_risk_decision — keep all three in sync.
        if proposal.get("asset_type") == "OPT":
            approved_qty = float(int(approved_qty + 0.5)) if approved_qty > 0 else 0.0
            if approved_qty == 0:
                final_decision = "VETO"  # downsize-to-zero == effectively veto
                gc.reasons.append("downsize_to_zero_contracts")
        # Allow up to 0.5% above proposed entry (for limit price headroom)
        max_entry_price = float(proposal.get("entry_price", 0)) * 1.005

    return RiskOutput(
        risk_decision_id=decision_id,
        decision=final_decision,
        approved_qty=approved_qty,
        max_entry_price=max_entry_price,
        reasons=gc.reasons + (council_reasons if invoke else []),
        hard_violations=[v.rule for v in gc.violations if v.severity == "VETO"],
        guardrails_check=gc,
        council_result=council,
        expires_at=expires_at,
        cost_usd=council.total_cost_usd if council else 0.0,
    )
