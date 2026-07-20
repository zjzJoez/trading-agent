"""Phase 2.3 — real risk nodes for the candidate_entry / intraday_monitor graphs.

Replaces the stub `active_risk_snapshot`, `deterministic_risk_guardrails`,
`maybe_risk_llm_council`, `finalize_risk_decision` from `nodes/stubs.py`.

Pipeline (matches §6 + §7 of plan v3):

    proposal (from build_trade_proposal + deterministic_sizing + regime_execution_gate)
        │
        ▼
    active_risk_snapshot         build PortfolioSnapshot with open positions
        │                          (Phase 2.5 wires real moomoo fetch; for now
        │                           uses the proposal alone as a hypothetical)
        ▼
    deterministic_risk_guardrails  run check_guardrails() against snapshot
        │
        ▼
    maybe_risk_llm_council         optionally invoke 3-step LLM council
        │                          (Phase 2.4 wires real router; stub for now)
        ▼
    finalize_risk_decision         combine into RiskDecision, persist, emit

Each node writes back the partial result it owns to `state["risk"]`. The
`risk` slot doubles as the audit-trail container until finalize_risk_decision
emits the canonical RiskDecision shape per state.py.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from trading_agent.events import emit
from trading_agent.notify import send as ntfy_send
from trading_agent.risk.agent import RiskInput, decide
from trading_agent.risk.guardrails import check_guardrails
from trading_agent.risk.llm import (
    real_council_review,
    should_invoke_council,
)
from trading_agent.risk.persist import insert_risk_decision, insert_risk_snapshot
from trading_agent.risk.portfolio import OpenPosition, build_snapshot

log = logging.getLogger(__name__)


# Default paper-account starting equity used when no real fetch wired yet.
# Phase 2.5 will pull this from moomoo OpenD.
DEFAULT_PAPER_EQUITY = 100_000.0


def active_risk_snapshot(state: dict) -> dict:
    """Build the portfolio snapshot, including a hypothetical for the new proposal.

    For Phase 2.3 we don't yet have load_open_positions wired (Phase 2.5).
    The snapshot uses any positions already in state plus the new proposal
    treated as an open position. This lets guardrails see the
    after-this-trade portfolio.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[risk/active_risk_snapshot] run_id=%s", run_id)

    proposal = state.get("proposal") or {}
    existing = _existing_open_positions_from_state(state)
    hypothetical = _proposal_to_hypothetical(proposal)

    positions: list[OpenPosition] = list(existing)
    if hypothetical is not None:
        positions.append(hypothetical)

    equity = float(state.get("account", {}).get("equity", DEFAULT_PAPER_EQUITY))
    cash = float(state.get("account", {}).get("cash", equity))

    snap = build_snapshot(equity=equity, cash=cash, open_positions=positions)

    emit(
        run_id=run_id,
        trigger=trigger,
        agent="active_risk_snapshot",
        event_type="snapshot_built",
        payload={
            "n_positions": snap.n_open,
            "heat_pct": snap.heat_pct,
            "equity": equity,
            "data_quality_level": snap.data_quality.get("degradation_level", 0),
        },
    )

    return {
        "risk": {
            **state.get("risk", {}),
            "_snapshot": {
                "as_of": snap.as_of.isoformat(),
                "equity": equity,
                "n_positions": snap.n_open,
                "aggregate_greeks": snap.aggregate_greeks,
                "factor_exposures": snap.factor_exposures,
                "correlations": snap.correlations,
                "heat_metrics": snap.heat_metrics,
                "data_quality": snap.data_quality,
                "liquidity_warnings": snap.liquidity_warnings,
                "_obj_positions": [_pos_to_dict(p) for p in snap.open_positions],
            },
        }
    }


def deterministic_risk_guardrails(state: dict) -> dict:
    """Run the deterministic guardrails layer; stash the GuardrailsCheck."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[risk/deterministic_risk_guardrails] run_id=%s", run_id)

    risk_state = state.get("risk", {})
    snap = risk_state.get("_snapshot")
    if not snap:
        log.warning("guardrails called with no snapshot in state — skipping")
        return {}

    regime_label = state.get("regime", {}).get("label", "VOLATILE_TRANSITION")
    proposal = state.get("proposal") or {}

    gc = check_guardrails(
        regime=regime_label,  # type: ignore[arg-type]
        proposal_after_sizing=proposal,
        portfolio_aggregate_greeks=snap["aggregate_greeks"],
        portfolio_factor_exposures=snap["factor_exposures"],
        portfolio_correlations=snap["correlations"],
        heat_metrics=snap["heat_metrics"],
        data_quality=snap["data_quality"],
        real_trading_marker=bool(state.get("real_trading_marker_detected")),
    )

    emit(
        run_id=run_id,
        trigger=trigger,
        agent="deterministic_risk_guardrails",
        event_type="guardrails_evaluated",
        payload={
            "decision": gc.decision,
            "n_violations": len(gc.violations),
            "suggested_qty_factor": gc.suggested_qty_factor,
            "rules_breached": [v.rule for v in gc.violations],
            "regime": regime_label,
        },
    )

    return {
        "risk": {
            **risk_state,
            "_guardrails_check": {
                "decision": gc.decision,
                "suggested_qty_factor": gc.suggested_qty_factor,
                "reasons": list(gc.reasons),
                "violations": [
                    {
                        "rule": v.rule,
                        "severity": v.severity,
                        "detail": v.detail,
                        "measured": v.measured,
                        "threshold": v.threshold,
                    }
                    for v in gc.violations
                ],
            },
        }
    }


def maybe_risk_llm_council(state: dict) -> dict:
    """Invoke the 3-agent risk council on borderline cases. Phase 2.3 = stub."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[risk/maybe_risk_llm_council] run_id=%s", run_id)

    risk_state = state.get("risk", {})
    gc = risk_state.get("_guardrails_check")
    if not gc:
        return {}

    regime_label = state.get("regime", {}).get("label", "VOLATILE_TRANSITION")
    invoke, council_reasons = should_invoke_council(
        deterministic_decision=gc["decision"],
        regime_label=regime_label,
        is_earnings_play=bool(state.get("proposal", {}).get("is_earnings_play")),
        canary_param_version=bool(state.get("proposal", {}).get("canary_param_version")),
    )

    if not invoke:
        emit(
            run_id=run_id,
            trigger=trigger,
            agent="maybe_risk_llm_council",
            event_type="skipped",
            payload={"reason": "deterministic_clear"},
        )
        return {
            "risk": {
                **risk_state,
                "_council_invoked": False,
                "_council_reasons": [],
            }
        }

    proposal = state.get("proposal") or {}
    council = real_council_review(
        deterministic_decision=gc["decision"],
        deterministic_factor=gc["suggested_qty_factor"],
        deterministic_reasons=gc["reasons"],
        proposal=proposal,
        portfolio_summary={
            "equity": risk_state.get("_snapshot", {}).get("equity", 0),
            "n_positions": risk_state.get("_snapshot", {}).get("n_positions", 0),
            "aggregate_greeks": risk_state.get("_snapshot", {}).get("aggregate_greeks", {}),
            "factor_exposures": risk_state.get("_snapshot", {}).get("factor_exposures", {}),
            "heat_metrics": risk_state.get("_snapshot", {}).get("heat_metrics", {}),
        },
        regime_state=state.get("regime", {}),
    )

    emit(
        run_id=run_id,
        trigger=trigger,
        agent="maybe_risk_llm_council",
        event_type="council_reviewed",
        payload={
            "decision": council.decision,
            "approved_factor": council.approved_qty_factor,
            "is_stub": True,
            "trigger_reasons": council_reasons,
        },
    )

    return {
        "risk": {
            **risk_state,
            "_council_invoked": True,
            "_council_reasons": council_reasons,
            "_council_result": {
                "decision": council.decision,
                "approved_qty_factor": council.approved_qty_factor,
                "arbiter_reason": council.arbiter_reason,
                "total_cost_usd": council.total_cost_usd,
                "reviewers": [
                    {
                        "role": r.role,
                        "decision": r.decision,
                        "primary_risks": r.primary_risks,
                        "reason": r.reason,
                        "cost_usd": r.cost_usd,
                    }
                    for r in council.reviewers
                ],
            },
        }
    }


def finalize_risk_decision(state: dict) -> dict:
    """Combine the stashed _guardrails_check + _council_result into the canonical
    RiskDecision; persist; emit event; ntfy on VETO/DEFER.

    Reads (does NOT re-run) the deterministic guardrails and LLM council that
    earlier nodes computed and stashed under state["risk"].
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[risk/finalize_risk_decision] run_id=%s", run_id)

    import uuid
    from datetime import datetime, timedelta, timezone

    from trading_agent.risk.agent import DECISION_TTL_SECONDS, RiskOutput
    from trading_agent.risk.guardrails import GuardrailsCheck, GuardrailViolation
    from trading_agent.risk.llm import CouncilDecisionResult, CouncilReview

    risk_state = state.get("risk", {})
    snap_dict = risk_state.get("_snapshot")
    proposal = state.get("proposal") or {}

    gc_dict = risk_state.get("_guardrails_check") or {}
    deterministic_decision = gc_dict.get("decision", "DEFER")
    deterministic_factor = gc_dict.get("suggested_qty_factor", 1.0)

    council_dict = risk_state.get("_council_result")
    final_decision = deterministic_decision
    final_factor = deterministic_factor

    if council_dict is not None:
        cdec = council_dict.get("decision", deterministic_decision)
        cfac = council_dict.get("approved_qty_factor", deterministic_factor)
        # Council can only make decisions stricter, never more permissive
        if cdec in ("VETO", "DEFER") and final_decision != "VETO":
            final_decision = cdec
        elif final_decision == "DOWNSIZE" and cdec == "DOWNSIZE":
            final_factor = min(final_factor, cfac)

    requested_qty = float(proposal.get("qty", 0))
    if final_decision in ("VETO", "DEFER"):
        approved_qty = 0.0
        max_entry_price = 0.0
    else:
        approved_qty = max(0.0, requested_qty * final_factor)
        if proposal.get("asset_type") == "OPT":
            # Round-half-up, NOT int() truncation.
            # int(0.6) = 0 would convert "DOWNSIZE to 60% of 1 contract" into
            # a forced VETO ("downsize_to_zero_contracts") — that's a bug we
            # already fixed in regime_execution_gate; the same fix has to
            # live here. 1 × 0.6 should round to 1, not 0.
            approved_qty = float(int(approved_qty + 0.5)) if approved_qty > 0 else 0.0
            if approved_qty == 0 and requested_qty > 0:
                final_decision = "VETO"
                gc_dict.setdefault("reasons", []).append("downsize_to_zero_contracts")
        max_entry_price = float(proposal.get("entry_price", 0)) * 1.005

    decision_id = (
        f"risk_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )
    expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=DECISION_TTL_SECONDS)
    ).isoformat()

    # Reconstruct GuardrailsCheck for persist
    violations_objs = [
        GuardrailViolation(
            rule=v["rule"], severity=v["severity"], detail=v["detail"],
            measured=v["measured"], threshold=v["threshold"],
        )
        for v in gc_dict.get("violations", [])
    ]
    gc = GuardrailsCheck(
        decision=deterministic_decision,
        violations=violations_objs,
        suggested_qty_factor=deterministic_factor,
        reasons=list(gc_dict.get("reasons", [])),
    )

    # Reconstruct CouncilDecisionResult for persist
    council_obj: CouncilDecisionResult | None = None
    if council_dict is not None:
        council_obj = CouncilDecisionResult(
            decision=council_dict.get("decision", deterministic_decision),
            approved_qty_factor=council_dict.get("approved_qty_factor", deterministic_factor),
            reviewers=[
                CouncilReview(
                    role=r.get("role", "conservative"),
                    decision=r.get("decision", deterministic_decision),
                    primary_risks=list(r.get("primary_risks", [])),
                    reason=r.get("reason", ""),
                    cost_usd=float(r.get("cost_usd") or 0.0),
                )
                for r in council_dict.get("reviewers", [])
            ],
            arbiter_reason=council_dict.get("arbiter_reason", ""),
            total_cost_usd=float(council_dict.get("total_cost_usd") or 0.0),
        )

    out = RiskOutput(
        risk_decision_id=decision_id,
        decision=final_decision,
        approved_qty=approved_qty,
        max_entry_price=max_entry_price,
        reasons=list(gc.reasons) + (
            risk_state.get("_council_reasons", []) if council_dict else []
        ),
        hard_violations=[v.rule for v in gc.violations if v.severity == "VETO"],
        guardrails_check=gc,
        council_result=council_obj,
        expires_at=expires_at,
        cost_usd=council_obj.total_cost_usd if council_obj else 0.0,
    )

    # Build a transient snapshot just for persistence (DB only needs the dict payload)
    positions = [_pos_from_dict(d) for d in (snap_dict or {}).get("_obj_positions", [])]
    snap = build_snapshot(
        equity=(snap_dict or {}).get("equity", DEFAULT_PAPER_EQUITY),
        cash=(snap_dict or {}).get("equity", DEFAULT_PAPER_EQUITY),
        open_positions=positions,
    )

    snapshot_id: int | None = None
    decision_db_id: int | None = None
    try:
        snapshot_id = insert_risk_snapshot(
            snap, regime_state_id=state.get("regime", {}).get("state_id")
        )
        decision_db_id = insert_risk_decision(
            out,
            proposal=proposal,
            thesis_id=proposal.get("thesis_id"),
            regime_state_id=state.get("regime", {}).get("state_id"),
            risk_snapshot_id=snapshot_id,
        )
    except Exception as e:
        log.error("[risk/finalize_risk_decision] DB persist failed: %s", e)
        emit(
            run_id=run_id, trigger=trigger, agent="finalize_risk_decision",
            event_type="persist_failed", payload={"error": str(e)}, severity=2,
        )

    emit(
        run_id=run_id, trigger=trigger, agent="finalize_risk_decision",
        event_type="decision_finalized",
        payload={
            "decision": out.decision,
            "approved_qty": out.approved_qty,
            "n_violations": len(out.hard_violations),
            "council_invoked": council_obj is not None,
            "snapshot_id": snapshot_id,
            "decision_db_id": decision_db_id,
        },
    )

    if out.decision in ("VETO", "DEFER"):
        ntfy_send(
            "risk",
            title=f"Risk {out.decision}: {proposal.get('symbol', '?')}",
            body=(
                f"Decision: {out.decision}\n"
                f"Reasons: {', '.join(out.reasons[:5])}\n"
                f"Hard violations: {', '.join(out.hard_violations) or 'none'}"
            ),
            priority=4,
            tags=["no_entry"] if out.decision == "VETO" else ["hourglass"],
        )

    risk_decision_payload = {
        "risk_decision_id": out.risk_decision_id,
        "decision": out.decision,
        "approved_qty": out.approved_qty,
        "max_entry_price": out.max_entry_price,
        "reasons": out.reasons,
        "hard_violations": out.hard_violations,
        "risk_snapshot_id": snapshot_id or 0,
        "expires_at": out.expires_at,
    }
    return {"risk": {**risk_state, **risk_decision_payload}}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _existing_open_positions_from_state(state: dict) -> list[OpenPosition]:
    """Pull existing positions from state (intraday_monitor sets these);
    candidate_entry runs typically have an empty list here until Phase 2.5."""
    raw = state.get("positions") or []
    return [_pos_from_dict(p) for p in raw]


def _proposal_to_hypothetical(proposal: dict) -> OpenPosition | None:
    """Treat the new proposal as a hypothetical open position for after-trade view.

    The trader's schema only carries `option_delta` (not gamma/vega/theta), so
    we default the rest to 0.0 — this slightly understates aggregate Greek
    exposure but avoids DEFER from missing-greeks data quality flag (the
    hypothetical hasn't been filled yet; real Greeks land at fill time).
    """
    if not proposal or not proposal.get("symbol"):
        return None
    asset_type = proposal.get("asset_type", "OPT")
    return OpenPosition(
        symbol=proposal["symbol"],
        underlying=proposal.get("ticker") or proposal.get("underlying", "UNKNOWN"),
        asset_type=asset_type,
        side=proposal.get("side", "BUY"),
        qty=float(proposal.get("qty", 0)),
        entry_price=float(proposal.get("entry_price", 0)),
        mark=float(proposal.get("entry_price", 0)),
        stop=proposal.get("stop"),
        target=proposal.get("target"),
        # Greeks must default to 0.0 and never be None: a None delta/gamma reads
        # as "missing critical greeks" → data_quality degradation_level=2 → DEFER.
        # candidate_entry's only snapshot position is this not-yet-filled
        # hypothetical (held positions aren't wired into its state), so a
        # proposal that omits option_delta used to freeze EVERY new entry.
        # `or 0.0` covers both an absent key and a present-but-None value.
        delta=(proposal.get("option_delta") or 0.0) if asset_type == "OPT" else 1.0,
        gamma=proposal.get("option_gamma") or 0.0,
        vega=proposal.get("option_vega") or 0.0,
        theta=proposal.get("option_theta") or 0.0,
        iv=proposal.get("option_iv") or 0.0,
        notional=float(proposal.get("entry_price", 0)) * float(proposal.get("qty", 0))
        * (100 if asset_type == "OPT" else 1),
        unrealized_pnl=0.0,
        thesis_id=proposal.get("thesis_id"),
        sector=proposal.get("sector"),
        strategy_label=proposal.get("strategy_label"),
        age_minutes=0.0,
    )


def _pos_to_dict(p: OpenPosition) -> dict[str, Any]:
    return {k: getattr(p, k) for k in p.__dataclass_fields__}


def _pos_from_dict(d: dict) -> OpenPosition:
    return OpenPosition(**{k: d.get(k) for k in OpenPosition.__dataclass_fields__})


def route_risk_decision(state: dict) -> str:
    """Conditional router after finalize_risk_decision.

    Real version (replaces stubs.route_risk_decision). Reads the canonical
    `state["risk"]["decision"]` field set by finalize_risk_decision.
    """
    risk = state.get("risk") or {}
    decision = risk.get("decision", "DEFER")
    return {
        "APPROVE": "execute_paper_order",
        "DOWNSIZE": "execute_paper_order",
        "VETO": "persist_veto",
        "DEFER": "persist_defer",
    }.get(decision, "persist_defer")
