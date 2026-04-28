"""Phase 2.1 stub nodes — emit agent_events + ntfy heartbeats; no real work.

Each stub:
1. Logs an `agent_events.start` row with the node name + run_id
2. Returns a small state mutation showing it ran
3. Logs `agent_events.finish`

This proves end-to-end wiring (state → checkpointer → events → notify)
without requiring regime / risk / LLM components to exist yet.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from trading_agent.events import emit
from trading_agent.graph.state import TradingGraphState

log = logging.getLogger(__name__)


def _stub(node_name: str):
    """Factory: return a stub node function for the given name."""

    def stub_node(state: TradingGraphState) -> dict[str, Any]:
        run_id = state.get("run_id", "unknown")
        trigger = state.get("trigger", "unknown")
        log.info("[stub:%s] running run_id=%s trigger=%s", node_name, run_id, trigger)
        emit(
            run_id=run_id,
            trigger=trigger,
            agent=node_name,
            event_type="stub_executed",
            payload={"node": node_name, "phase": "2.1-skeleton"},
        )
        # Always-safe partial state update: append a notification record
        return {
            "notifications": state.get("notifications", []) + [
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "node": node_name,
                    "kind": "stub_executed",
                }
            ]
        }

    stub_node.__name__ = f"stub_{node_name}"
    return stub_node


# Stubs for each topology slot in §5.2 of the plan.
# Real implementations replace these one at a time in Phase 2.2 → 2.5.

# Premarket
collect_macro_market_data = _stub("collect_macro_market_data")
compute_regime_features = _stub("compute_regime_features")
classify_regime = _stub("classify_regime")
maybe_llm_regime_review = _stub("maybe_llm_regime_review")
persist_regime = _stub("persist_regime")
collect_watchlist_data = _stub("collect_watchlist_data")
rank_candidates = _stub("rank_candidates")
ntfy_scan_digest = _stub("ntfy_scan_digest")

# Candidate entry
load_latest_regime = _stub("load_latest_regime")
research_ticker = _stub("research_ticker")
researcher_debate = _stub("researcher_debate")
retrieve_past_lessons = _stub("retrieve_past_lessons")
create_or_refresh_thesis = _stub("create_or_refresh_thesis")
build_trade_proposal = _stub("build_trade_proposal")
deterministic_sizing = _stub("deterministic_sizing")
regime_execution_gate = _stub("regime_execution_gate")
active_risk_snapshot = _stub("active_risk_snapshot")
deterministic_risk_guardrails = _stub("deterministic_risk_guardrails")
maybe_risk_llm_council = _stub("maybe_risk_llm_council")
finalize_risk_decision = _stub("finalize_risk_decision")
execute_paper_order = _stub("execute_paper_order")
capture_fill = _stub("capture_fill")
persist_trade_event = _stub("persist_trade_event")
ntfy_trade_event = _stub("ntfy_trade_event")
persist_veto = _stub("persist_veto")
ntfy_risk_block = _stub("ntfy_risk_block")
persist_defer = _stub("persist_defer")
ntfy_defer = _stub("ntfy_defer")

# Intraday monitor
opend_health = _stub("opend_health")
load_open_positions = _stub("load_open_positions")
refresh_quotes_and_greeks = _stub("refresh_quotes_and_greeks")
detect_exit_triggers = _stub("detect_exit_triggers")
route_exit_or_hold = _stub("route_exit_or_hold")

# EOD
reconcile_journal = _stub("reconcile_journal")
mark_to_market = _stub("mark_to_market")
persist_daily_marks = _stub("persist_daily_marks")
update_regime_accuracy_labels = _stub("update_regime_accuracy_labels")
generate_eod_digest = _stub("generate_eod_digest")
ntfy_daily_summary = _stub("ntfy_daily_summary")

# Health
postgres_health = _stub("postgres_health")
ntfy_health = _stub("ntfy_health")


def route_risk_decision(state: TradingGraphState) -> str:
    """Conditional router after the risk subgraph.

    Returns the name of the next node based on the risk decision. Until
    Phase 2.3 wires real risk, the stub `finalize_risk_decision` always
    sets a default APPROVE so this routes to executor.
    """
    risk = state.get("risk")
    if not risk:
        return "execute_paper_order"  # phase 2.1 stub default
    decision = risk.get("decision", "DEFER")
    return {
        "APPROVE": "execute_paper_order",
        "DOWNSIZE": "execute_paper_order",
        "VETO": "persist_veto",
        "DEFER": "persist_defer",
    }.get(decision, "persist_defer")
