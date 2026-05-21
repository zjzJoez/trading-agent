"""Compile the LangGraph subgraphs.

Phase 2.5+ — all nodes are real implementations. Stubs.py is kept for
future scaffolding but no longer imported here.

Subgraph → primary module mapping:
    premarket_scan_graph   regime_nodes + premarket_nodes
    candidate_entry_graph  trade_nodes + risk_nodes + learning_nodes
    intraday_monitor_graph intraday_nodes + eod_learning + health_nodes
    eod_review_graph       eod_nodes + eod_learning
    healthcheck_graph      health_nodes
    weekly_learning_graph  weekly_learning
"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from trading_agent.graph.checkpointer import get_saver
from trading_agent.graph.nodes import eod_learning as EL
from trading_agent.graph.nodes import eod_nodes as EOD
from trading_agent.graph.nodes import health_nodes as H
from trading_agent.graph.nodes import intraday_nodes as IN
from trading_agent.graph.nodes import learning_nodes as L
from trading_agent.graph.nodes import premarket_nodes as PM
from trading_agent.graph.nodes import regime_nodes as R
from trading_agent.graph.nodes import risk_nodes as RK
from trading_agent.graph.nodes import trade_nodes as T
from trading_agent.graph.nodes import weekly_learning as WL
from trading_agent.graph.state import TradingGraphState

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# premarket_scan_graph
# -----------------------------------------------------------------------------
def build_premarket_scan_graph():
    g = StateGraph(TradingGraphState)
    g.add_node("collect_macro_market_data", R.collect_macro_market_data)
    g.add_node("compute_regime_features", R.compute_regime_features)
    g.add_node("classify_regime", R.classify_regime)
    g.add_node("maybe_llm_regime_review", R.maybe_llm_regime_review)
    g.add_node("persist_regime", R.persist_regime)
    g.add_node("collect_watchlist_data", PM.collect_watchlist_data)
    g.add_node("rank_candidates", PM.rank_candidates)
    g.add_node("ntfy_scan_digest", PM.ntfy_scan_digest)

    g.add_edge(START, "collect_macro_market_data")
    g.add_edge("collect_macro_market_data", "compute_regime_features")
    g.add_edge("compute_regime_features", "classify_regime")
    g.add_edge("classify_regime", "maybe_llm_regime_review")
    g.add_edge("maybe_llm_regime_review", "persist_regime")
    g.add_edge("persist_regime", "collect_watchlist_data")
    g.add_edge("collect_watchlist_data", "rank_candidates")
    g.add_edge("rank_candidates", "ntfy_scan_digest")
    g.add_edge("ntfy_scan_digest", END)

    return g.compile(checkpointer=get_saver())


# -----------------------------------------------------------------------------
# candidate_entry_graph
# -----------------------------------------------------------------------------
def build_candidate_entry_graph():
    g = StateGraph(TradingGraphState)
    for name, fn in [
        ("load_active_params", L.load_active_params_node),
        ("load_latest_regime", T.load_latest_regime),
        ("load_open_positions", T.load_open_positions),
        ("research_ticker", T.research_ticker),
        ("assign_canary", L.assign_canary_node),
        ("researcher_debate", T.researcher_debate),
        ("retrieve_past_lessons", T.retrieve_past_lessons),
        ("build_trade_proposal", T.build_trade_proposal),
        ("create_or_refresh_thesis", T.create_or_refresh_thesis),
        ("deterministic_sizing", T.deterministic_sizing),
        ("shadow_track", L.shadow_track_node),
        ("regime_execution_gate", T.regime_execution_gate),
        ("active_risk_snapshot", RK.active_risk_snapshot),
        ("deterministic_risk_guardrails", RK.deterministic_risk_guardrails),
        ("maybe_risk_llm_council", RK.maybe_risk_llm_council),
        ("finalize_risk_decision", RK.finalize_risk_decision),
        ("execute_paper_order", T.execute_paper_order),
        ("capture_fill", T.capture_fill),
        ("persist_trade_event", T.persist_trade_event),
        ("ntfy_trade_event", T.ntfy_trade_event),
        ("persist_veto", T.persist_veto),
        ("ntfy_risk_block", T.ntfy_risk_block),
        ("persist_defer", T.persist_defer),
        ("ntfy_defer", T.ntfy_defer),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "load_active_params")
    g.add_edge("load_active_params", "load_latest_regime")
    g.add_edge("load_latest_regime", "load_open_positions")
    g.add_edge("load_open_positions", "research_ticker")
    g.add_edge("research_ticker", "assign_canary")
    g.add_edge("assign_canary", "researcher_debate")
    g.add_edge("researcher_debate", "retrieve_past_lessons")
    g.add_edge("retrieve_past_lessons", "build_trade_proposal")
    g.add_edge("build_trade_proposal", "create_or_refresh_thesis")
    g.add_edge("create_or_refresh_thesis", "deterministic_sizing")
    g.add_edge("deterministic_sizing", "shadow_track")
    g.add_edge("shadow_track", "regime_execution_gate")
    g.add_edge("regime_execution_gate", "active_risk_snapshot")
    g.add_edge("active_risk_snapshot", "deterministic_risk_guardrails")
    g.add_edge("deterministic_risk_guardrails", "maybe_risk_llm_council")
    g.add_edge("maybe_risk_llm_council", "finalize_risk_decision")

    g.add_conditional_edges(
        "finalize_risk_decision",
        RK.route_risk_decision,
        {
            "execute_paper_order": "execute_paper_order",
            "persist_veto": "persist_veto",
            "persist_defer": "persist_defer",
        },
    )

    g.add_edge("execute_paper_order", "capture_fill")
    g.add_edge("capture_fill", "persist_trade_event")
    g.add_edge("persist_trade_event", "ntfy_trade_event")
    g.add_edge("ntfy_trade_event", END)

    g.add_edge("persist_veto", "ntfy_risk_block")
    g.add_edge("ntfy_risk_block", END)

    g.add_edge("persist_defer", "ntfy_defer")
    g.add_edge("ntfy_defer", END)

    return g.compile(checkpointer=get_saver())


# -----------------------------------------------------------------------------
# intraday_monitor_graph
# -----------------------------------------------------------------------------
def build_intraday_monitor_graph():
    g = StateGraph(TradingGraphState)
    g.add_node("load_active_params", L.load_active_params_node)
    g.add_node("opend_health", H.opend_health)
    g.add_node("load_open_positions", T.load_open_positions)
    g.add_node("refresh_quotes_and_greeks", IN.refresh_quotes_and_greeks)
    g.add_node("update_excursions", EL.update_excursions_node)
    g.add_node("load_latest_regime", T.load_latest_regime)
    g.add_node("active_risk_snapshot", RK.active_risk_snapshot)
    g.add_node("detect_exit_triggers", IN.detect_exit_triggers)
    g.add_node("route_exit_or_hold", IN.route_exit_or_hold)

    g.add_edge(START, "load_active_params")
    g.add_edge("load_active_params", "opend_health")
    g.add_edge("opend_health", "load_open_positions")
    g.add_edge("load_open_positions", "refresh_quotes_and_greeks")
    g.add_edge("refresh_quotes_and_greeks", "update_excursions")
    g.add_edge("update_excursions", "load_latest_regime")
    g.add_edge("load_latest_regime", "active_risk_snapshot")
    g.add_edge("active_risk_snapshot", "detect_exit_triggers")
    g.add_edge("detect_exit_triggers", "route_exit_or_hold")
    g.add_edge("route_exit_or_hold", END)

    return g.compile(checkpointer=get_saver())


# -----------------------------------------------------------------------------
# eod_review_graph
# -----------------------------------------------------------------------------
def build_eod_review_graph():
    g = StateGraph(TradingGraphState)
    g.add_node("reconcile_journal", EOD.reconcile_journal)
    g.add_node("auto_void_stale_theses", EOD.auto_void_stale_theses)
    g.add_node("mark_to_market", EOD.mark_to_market)
    g.add_node("persist_daily_marks", EOD.persist_daily_marks)
    g.add_node("update_regime_accuracy_labels", EOD.update_regime_accuracy_labels)
    g.add_node("enrich_outcomes", EL.enrich_outcomes_node)
    g.add_node("promote_or_rollback", EL.promote_or_rollback_node)
    g.add_node("generate_eod_digest", EOD.generate_eod_digest)
    g.add_node("ntfy_daily_summary", EOD.ntfy_daily_summary)

    g.add_edge(START, "reconcile_journal")
    g.add_edge("reconcile_journal", "auto_void_stale_theses")
    g.add_edge("auto_void_stale_theses", "mark_to_market")
    g.add_edge("mark_to_market", "persist_daily_marks")
    g.add_edge("persist_daily_marks", "update_regime_accuracy_labels")
    g.add_edge("update_regime_accuracy_labels", "enrich_outcomes")
    g.add_edge("enrich_outcomes", "promote_or_rollback")
    g.add_edge("promote_or_rollback", "generate_eod_digest")
    g.add_edge("generate_eod_digest", "ntfy_daily_summary")
    g.add_edge("ntfy_daily_summary", END)

    return g.compile(checkpointer=get_saver())


# -----------------------------------------------------------------------------
# healthcheck_graph
# -----------------------------------------------------------------------------
def build_healthcheck_graph():
    g = StateGraph(TradingGraphState)
    g.add_node("opend_health", H.opend_health)
    g.add_node("postgres_health", H.postgres_health)
    g.add_node("ntfy_health", H.ntfy_health)

    g.add_edge(START, "opend_health")
    g.add_edge("opend_health", "postgres_health")
    g.add_edge("postgres_health", "ntfy_health")
    g.add_edge("ntfy_health", END)

    return g.compile(checkpointer=get_saver())


# -----------------------------------------------------------------------------
# weekly_learning_graph — Saturday post-close LLM Critic loop
# -----------------------------------------------------------------------------
def build_weekly_learning_graph():
    g = StateGraph(TradingGraphState)
    g.add_node("collect_outcomes_weekly", WL.collect_outcomes_node)
    g.add_node("critic_propose", WL.critic_propose_node)
    g.add_node("replay_validate", WL.replay_validate_node)
    g.add_node("promote_to_canary", WL.promote_to_canary_node)
    g.add_node("ntfy_learning_digest", WL.ntfy_learning_digest_node)

    g.add_edge(START, "collect_outcomes_weekly")
    g.add_edge("collect_outcomes_weekly", "critic_propose")
    g.add_edge("critic_propose", "replay_validate")
    g.add_edge("replay_validate", "promote_to_canary")
    g.add_edge("promote_to_canary", "ntfy_learning_digest")
    g.add_edge("ntfy_learning_digest", END)

    return g.compile(checkpointer=get_saver())


# -----------------------------------------------------------------------------
# Registry — used by orchestrator dispatcher
# -----------------------------------------------------------------------------
GRAPH_BUILDERS = {
    "premarket_scan": build_premarket_scan_graph,
    "candidate_entry": build_candidate_entry_graph,
    "intraday_monitor": build_intraday_monitor_graph,
    "eod_review": build_eod_review_graph,
    "weekly_learning": build_weekly_learning_graph,
    "healthcheck": build_healthcheck_graph,
}


def build(trigger: str):
    """Return a compiled graph for the given trigger."""
    if trigger not in GRAPH_BUILDERS:
        raise KeyError(f"Unknown trigger: {trigger}. Known: {list(GRAPH_BUILDERS)}")
    return GRAPH_BUILDERS[trigger]()
