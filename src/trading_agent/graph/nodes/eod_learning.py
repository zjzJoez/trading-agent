"""Phase 2.6 — EOD enrichment node bridging into the learning loop."""
from __future__ import annotations

import logging

from trading_agent.events import emit
from trading_agent.graph.state import TradingGraphState
from trading_agent.learning.outcome import enrich_closed_trades

log = logging.getLogger(__name__)


def enrich_outcomes_node(state: TradingGraphState) -> dict:
    """Compute realized stats for any newly-closed trades."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[learning/enrich_outcomes] run_id=%s", run_id)

    try:
        ids = enrich_closed_trades(limit=50)
    except Exception as e:
        log.warning("[learning/enrich_outcomes] failed: %s", e)
        ids = []

    emit(
        run_id=run_id, trigger=trigger, agent="enrich_outcomes",
        event_type="outcomes_enriched",
        payload={"trade_ids": ids, "n": len(ids)},
    )
    return {}


__all__ = ["enrich_outcomes_node"]
