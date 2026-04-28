"""EOD + intraday learning-loop nodes."""
from __future__ import annotations

import logging

from trading_agent.events import emit
from trading_agent.graph.state import TradingGraphState
from trading_agent.learning.excursion import update_excursions_once
from trading_agent.learning.outcome import enrich_closed_trades
from trading_agent.learning.promote import evaluate_and_apply_all

log = logging.getLogger(__name__)


def enrich_outcomes_node(state: TradingGraphState) -> dict:
    """Phase 2.6 — compute realized stats for any newly-closed trades."""
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


def promote_or_rollback_node(state: TradingGraphState) -> dict:
    """Phase 2.7 — evaluate every CANARY param_version and act.

    Auto-rollback fires for hard-risk breaches in the first 5 trades or
    a 5-day losing streak.  Promotion to ACTIVE fires only after at least
    20 trades and Wilson LB / profit-factor / drawdown gates all pass.
    Otherwise the row is RETAINED and the traffic ramp may bump it from
    10% → 20% via ``canary.maybe_ramp_traffic``.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[learning/promote_or_rollback] run_id=%s", run_id)

    try:
        decisions = evaluate_and_apply_all()
    except Exception as e:
        log.warning("[promote_or_rollback] failed: %s", e)
        decisions = []

    by_decision: dict[str, int] = {}
    for d in decisions:
        by_decision[d["decision"]] = by_decision.get(d["decision"], 0) + 1

    emit(
        run_id=run_id, trigger=trigger, agent="promote_or_rollback",
        event_type="canary_evaluation",
        payload={
            "n_canaries": len(decisions),
            "by_decision": by_decision,
            "decisions": decisions,
        },
    )
    return {}


def update_excursions_node(state: TradingGraphState) -> dict:
    """Phase 2.7.5 — refresh MAE / MFE for every open position.

    Wired into ``intraday_monitor_graph`` after ``refresh_quotes_and_greeks``.
    Best-effort; OpenD or DB failures degrade to a logged WARN without
    blocking the rest of the monitor pass.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[learning/update_excursions] run_id=%s", run_id)

    try:
        updates = update_excursions_once()
    except Exception as e:
        log.warning("[update_excursions] failed: %s", e)
        updates = []

    emit(
        run_id=run_id, trigger=trigger, agent="update_excursions",
        event_type="excursions_updated",
        payload={
            "n_open_positions": len(updates),
            "snapshots": [
                {
                    "trade_id": u.trade_id,
                    "symbol": u.symbol,
                    "realized_R_now": round(u.realized_R_now, 4),
                    "mae_so_far": round(u.mae_so_far, 4),
                    "mfe_so_far": round(u.mfe_so_far, 4),
                }
                for u in updates
            ],
        },
    )
    return {}


__all__ = [
    "enrich_outcomes_node",
    "promote_or_rollback_node",
    "update_excursions_node",
]
