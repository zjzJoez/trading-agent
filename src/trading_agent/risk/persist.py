"""Persist risk_snapshots + risk_decisions to Postgres.

Tables defined in migrations/002_phase2.sql.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from trading_agent.risk.agent import RiskOutput
from trading_agent.risk.portfolio import PortfolioSnapshot
from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)


def insert_risk_snapshot(
    snap: PortfolioSnapshot, *, regime_state_id: int | None = None
) -> int:
    """Persist a portfolio snapshot. Returns the new row id."""
    payload = snap.to_db_payload()
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO risk_snapshots
                (created_at, regime_state_id, equity, open_positions,
                 correlations, factor_exposures, aggregate_greeks,
                 heat, liquidity, data_quality)
            VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb,
                    %s::jsonb, %s::jsonb, %s::jsonb)
            RETURNING id
            """,
            (
                snap.as_of,
                regime_state_id,
                snap.equity,
                json.dumps(payload["open_positions"], default=str),
                json.dumps(payload["correlations"]),
                json.dumps(payload["factor_exposures"]),
                json.dumps(payload["aggregate_greeks"]),
                json.dumps(payload["heat_metrics"]),
                json.dumps(payload["liquidity_warnings"]),
                json.dumps(payload["data_quality"]),
            ),
        )
        return int(cur.fetchone()[0])


def insert_risk_decision(
    out: RiskOutput,
    *,
    proposal: dict[str, Any],
    thesis_id: int | None,
    regime_state_id: int | None,
    risk_snapshot_id: int | None,
) -> int:
    """Persist the final risk decision."""
    approved = {
        "qty": out.approved_qty,
        "max_entry_price": out.max_entry_price,
    }
    llm_reviews = None
    if out.council_result:
        llm_reviews = {
            "reviewers": [
                {
                    "role": r.role,
                    "decision": r.decision,
                    "primary_risks": r.primary_risks,
                    "reason": r.reason,
                    "cost_usd": r.cost_usd,
                }
                for r in out.council_result.reviewers
            ],
            "arbiter_reason": out.council_result.arbiter_reason,
            "approved_factor": out.council_result.approved_qty_factor,
            "total_cost_usd": out.council_result.total_cost_usd,
        }

    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO risk_decisions
                (risk_decision_id, expires_at, proposal_id, thesis_id,
                 regime_state_id, risk_snapshot_id, decision,
                 proposed, approved, hard_violations, llm_reviews,
                 reasons)
            VALUES (%s, %s, %s, %s, %s, %s, %s,
                    %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
            RETURNING id
            """,
            (
                out.risk_decision_id,
                out.expires_at,
                proposal.get("proposal_id", out.risk_decision_id),
                thesis_id,
                regime_state_id,
                risk_snapshot_id,
                out.decision,
                json.dumps(proposal, default=str),
                json.dumps(approved),
                json.dumps(out.hard_violations),
                json.dumps(llm_reviews) if llm_reviews else None,
                json.dumps(out.reasons),
            ),
        )
        return int(cur.fetchone()[0])
