"""agent_events — append-only domain audit log.

Every node in every LangGraph subgraph emits events here. This is the
human-readable companion to LangGraph's checkpoint store: checkpoints
recover machine state, agent_events explain the decisions to humans
(and replay tooling).

Usage:
    from trading_agent.events import emit
    emit(run_id="...", trigger="candidate_entry", agent="risk_arbiter",
         event_type="decision", payload={"decision": "VETO", ...},
         severity=2)
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)

SEV_INFO = 0
SEV_WARN = 1
SEV_ERROR = 2
SEV_CRITICAL = 3


def emit(
    *,
    run_id: str,
    trigger: str,
    agent: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
    severity: int = SEV_INFO,
    cost_usd: float | None = None,
    ts: datetime | None = None,
) -> int:
    """Append an event to agent_events. Returns the new row id.

    Never raises — if Postgres is down, falls back to stderr log so the
    caller's flow doesn't break. (Trading decisions must continue even if
    audit storage is degraded.)
    """
    payload = payload or {}
    try:
        with cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_events
                    (ts, run_id, trigger, agent, event_type, severity, payload, cost_usd)
                VALUES (COALESCE(%s, NOW()), %s, %s, %s, %s, %s, %s::jsonb, %s)
                RETURNING id
                """,
                (
                    ts,
                    run_id,
                    trigger,
                    agent,
                    event_type,
                    severity,
                    json.dumps(payload, default=_json_default),
                    cost_usd,
                ),
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0
    except Exception as e:
        log.error(
            "agent_events emit failed (degraded): run=%s agent=%s evt=%s err=%s",
            run_id, agent, event_type, e,
        )
        # Best-effort: append to a fallback file so we don't lose audit data.
        try:
            from pathlib import Path
            fallback = Path.home() / "agent_events.fallback.jsonl"
            with open(fallback, "a") as f:
                f.write(json.dumps({
                    "ts": (ts or datetime.now(timezone.utc)).isoformat(),
                    "run_id": run_id,
                    "trigger": trigger,
                    "agent": agent,
                    "event_type": event_type,
                    "severity": severity,
                    "payload": payload,
                    "cost_usd": cost_usd,
                }, default=_json_default) + "\n")
        except Exception:
            pass
        return 0


def new_run_id(prefix: str = "run") -> str:
    """Generate a ULID-like run id (sortable by time, opaque to humans)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{rand}"


def _json_default(o: Any) -> Any:
    """Pydantic models, datetime, decimal, etc. — keep audit JSON serializable."""
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if isinstance(o, datetime):
        return o.isoformat()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)
