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


def _fallback_path():
    from pathlib import Path
    return Path.home() / "agent_events.fallback.jsonl"


def replay_fallback_events(max_lines: int = 5_000) -> dict:
    """Drain ``~/agent_events.fallback.jsonl`` back into Postgres ``agent_events``.

    Why: ``emit()`` writes here when Postgres is unreachable so decisions are
    still recorded. But without a replay step the data sits there forever —
    a real Postgres blip silently drops audit rows.

    Mechanics:
        * Read whole file (capped at ``max_lines``).
        * Truncate the file BEFORE attempting inserts. If we insert-first
          then truncate, a crash mid-replay double-counts rows. If we
          truncate-first then crash, we lose at most one batch — same
          failure mode as a Postgres outage that started one tick later.
        * On per-row insert failure, append the failed row back to the
          file so the next replay tick picks it up.

    Returns ``{replayed, failed, file_existed}``. Idempotent. Safe to call
    from healthcheck every tick — if the file is missing/empty, returns in
    one ``stat()`` call.
    """
    out = {"replayed": 0, "failed": 0, "file_existed": False}
    path = _fallback_path()
    if not path.exists() or path.stat().st_size == 0:
        return out
    out["file_existed"] = True

    try:
        lines = path.read_text().splitlines()
    except Exception as e:
        log.warning("replay_fallback_events: read failed: %s", e)
        return out

    if len(lines) > max_lines:
        # Huge file — leave the tail for the next tick so healthcheck stays fast.
        replay_lines, leftover = lines[:max_lines], lines[max_lines:]
    else:
        replay_lines, leftover = lines, []

    try:
        path.write_text("\n".join(leftover) + ("\n" if leftover else ""))
    except Exception as e:
        log.warning("replay_fallback_events: truncate failed: %s — bailing", e)
        return out

    requeue: list[str] = []
    for line in replay_lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            # malformed — drop (not worth requeuing forever)
            out["failed"] += 1
            continue
        try:
            with cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_events
                        (ts, run_id, trigger, agent, event_type, severity, payload, cost_usd)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                    """,
                    (
                        rec.get("ts"),
                        rec.get("run_id"),
                        rec.get("trigger"),
                        rec.get("agent"),
                        rec.get("event_type"),
                        rec.get("severity", 0),
                        json.dumps(rec.get("payload") or {}, default=_json_default),
                        rec.get("cost_usd"),
                    ),
                )
            out["replayed"] += 1
        except Exception as e:
            log.warning(
                "replay_fallback_events: row insert failed (will retry next tick): %s", e,
            )
            requeue.append(line)
            out["failed"] += 1

    if requeue:
        try:
            with open(path, "a") as f:
                for line in requeue:
                    f.write(line + "\n")
        except Exception as e:
            log.error(
                "replay_fallback_events: requeue write failed — events lost: %s", e,
            )

    if out["replayed"] or out["failed"]:
        log.info(
            "replay_fallback_events: replayed=%d failed=%d leftover=%d",
            out["replayed"], out["failed"], len(leftover),
        )
    return out


def _json_default(o: Any) -> Any:
    """Pydantic models, datetime, decimal, etc. — keep audit JSON serializable."""
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if isinstance(o, datetime):
        return o.isoformat()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)
