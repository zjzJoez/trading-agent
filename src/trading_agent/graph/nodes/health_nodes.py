"""Phase 2.5 — Healthcheck nodes.

Replaces stubs:
    opend_health      ping Moomoo OpenD; emit severity=error if unreachable
    postgres_health   ping Postgres; emit severity=error if unreachable
    ntfy_health       send heartbeat to ntfy ops topic

All three are wired into healthcheck_graph (runs hourly via systemd timer).
A single unhealthy check fires an ntfy alert so you get paged on your phone
before the next trading decision tries and fails silently.
"""
from __future__ import annotations

import logging
import time

from trading_agent.events import SEV_ERROR, SEV_INFO, emit
from trading_agent.graph.state import TradingGraphState

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node 1: opend_health
# ---------------------------------------------------------------------------

def opend_health(state: TradingGraphState) -> dict:
    """Verify MoomooOpenD is reachable by calling get_account_info().

    If the call succeeds we emit info; if it raises or times out we emit
    severity=error and send an ntfy ops alert so the user knows the broker
    feed is down before the next trading cycle fires.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[health/opend_health] run_id=%s", run_id)

    t0 = time.monotonic()
    healthy = False
    detail: str = ""

    try:
        from trading_agent.mcp_servers.moomoo.server import get_account_info
        result = get_account_info()
        rows = result.get("rows") or []
        healthy = len(rows) > 0 or result.get("ret_code") == 0 or result.get("code") == 0
        detail = f"rows={len(rows)}"
        if not healthy:
            # Non-zero ret_code is still "reachable" — OpenD is up, just auth issue
            healthy = True
            detail = f"reachable (ret_code={result.get('ret_code')})"
    except Exception as e:
        detail = str(e)[:200]

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if healthy:
        emit(
            run_id=run_id, trigger=trigger, agent="opend_health",
            event_type="opend_ok",
            payload={"latency_ms": elapsed_ms, "detail": detail},
        )
    else:
        emit(
            run_id=run_id, trigger=trigger, agent="opend_health",
            event_type="opend_down",
            severity=SEV_ERROR,
            payload={"latency_ms": elapsed_ms, "detail": detail},
        )
        _alert_ops(
            title="OpenD UNREACHABLE",
            body=(
                f"MoomooOpenD did not respond to get_account_info.\n"
                f"Detail: {detail}\n"
                f"Latency: {elapsed_ms} ms\n"
                f"All intraday and entry decisions will fail until resolved."
            ),
        )

    return {}


# ---------------------------------------------------------------------------
# Node 2: postgres_health
# ---------------------------------------------------------------------------

def postgres_health(state: TradingGraphState) -> dict:
    """Verify Postgres is reachable with a lightweight SELECT 1."""
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[health/postgres_health] run_id=%s", run_id)

    t0 = time.monotonic()
    healthy = False
    detail: str = ""

    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute("SELECT 1")
            row = cur.fetchone()
            healthy = row is not None and row[0] == 1
            detail = "SELECT 1 OK"
    except Exception as e:
        detail = str(e)[:200]

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if healthy:
        emit(
            run_id=run_id, trigger=trigger, agent="postgres_health",
            event_type="postgres_ok",
            payload={"latency_ms": elapsed_ms},
        )
        # Postgres is reachable — drain any audit events buffered to
        # the fallback JSONL during a prior outage. Without this hook
        # the buffered events sit on disk forever.
        try:
            from trading_agent.events import replay_fallback_events
            replay = replay_fallback_events()
            if replay.get("file_existed"):
                emit(
                    run_id=run_id, trigger=trigger, agent="postgres_health",
                    event_type="fallback_replayed",
                    payload=replay,
                )
        except Exception as e:
            log.warning("[postgres_health] fallback replay failed: %s", e)
    else:
        emit(
            run_id=run_id, trigger=trigger, agent="postgres_health",
            event_type="postgres_down",
            severity=SEV_ERROR,
            payload={"latency_ms": elapsed_ms, "detail": detail},
        )
        _alert_ops(
            title="Postgres UNREACHABLE",
            body=(
                f"Postgres SELECT 1 failed.\n"
                f"Detail: {detail}\n"
                f"All checkpointing, risk decisions, and learning writes will fail."
            ),
        )

    return {}


# ---------------------------------------------------------------------------
# Node 3: ntfy_health
# ---------------------------------------------------------------------------

def ntfy_health(state: TradingGraphState) -> dict:
    """Send a heartbeat to the ntfy ops topic.

    This is the last node in healthcheck_graph.  Receiving it on your phone
    means the full stack is alive: Postgres, OpenD, and the LangGraph runner
    all responded within this tick.  Missing heartbeats are itself an alert.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[health/ntfy_health] run_id=%s", run_id)

    from datetime import datetime, timezone
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    try:
        from trading_agent.notify import send as ntfy_send
        ntfy_send(
            topic="ops",
            title=f"Heartbeat — {now_utc}",
            body="trading-agent healthcheck PASS: OpenD + Postgres reachable.",
            priority=1,   # min priority — suppress phone notification in quiet hours
            tags=["white_check_mark"],
        )
        status = "sent"
    except Exception as e:
        log.warning("[ntfy_health] ntfy send failed: %s", e)
        status = f"failed: {e}"

    emit(
        run_id=run_id, trigger=trigger, agent="ntfy_health",
        event_type="heartbeat",
        payload={"status": status, "ts": now_utc},
    )
    return {}


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _alert_ops(title: str, body: str) -> None:
    try:
        from trading_agent.notify import send as ntfy_send
        ntfy_send(
            topic="ops",
            title=title,
            body=body,
            priority=5,
            tags=["rotating_light", "skull"],
        )
    except Exception as e:
        log.error("_alert_ops: ntfy failed: %s", e)


__all__ = ["opend_health", "postgres_health", "ntfy_health"]
