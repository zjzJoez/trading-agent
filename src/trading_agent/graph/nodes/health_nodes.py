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
        # Run two audibility checks now that Postgres is known good.
        # Both are best-effort; any DB or ntfy failure must not bring
        # down healthcheck.
        try:
            _check_dispatch_silent_die(run_id, trigger)
        except Exception as e:
            log.warning("[postgres_health] dispatch watchdog failed: %s", e)
        try:
            _check_llm_schema_violations(run_id, trigger)
        except Exception as e:
            log.warning("[postgres_health] llm schema watchdog failed: %s", e)
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


# ---------------------------------------------------------------------------
# Audibility watchdogs — these turn "silent failure" into "loud failure".
# Run from postgres_health (once per healthcheck tick = hourly) when DB is up.
# ---------------------------------------------------------------------------

# Dispatch watchdog: a candidate_entry that's been dispatched > this many
# minutes ago but never emitted a matching `run_start` event is presumed
# silently dead (cgroup kill, module-import crash, etc.). 5 min is generous —
# the candidate_entry graph normally emits run_start within ~10 s of
# systemd starting the unit.
_DISPATCH_SILENT_DIE_MIN = 5
_DISPATCH_LOOKBACK_HOURS = 6   # don't alert about stale history
# Suppress repeat alerts for the same ticker within this window.
_DISPATCH_ALERT_COOLDOWN_MIN = 60

# LLM schema-violation alert: ≥ this many `*_schema_violation` events in
# the lookback window across all roles → audible alert. The router emits
# one of these per role per failed call; 5+ in 60 min means a channel is
# silently fabricating output and our prompts are landing in the wrong
# place. Mirrors the exit_monitor escalation pattern.
_LLM_SV_WINDOW_MIN = 60
_LLM_SV_THRESHOLD = 5
_LLM_SV_ALERT_COOLDOWN_MIN = 60


def _check_dispatch_silent_die(run_id: str, trigger: str) -> None:
    """Find any candidate_entry_dispatched event whose ticker has no matching
    `run_start` event within the silent-die window. Emit + ntfy on each.

    SQL is a LEFT JOIN against run_start events filtered to the same ticker;
    NULL on the right side means the dispatched child never reached the
    orchestrator's emit-run_start step.
    """
    from trading_agent.store.postgres import cursor

    with cursor() as cur:
        cur.execute(
            """
            WITH recent_dispatches AS (
                SELECT
                    ts AS dispatched_at,
                    payload->>'ticker' AS ticker
                FROM agent_events
                WHERE event_type = 'candidate_entry_dispatched'
                  AND ts > NOW() - (%s::text || ' hours')::interval
                  AND ts < NOW() - (%s::text || ' minutes')::interval
            ),
            recent_starts AS (
                SELECT
                    ts AS started_at,
                    payload->>'ticker' AS ticker
                FROM agent_events
                WHERE event_type = 'run_start'
                  AND trigger = 'candidate_entry'
                  AND ts > NOW() - (%s::text || ' hours')::interval
            ),
            recent_alerts AS (
                SELECT payload->>'ticker' AS ticker
                FROM agent_events
                WHERE event_type = 'candidate_entry_silent_die'
                  AND ts > NOW() - (%s::text || ' minutes')::interval
            )
            SELECT
                d.ticker,
                EXTRACT(EPOCH FROM (NOW() - d.dispatched_at))::int AS age_seconds
            FROM recent_dispatches d
            LEFT JOIN recent_starts s
                ON s.ticker = d.ticker
                AND s.started_at >= d.dispatched_at
                AND s.started_at <= d.dispatched_at + interval '10 minutes'
            WHERE s.ticker IS NULL
              AND d.ticker NOT IN (SELECT ticker FROM recent_alerts WHERE ticker IS NOT NULL)
            ORDER BY d.dispatched_at DESC
            LIMIT 5
            """,
            (
                str(_DISPATCH_LOOKBACK_HOURS),
                str(_DISPATCH_SILENT_DIE_MIN),
                str(_DISPATCH_LOOKBACK_HOURS),
                str(_DISPATCH_ALERT_COOLDOWN_MIN),
            ),
        )
        rows = cur.fetchall()

    for ticker, age_seconds in rows:
        emit(
            run_id=run_id, trigger=trigger, agent="postgres_health",
            event_type="candidate_entry_silent_die",
            severity=2,
            payload={
                "ticker": ticker,
                "dispatched_age_seconds": int(age_seconds),
                "silent_die_threshold_min": _DISPATCH_SILENT_DIE_MIN,
            },
        )
        _alert_ops(
            title=f"Silent dispatch death — {ticker}",
            body=(
                f"candidate_entry was dispatched {int(age_seconds)//60} min ago "
                f"for {ticker} but the child never emitted run_start.\n\n"
                f"Likely a systemd cgroup kill or module-import crash. "
                f"Check: journalctl -u trading-agent-candidate-entry@{ticker}.service"
            ),
        )


def _check_llm_schema_violations(run_id: str, trigger: str) -> None:
    """Count distinct `*_schema_violation` audit rows in the lookback window;
    alert when total ≥ threshold and no recent alert covers it.

    Channels matter: an alert tells the operator which role is producing
    garbage so they know whether to suspect Claude-Code OAuth, Codex Plus,
    or DeepSeek fallback. We group by role for the alert body.
    """
    from trading_agent.store.postgres import cursor

    with cursor() as cur:
        cur.execute(
            """
            SELECT payload->>'role' AS role, COUNT(*) AS n
            FROM agent_events
            WHERE event_type = 'llm_schema_violation'
              AND ts > NOW() - (%s::text || ' minutes')::interval
            GROUP BY 1
            ORDER BY n DESC
            """,
            (str(_LLM_SV_WINDOW_MIN),),
        )
        per_role = cur.fetchall()
        total = sum(int(n) for _role, n in per_role)
        if total < _LLM_SV_THRESHOLD:
            return

        cur.execute(
            """
            SELECT 1 FROM agent_events
            WHERE event_type = 'llm_schema_violation_alert'
              AND ts > NOW() - (%s::text || ' minutes')::interval
            LIMIT 1
            """,
            (str(_LLM_SV_ALERT_COOLDOWN_MIN),),
        )
        if cur.fetchone() is not None:
            return  # already alerted within cooldown

    role_breakdown = ", ".join(f"{role}={n}" for role, n in per_role)
    emit(
        run_id=run_id, trigger=trigger, agent="postgres_health",
        event_type="llm_schema_violation_alert",
        severity=2,
        payload={
            "total": total,
            "window_minutes": _LLM_SV_WINDOW_MIN,
            "threshold": _LLM_SV_THRESHOLD,
            "per_role": dict(per_role),
        },
    )
    _alert_ops(
        title=f"LLM channel producing garbage — {total} schema violations",
        body=(
            f"{total} LLM schema-violation events in last {_LLM_SV_WINDOW_MIN} min "
            f"(threshold={_LLM_SV_THRESHOLD}).\n\n"
            f"Per-role: {role_breakdown}\n\n"
            f"This usually means an OAuth subprocess (claude-code / codex) "
            f"is returning empty stdout or echoing the prompt back. "
            f"Trader proposals and exit decisions are silently degraded. "
            f"Check OAuth session liveness."
        ),
    )


__all__ = ["opend_health", "postgres_health", "ntfy_health"]
