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
# Node 2b: disk_health
# ---------------------------------------------------------------------------

# Tuned for a 49G EC2 root volume. 80% = ~10G headroom remaining, still safe
# for a typical daily polymarket backup spike. 95% = critical: Postgres has
# died before at 100% (5/19 incident — "could not write init file"), so we
# want a phone-buzzing alert at 95 with ~5 min of action runway before things
# break, not a polite reminder.
DISK_WARN_PCT = 80
DISK_CRIT_PCT = 95


def disk_health(state: TradingGraphState) -> dict:
    """Check root filesystem usage. Ntfy ops if usage exceeds thresholds.

    Tiers:
      <80%  → silent OK
      80-94 → priority=4 warning (sound, but not urgent)
      ≥95%  → priority=5 critical (alarm), severity=error

    The 5/19/2026 outage (Postgres died with disk at 100%) is what motivated
    this node — there was no early-warning signal before PG started failing.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[health/disk_health] run_id=%s", run_id)

    import shutil
    t0 = time.monotonic()
    usage = shutil.disk_usage("/")
    used_pct = int((usage.used / usage.total) * 100)
    free_gb = round(usage.free / (1024 ** 3), 1)
    used_gb = round(usage.used / (1024 ** 3), 1)
    total_gb = round(usage.total / (1024 ** 3), 1)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    payload = {
        "used_pct": used_pct,
        "used_gb": used_gb,
        "free_gb": free_gb,
        "total_gb": total_gb,
        "latency_ms": elapsed_ms,
    }

    if used_pct >= DISK_CRIT_PCT:
        emit(
            run_id=run_id, trigger=trigger, agent="disk_health",
            event_type="disk_critical",
            severity=SEV_ERROR,
            payload=payload,
        )
        _alert_ops(
            title=f"DISK CRITICAL — {used_pct}% used",
            body=(
                f"Root filesystem at {used_pct}% ({used_gb}G/{total_gb}G).\n"
                f"Free: {free_gb}G. Postgres will fail to write very soon.\n"
                f"Investigate /home/ubuntu/{{backups,polymarket-mvp/var}} and /tmp."
            ),
        )
    elif used_pct >= DISK_WARN_PCT:
        emit(
            run_id=run_id, trigger=trigger, agent="disk_health",
            event_type="disk_high",
            severity=SEV_INFO,
            payload=payload,
        )
        # Priority 4 (sound notification) but NOT priority 5 (alarm).
        try:
            from trading_agent.notify import send as ntfy_send
            ntfy_send(
                topic="ops",
                title=f"Disk {used_pct}% — investigate within hours",
                body=(
                    f"Root filesystem at {used_pct}% ({used_gb}G/{total_gb}G).\n"
                    f"Free: {free_gb}G. Not critical yet but trend is bad.\n"
                    f"`df -h` + `du -sh /home/ubuntu/* /tmp/*` to investigate."
                ),
                priority=4,
                tags=["warning", "floppy_disk"],
            )
        except Exception as e:
            log.warning("[disk_health] ntfy send failed: %s", e)
    else:
        emit(
            run_id=run_id, trigger=trigger, agent="disk_health",
            event_type="disk_ok",
            payload=payload,
        )

    return {}


# ---------------------------------------------------------------------------
# Node 2c: failed_units_check
# ---------------------------------------------------------------------------

# Unit prefixes worth alerting on. We exclude well-known noisy units (man-db
# rebuilds, logrotate transient failures) that recover on their own. Anything
# starting with "trading-agent" or one of these prefixes will fire an alert
# when it lands in `systemctl --failed`.
TRACKED_UNIT_PREFIXES = (
    "trading-agent-",   # our own brain/dispatch/auto-deploy
    "polymarket-",      # sibling project (shared disk; failures often correlate)
    "alpha-lab-",       # sibling project
    "postgresql",       # the foundation
    "db-backup",        # daily DB backup
)


# A persistent failed unit will keep showing up on every healthcheck tick.
# Without a cooldown, we'd ntfy every hour for the same problem until the
# operator notices (5/22 incident: 4 hourly sev-2 alerts for watchlist-
# refresh in a failed state). 6h cooldown is a good balance: long enough
# that ops doesn't get spammed, short enough that a re-failure after
# a fix isn't silenced overnight.
FAILED_UNITS_NTFY_COOLDOWN_HOURS = 6


def _was_failed_units_alerted_recently(failed_units: list[str]) -> bool:
    """True iff we ntfy'd for THIS EXACT failed-units set within cooldown.

    Querying agent_events is best-effort: any DB error → return False
    (alert as a fallback). Keying on the sorted unit set means a NEW
    failure (different unit) still alerts immediately even if an older
    failure is still in cooldown.
    """
    if not failed_units:
        return False
    key = ",".join(sorted(set(failed_units)))
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM agent_events
                WHERE event_type = 'failed_units_detected'
                  AND ts > NOW() - (%s::text || ' hours')::interval
                  AND payload ? 'ntfy_key'
                  AND payload->>'ntfy_key' = %s
                LIMIT 1
                """,
                (str(FAILED_UNITS_NTFY_COOLDOWN_HOURS), key),
            )
            return cur.fetchone() is not None
    except Exception as e:
        log.warning("[failed_units_check] cooldown check failed: %s", e)
        return False


def failed_units_check(state: TradingGraphState) -> dict:
    """Scan systemctl --failed for tracked units and ntfy if any are failing.

    Caught the 5/19 outage post-hoc — db-backup, alpha-lab-clv, alpha-lab-odds,
    logrotate, and man-db were all silently failed for days before anyone
    noticed. This node flips that to a phone alert within the next healthcheck
    tick (worst case 1 hour).

    Ntfy is suppressed within ``FAILED_UNITS_NTFY_COOLDOWN_HOURS`` for the
    exact same failed-unit set so a persistent failure doesn't spam
    hourly alerts (5/22 watchlist-refresh incident). The event row is
    still emitted every tick for audit / dashboard visibility.
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[health/failed_units_check] run_id=%s", run_id)

    import subprocess
    t0 = time.monotonic()
    failed_units: list[str] = []
    try:
        r = subprocess.run(
            ["systemctl", "list-units", "--failed", "--no-legend", "--plain"],
            capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            # Format: "unit.service loaded failed failed Description..."
            unit_name = line.split()[0] if line.split() else ""
            if unit_name and any(unit_name.startswith(p) for p in TRACKED_UNIT_PREFIXES):
                failed_units.append(unit_name)
    except Exception as e:
        log.warning("[failed_units_check] systemctl call failed: %s", e)

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if not failed_units:
        emit(
            run_id=run_id, trigger=trigger, agent="failed_units_check",
            event_type="failed_units_clean",
            payload={"latency_ms": elapsed_ms},
        )
        return {}

    # Did we already alert for this exact set inside the cooldown window?
    suppress_ntfy = _was_failed_units_alerted_recently(failed_units)
    ntfy_key = ",".join(sorted(set(failed_units)))

    emit(
        run_id=run_id, trigger=trigger, agent="failed_units_check",
        event_type="failed_units_detected",
        severity=SEV_ERROR,
        payload={
            "count": len(failed_units),
            "units": failed_units,
            "latency_ms": elapsed_ms,
            "ntfy_key": ntfy_key,
            "ntfy_suppressed": suppress_ntfy,
            "cooldown_hours": FAILED_UNITS_NTFY_COOLDOWN_HOURS,
        },
    )

    if suppress_ntfy:
        log.info(
            "[failed_units_check] ntfy suppressed for %s (cooldown %dh)",
            ntfy_key, FAILED_UNITS_NTFY_COOLDOWN_HOURS,
        )
        return {}

    _alert_ops(
        title=f"{len(failed_units)} tracked unit(s) FAILED",
        body=(
            "Failed systemd units detected:\n"
            + "\n".join(f"  • {u}" for u in failed_units[:10])
            + (f"\n  ... and {len(failed_units)-10} more" if len(failed_units) > 10 else "")
            + f"\n\n(next ntfy for the same set suppressed for "
              f"{FAILED_UNITS_NTFY_COOLDOWN_HOURS}h)\n"
              "`sudo systemctl status <unit>` to investigate, then "
              "`sudo systemctl reset-failed <unit>` once fixed."
        ),
    )
    return {}


# ---------------------------------------------------------------------------
# Node 2d: trade_engine_silence_check
# ---------------------------------------------------------------------------

# If premarket has not dispatched ANY candidate_entry in this many days,
# the trade engine is functionally offline. 5/22 → 6/1 incident: scout
# LLM degraded silently for 11 days, 0 dispatches, 0 alerts until
# operator asked. Threshold of 4 = "missed at least one trading week"
# (Mon-Fri = 5 trading days, so 4 catches a Mon→Thu silence).
TRADE_ENGINE_SILENCE_ALERT_DAYS = 4
# Cooldown so we don't ntfy every hour for the same persistent silence.
SILENCE_ALERT_COOLDOWN_HOURS = 12


def _was_silence_alerted_recently() -> bool:
    """True iff we've already alerted on trade_engine_silence in cooldown."""
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM agent_events
                WHERE event_type = 'trade_engine_silence_detected'
                  AND ts > NOW() - (%s::text || ' hours')::interval
                LIMIT 1
                """,
                (str(SILENCE_ALERT_COOLDOWN_HOURS),),
            )
            return cur.fetchone() is not None
    except Exception as e:
        log.warning("[silence_check] cooldown query failed: %s", e)
        return False


def trade_engine_silence_check(state: TradingGraphState) -> dict:
    """Alert if no candidate_entry has dispatched in the last N days.

    Failure mode this catches: scout LLM degrades silently → fallback
    score 0.5 → DISPATCH_MIN_SCORE 0.55 skips everything → trade engine
    functionally offline → operator never knows.

    Looks at agent_events for the most recent `candidate_entry_dispatched`
    event. If older than TRADE_ENGINE_SILENCE_ALERT_DAYS, emits sev=2
    + ntfy (rate-limited to one alert per SILENCE_ALERT_COOLDOWN_HOURS).
    """
    run_id = state["run_id"]
    trigger = state["trigger"]
    log.info("[health/trade_engine_silence_check] run_id=%s", run_id)

    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                """
                SELECT ts, EXTRACT(EPOCH FROM (NOW() - ts)) / 86400 AS days_ago
                FROM agent_events
                WHERE event_type = 'candidate_entry_dispatched'
                ORDER BY ts DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
    except Exception as e:
        log.warning("[silence_check] query failed: %s", e)
        return {}

    if row is None:
        days_ago = None
        last_ts_str = "never"
    else:
        last_ts, days_ago = row
        last_ts_str = last_ts.isoformat() if last_ts else "unknown"
        days_ago = float(days_ago) if days_ago is not None else None

    is_silent = days_ago is None or days_ago > TRADE_ENGINE_SILENCE_ALERT_DAYS
    if not is_silent:
        emit(
            run_id=run_id, trigger=trigger, agent="trade_engine_silence_check",
            event_type="trade_engine_active",
            payload={"last_dispatch": last_ts_str, "days_ago": days_ago},
        )
        return {}

    if _was_silence_alerted_recently():
        log.info("[silence_check] silence persists but ntfy in cooldown")
        emit(
            run_id=run_id, trigger=trigger, agent="trade_engine_silence_check",
            event_type="trade_engine_silence_detected",
            severity=SEV_ERROR,
            payload={
                "last_dispatch": last_ts_str,
                "days_ago": round(days_ago, 1) if days_ago else None,
                "threshold_days": TRADE_ENGINE_SILENCE_ALERT_DAYS,
                "ntfy_suppressed": True,
            },
        )
        return {}

    emit(
        run_id=run_id, trigger=trigger, agent="trade_engine_silence_check",
        event_type="trade_engine_silence_detected",
        severity=SEV_ERROR,
        payload={
            "last_dispatch": last_ts_str,
            "days_ago": round(days_ago, 1) if days_ago else None,
            "threshold_days": TRADE_ENGINE_SILENCE_ALERT_DAYS,
            "ntfy_suppressed": False,
        },
    )

    days_str = f"{days_ago:.1f}" if days_ago is not None else "∞ (never)"
    _alert_ops(
        title=f"Trade engine silent for {days_str} days",
        body=(
            f"No candidate_entry dispatch since {last_ts_str}.\n\n"
            f"Likely cause: scout LLM degraded → fallback score 0.5 → "
            f"all candidates SKIPPED by DISPATCH_MIN_SCORE 0.55.\n\n"
            f"Investigate:\n"
            f"  grep 'scout' /var/log/trading-agent-brain.err | tail\n"
            f"  curl localhost:8002/api/ops/summary\n\n"
            f"Next alert suppressed for "
            f"{SILENCE_ALERT_COOLDOWN_HOURS}h."
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
                -- Window is SYMMETRIC (±10 min), not forward-only. When the
                -- same ticker is dispatched twice in quick succession (e.g.
                -- a manual run + the scheduled 12:30 premarket both surface
                -- the same top pick), the SECOND dispatch's `systemctl start`
                -- no-ops because the unit is already running from the FIRST
                -- dispatch — so it emits no new run_start. Matching a
                -- run_start slightly BEFORE the dispatch recognizes that the
                -- ticker is already being processed and is NOT a silent die.
                -- (6/2 false-positive: IBM/CBOE/SNOW dispatched 12:27 + 12:30,
                -- ran fine, but the 12:30 dispatch alarmed.)
                AND s.started_at >= d.dispatched_at - interval '10 minutes'
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
