"""Weekly token budget governor for OAuth LLM channels.

Subscriptions have weekly limits we estimate from observed usage. We track
tokens and call counts in Postgres `agent_events` (event_type=llm_call) and
provide pre-flight checks. Defensive caps are conservative.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

log = logging.getLogger(__name__)

Channel = Literal["claude_code", "codex", "deepseek"]


# Conservative weekly token caps. Tune from Postgres usage stats over the
# first 30 days of operation.
WEEKLY_CAP: dict[Channel, int] = {
    "claude_code": 5_000_000,
    "codex": 800_000,
    "deepseek": 5_000_000,  # universal fallback — prepaid API, can absorb load
}

# Soft trigger fraction: degrade kicks in at this much of cap consumed.
SOFT_TRIGGER = 0.95


@dataclass
class UsageRow:
    week_start_iso: str
    channel: Channel
    role: str
    tokens: int
    calls: int


class WeeklyBudget:
    """Tracks LLM usage from agent_events and pre-flight checks new calls."""

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn  # None for unit-tests / stub mode

    def used_this_week(self, channel: Channel) -> int:
        """Sum of tokens used across all roles in the current ISO week."""
        if not self.dsn:
            return 0
        try:
            from trading_agent.store.postgres import cursor

            week_start = _week_start_utc()
            with cursor() as cur:
                cur.execute(
                    """
                    SELECT COALESCE(SUM((payload->>'tokens')::int), 0)
                    FROM agent_events
                    WHERE event_type = 'llm_call'
                      AND payload->>'channel' = %s
                      AND created_at >= %s
                    """,
                    (channel, week_start),
                )
                row = cur.fetchone()
                return int(row[0] if row else 0)
        except Exception as e:
            log.warning("budget query failed: %s", e)
            return 0

    def used_for_role(self, role: str) -> int:
        if not self.dsn:
            return 0
        try:
            from trading_agent.store.postgres import cursor

            week_start = _week_start_utc()
            with cursor() as cur:
                cur.execute(
                    """
                    SELECT COALESCE(SUM((payload->>'tokens')::int), 0)
                    FROM agent_events
                    WHERE event_type = 'llm_call'
                      AND payload->>'role' = %s
                      AND created_at >= %s
                    """,
                    (role, week_start),
                )
                row = cur.fetchone()
                return int(row[0] if row else 0)
        except Exception as e:
            log.warning("budget per-role query failed: %s", e)
            return 0

    def would_exceed_weekly(self, channel: Channel, projected_tokens: int) -> bool:
        used = self.used_this_week(channel)
        cap = WEEKLY_CAP[channel]
        return (used + projected_tokens) > cap * SOFT_TRIGGER

    def would_exceed_role(self, role: str, role_cap: int, projected_tokens: int) -> bool:
        used = self.used_for_role(role)
        return (used + projected_tokens) > role_cap * SOFT_TRIGGER

    def record(self, *, role: str, channel: Channel, tokens: int, cost_usd: float = 0.0) -> None:
        """Record a completed LLM call. Goes via agent_events.emit."""
        try:
            from trading_agent.events import emit

            emit(
                run_id="llm_router",
                trigger="llm_call",  # type: ignore[arg-type]
                agent="oauth_llm_router",
                event_type="llm_call",
                payload={
                    "role": role,
                    "channel": channel,
                    "tokens": tokens,
                    "cost_usd": cost_usd,
                },
            )
        except Exception as e:
            log.warning("budget record failed: %s", e)


def _week_start_utc() -> datetime:
    """ISO week start (Monday 00:00 UTC)."""
    now = datetime.now(timezone.utc)
    monday = now - timedelta(days=now.weekday())
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)
