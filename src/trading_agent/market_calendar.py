"""US equity/option market-hours check.

Used to gate order DISPATCH and EXECUTION to regular trading hours. The
6/2 SNOW incident: the pipeline ran at 12:30 UTC (1h before the 13:30
open) and bought the pre-market spike, which faded at the open. Trading
only in regular hours against live quotes is the fix.

Primary: pandas_market_calendars (a project dep) — handles DST + holidays
+ early closes. Fallback: a plain 13:30-20:00 UTC weekday window (correct
during US EDT, ~1h off during EST but never lets a weekend/clearly-closed
session through).
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


def _fallback_is_open(now_utc: datetime) -> bool:
    """Plain UTC window: Mon-Fri 13:30-20:00 UTC (US regular hours in EDT)."""
    if now_utc.weekday() >= 5:  # Sat/Sun
        return False
    t = now_utc.timetz().replace(tzinfo=None)
    return time(13, 30) <= t <= time(20, 0)


def _coerce_utc(dt: datetime | None) -> datetime:
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def is_us_market_open(now_utc: datetime | None = None) -> bool:
    """True iff the US equity market is in its regular cash session now.

    Best-effort: any calendar error degrades to the UTC-window fallback.
    Never raises.
    """
    now_utc = _coerce_utc(now_utc)
    bounds = us_session_bounds(now_utc)
    if bounds is None:
        return False
    market_open, market_close = bounds
    return market_open <= now_utc <= market_close


def us_session_bounds(now_utc: datetime | None = None) -> tuple[datetime, datetime] | None:
    """(open_utc, close_utc) of today's US regular cash session, else None.

    "Today" is the America/New_York calendar date of ``now_utc`` (a UTC
    evening after an ET midnight belongs to the NEXT session). Half-days
    come back with their real early close. Best-effort: any calendar
    error degrades to the plain 13:30-20:00 UTC weekday window. Never
    raises.
    """
    now_utc = _coerce_utc(now_utc)
    day = now_utc.astimezone(ET).date()
    try:
        import pandas_market_calendars as mcal

        cal = mcal.get_calendar("XNYS")  # NYSE regular hours
        sched = cal.schedule(start_date=day.isoformat(), end_date=day.isoformat())
        if sched.empty:
            return None  # holiday / weekend
        market_open = sched.iloc[0]["market_open"].to_pydatetime()
        market_close = sched.iloc[0]["market_close"].to_pydatetime()
        return market_open, market_close
    except Exception as e:
        log.warning("[market_calendar] mcal schedule failed (%s) — UTC fallback", e)
        if day.weekday() >= 5:
            return None
        return (
            datetime.combine(day, time(13, 30), tzinfo=timezone.utc),
            datetime.combine(day, time(20, 0), tzinfo=timezone.utc),
        )


def minutes_since_open(now_utc: datetime | None = None) -> tuple[float, float] | None:
    """(minutes since today's open, session length in minutes) or None.

    None when there is no US session today (weekend / holiday). The first
    element is negative pre-open and exceeds the session length after the
    close — callers decide how to clamp. Session length is 390 on a full
    day, less on half-days. Never raises.
    """
    now_utc = _coerce_utc(now_utc)
    bounds = us_session_bounds(now_utc)
    if bounds is None:
        return None
    market_open, market_close = bounds
    return (
        (now_utc - market_open).total_seconds() / 60.0,
        (market_close - market_open).total_seconds() / 60.0,
    )


def is_us_trading_day(day: date | None = None) -> bool:
    """True iff ``day`` (an America/New_York calendar date; default today)
    has a US regular session. Falls back to a plain weekday check on any
    calendar error. Never raises."""
    if day is None:
        day = datetime.now(timezone.utc).astimezone(ET).date()
    try:
        import pandas_market_calendars as mcal

        cal = mcal.get_calendar("XNYS")
        return not cal.schedule(start_date=day.isoformat(), end_date=day.isoformat()).empty
    except Exception as e:
        log.warning("[market_calendar] mcal trading-day check failed (%s) — weekday fallback", e)
        return day.weekday() < 5


def trading_days_between(start_utc: datetime, end_utc: datetime | None = None) -> int:
    """Count of US trading sessions with an ET date in (start, end].

    Used for trading-day cooldowns: an event on Tuesday queried on
    Wednesday (both trading days) is 1 trading day old; queried the same
    day it is 0. Falls back to counting weekdays on any calendar error.
    Never raises.
    """
    start_utc = _coerce_utc(start_utc)
    end_utc = _coerce_utc(end_utc)
    start_day = start_utc.astimezone(ET).date()
    end_day = end_utc.astimezone(ET).date()
    if end_day <= start_day:
        return 0
    try:
        import pandas_market_calendars as mcal

        cal = mcal.get_calendar("XNYS")
        sched = cal.schedule(
            start_date=(start_day + timedelta(days=1)).isoformat(),
            end_date=end_day.isoformat(),
        )
        return len(sched)
    except Exception as e:
        log.warning("[market_calendar] mcal day count failed (%s) — weekday fallback", e)
        n = 0
        d = start_day + timedelta(days=1)
        while d <= end_day:
            if d.weekday() < 5:
                n += 1
            d += timedelta(days=1)
        return n


__all__ = [
    "is_us_market_open",
    "us_session_bounds",
    "minutes_since_open",
    "is_us_trading_day",
    "trading_days_between",
]
