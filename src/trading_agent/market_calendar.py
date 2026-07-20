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
from datetime import datetime, time, timedelta, timezone

log = logging.getLogger(__name__)


def _fallback_is_open(now_utc: datetime) -> bool:
    """Plain UTC window: Mon-Fri 13:30-20:00 UTC (US regular hours in EDT)."""
    if now_utc.weekday() >= 5:  # Sat/Sun
        return False
    t = now_utc.timetz().replace(tzinfo=None)
    return time(13, 30) <= t <= time(20, 0)


def is_us_market_open(now_utc: datetime | None = None) -> bool:
    """True iff the US equity market is in its regular cash session now.

    Best-effort: any calendar error degrades to the UTC-window fallback.
    Never raises.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    try:
        import pandas as pd
        import pandas_market_calendars as mcal

        cal = mcal.get_calendar("XNYS")  # NYSE regular hours
        day = now_utc.date()
        sched = cal.schedule(start_date=day.isoformat(), end_date=day.isoformat())
        if sched.empty:
            return False  # holiday / weekend
        market_open = sched.iloc[0]["market_open"].to_pydatetime()
        market_close = sched.iloc[0]["market_close"].to_pydatetime()
        ts = pd.Timestamp(now_utc).to_pydatetime()
        return market_open <= ts <= market_close
    except Exception as e:
        log.warning("[market_calendar] mcal check failed (%s) — UTC fallback", e)
        return _fallback_is_open(now_utc)


def _overlap_hours(lo_a: datetime, hi_a: datetime,
                   lo_b: datetime, hi_b: datetime) -> float:
    lo = max(lo_a, lo_b)
    hi = min(hi_a, hi_b)
    if hi <= lo:
        return 0.0
    return (hi - lo).total_seconds() / 3600.0


def _fallback_hours_between(start_utc: datetime, end_utc: datetime) -> float:
    """Plain UTC window sum: Mon-Fri 13:30-20:00 UTC sessions in [start, end]."""
    total = 0.0
    day = start_utc.date()
    while day <= end_utc.date():
        if day.weekday() < 5:
            sess_open = datetime.combine(day, time(13, 30), tzinfo=timezone.utc)
            sess_close = datetime.combine(day, time(20, 0), tzinfo=timezone.utc)
            total += _overlap_hours(start_utc, end_utc, sess_open, sess_close)
        day += timedelta(days=1)
    return total


def market_hours_between(start_utc: datetime, end_utc: datetime) -> float:
    """Cumulative REGULAR-SESSION hours between two instants.

    Overnight, weekend, and holiday time contributes zero — so "24
    market-hours of silence" means roughly 3.7 trading days, and a long
    weekend can never trip a market-hours threshold on its own. Used by
    the deadman escalation in health_nodes (the 6/20→7/8 OAuth-expiry
    incident measured its silence in exactly these units).

    Best-effort: any calendar error degrades to the UTC-window fallback.
    Never raises. Naive inputs are interpreted as UTC.
    """
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=timezone.utc)
    if end_utc.tzinfo is None:
        end_utc = end_utc.replace(tzinfo=timezone.utc)
    if end_utc <= start_utc:
        return 0.0
    try:
        import pandas_market_calendars as mcal

        cal = mcal.get_calendar("XNYS")  # NYSE regular hours
        sched = cal.schedule(
            start_date=start_utc.date().isoformat(),
            end_date=end_utc.date().isoformat(),
        )
        total = 0.0
        for _, row in sched.iterrows():
            sess_open = row["market_open"].to_pydatetime()
            sess_close = row["market_close"].to_pydatetime()
            total += _overlap_hours(start_utc, end_utc, sess_open, sess_close)
        return total
    except Exception as e:
        log.warning("[market_calendar] mcal hours-between failed (%s) — "
                    "UTC fallback", e)
        return _fallback_hours_between(start_utc, end_utc)


__all__ = ["is_us_market_open", "market_hours_between"]
