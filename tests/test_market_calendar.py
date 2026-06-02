"""Market-hours gate tests."""
from datetime import datetime, timezone

from trading_agent.market_calendar import _fallback_is_open, is_us_market_open


def test_fallback_weekend_closed():
    # 2026-06-06 is a Saturday
    assert _fallback_is_open(datetime(2026, 6, 6, 15, 0, tzinfo=timezone.utc)) is False


def test_fallback_weekday_regular_hours_open():
    # Tue 2026-06-02 15:00 UTC = mid-session
    assert _fallback_is_open(datetime(2026, 6, 2, 15, 0, tzinfo=timezone.utc)) is True


def test_fallback_premarket_closed():
    # Tue 2026-06-02 12:30 UTC = 1h before open
    assert _fallback_is_open(datetime(2026, 6, 2, 12, 30, tzinfo=timezone.utc)) is False


def test_fallback_after_hours_closed():
    # Tue 2026-06-02 20:30 UTC = after 20:00 close
    assert _fallback_is_open(datetime(2026, 6, 2, 20, 30, tzinfo=timezone.utc)) is False


def test_is_open_never_raises():
    # Whatever the calendar lib does, the public fn must return a bool.
    assert isinstance(is_us_market_open(datetime(2026, 6, 6, 15, 0, tzinfo=timezone.utc)), bool)
