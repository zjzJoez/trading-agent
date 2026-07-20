"""Market-hours gate tests."""
from datetime import date, datetime, timezone

from trading_agent.market_calendar import (
    _fallback_is_open,
    is_us_market_open,
    is_us_trading_day,
    minutes_since_open,
    trading_days_between,
)


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


def test_minutes_since_open_mid_session():
    # Tue 2026-06-02 14:30 UTC = 10:30 ET, 60 min after the 09:30 ET open
    res = minutes_since_open(datetime(2026, 6, 2, 14, 30, tzinfo=timezone.utc))
    assert res is not None
    mins, session = res
    assert abs(mins - 60.0) < 1e-6
    assert abs(session - 390.0) < 1e-6


def test_minutes_since_open_weekend_is_none():
    assert minutes_since_open(datetime(2026, 6, 6, 15, 0, tzinfo=timezone.utc)) is None


def test_minutes_since_open_negative_pre_open():
    # Tue 2026-06-02 13:00 UTC = 09:00 ET, 30 min before the open
    res = minutes_since_open(datetime(2026, 6, 2, 13, 0, tzinfo=timezone.utc))
    assert res is not None
    assert res[0] < 0


def test_is_us_trading_day():
    assert is_us_trading_day(date(2026, 6, 2)) is True     # Tuesday
    assert is_us_trading_day(date(2026, 6, 6)) is False    # Saturday
    assert is_us_trading_day(date(2026, 7, 3)) is False    # July 4th observed (Fri)


def test_trading_days_between_consecutive_days():
    # Tue 6/9 → Wed 6/10 = 1 trading day
    assert trading_days_between(
        datetime(2026, 6, 9, 15, 0, tzinfo=timezone.utc),
        datetime(2026, 6, 10, 18, 0, tzinfo=timezone.utc),
    ) == 1


def test_trading_days_between_spans_weekend():
    # Fri 6/5 → Mon 6/8: the weekend contributes nothing
    assert trading_days_between(
        datetime(2026, 6, 5, 15, 0, tzinfo=timezone.utc),
        datetime(2026, 6, 8, 15, 0, tzinfo=timezone.utc),
    ) == 1


def test_trading_days_between_same_day_is_zero():
    assert trading_days_between(
        datetime(2026, 6, 10, 14, 0, tzinfo=timezone.utc),
        datetime(2026, 6, 10, 19, 0, tzinfo=timezone.utc),
    ) == 0


def test_trading_days_between_four_sessions():
    # Thu 6/4 → Wed 6/10: Fri 6/5, Mon 6/8, Tue 6/9, Wed 6/10
    assert trading_days_between(
        datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
        datetime(2026, 6, 10, 18, 0, tzinfo=timezone.utc),
    ) == 4
