"""Market-hours gate tests."""
import sys
from datetime import date, datetime, timezone

import pytest

from trading_agent.market_calendar import (
    is_us_market_open,
    is_us_trading_day,
    minutes_since_open,
    trading_days_between,
)


@pytest.fixture
def mcal_broken(monkeypatch):
    """Force the us_session_bounds() except branch: poisoning the sys.modules
    entry makes ``import pandas_market_calendars`` raise ImportError, so the
    ET-local fallback window is what gets exercised."""
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)


def test_fallback_weekend_closed(mcal_broken):
    # 2026-06-06 is a Saturday
    assert is_us_market_open(datetime(2026, 6, 6, 15, 0, tzinfo=timezone.utc)) is False


def test_fallback_weekday_regular_hours_open(mcal_broken):
    # Tue 2026-06-02 15:00 UTC = 11:00 EDT, mid-session
    assert is_us_market_open(datetime(2026, 6, 2, 15, 0, tzinfo=timezone.utc)) is True


def test_fallback_premarket_closed(mcal_broken):
    # Tue 2026-06-02 12:30 UTC = 08:30 EDT, 1h before open
    assert is_us_market_open(datetime(2026, 6, 2, 12, 30, tzinfo=timezone.utc)) is False


def test_fallback_after_hours_closed(mcal_broken):
    # Tue 2026-06-02 20:30 UTC = 16:30 EDT, after the close
    assert is_us_market_open(datetime(2026, 6, 2, 20, 30, tzinfo=timezone.utc)) is False


def test_fallback_winter_1330_utc_is_premarket_closed(mcal_broken):
    """EST regression: 13:30 UTC on a winter trading day is 08:30 ET — the
    exact instant the winter premarket digest fires. The old fixed
    13:30-20:00 UTC fallback read OPEN here; the ET-local fallback must
    read CLOSED so a calendar failure can't let the digest scan dispatch
    on pre-market quotes."""
    # Tue 2026-01-13 (EST)
    assert is_us_market_open(datetime(2026, 1, 13, 13, 30, tzinfo=timezone.utc)) is False


def test_fallback_winter_1430_utc_is_open(mcal_broken):
    # Tue 2026-01-13 14:30 UTC = 09:30 EST, the open (inclusive)
    assert is_us_market_open(datetime(2026, 1, 13, 14, 30, tzinfo=timezone.utc)) is True
    # 20:30 UTC = 15:30 EST, still in session (old fallback also OPEN here)
    assert is_us_market_open(datetime(2026, 1, 13, 20, 30, tzinfo=timezone.utc)) is True


def test_fallback_winter_after_close_closed(mcal_broken):
    # Tue 2026-01-13 21:30 UTC = 16:30 EST, after the close
    assert is_us_market_open(datetime(2026, 1, 13, 21, 30, tzinfo=timezone.utc)) is False


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
