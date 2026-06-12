"""Tests for the daily EOD option-chain snapshot job.

All broker access goes through cache_option_chains._market_fns (lazy moomoo
import) and all Postgres writes through trading_agent.store.postgres.cursor —
both are mocked here; nothing touches OpenD or a real database.
"""
from __future__ import annotations

import json
import math
import re
from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest

from trading_agent.jobs import cache_option_chains as m

TODAY = date(2026, 6, 12)
EXP_NEAR = (TODAY + timedelta(days=21)).isoformat()   # in 7-90 window

# <UNDERLYING><YYMMDD><C|P><strike*1000, unpadded> — same shape the broker uses.
_CP_STRIKE = re.compile(r"([CP])(\d+)$")


def _code_parts(code: str) -> tuple[str, float]:
    mm = _CP_STRIKE.search(code.split(".", 1)[-1])
    assert mm, f"unparseable option code in test fake: {code}"
    return ("CALL" if mm.group(1) == "C" else "PUT"), int(mm.group(2)) / 1000.0


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _fake_market(spot: float = 100.0, n_strikes_each_side: int = 20,
                 fail_spot_for: set[str] | None = None,
                 calls: dict | None = None):
    """Return (get_quote, list_option_expiries, get_option_chain) fakes that
    mimic the moomoo server's field names (last_price, strike_time,
    strike_price/option_type on chain rows, option_* greeks on quote rows)."""
    fail_spot_for = fail_spot_for or set()
    calls = calls if calls is not None else {}
    calls.setdefault("quote_batches", [])

    def get_quote(symbols):
        calls["quote_batches"].append(list(symbols))
        rows = []
        for s in symbols:
            bare = s.split(".", 1)[-1]
            if bare.isalpha():  # underlying spot quote
                if bare in fail_spot_for:
                    raise RuntimeError(f"quote failed for {s}")
                rows.append({"code": s, "last_price": spot})
            else:  # option code quote
                opt_type, strike = _code_parts(s)
                rows.append({
                    "code": s,
                    "option_type": opt_type,
                    "option_strike_price": strike,
                    "bid_price": 1.10, "ask_price": 1.30, "last_price": 1.20,
                    "volume": 250, "option_open_interest": 1234,
                    "option_implied_volatility": 28.5,   # broker reports PERCENT
                    "option_delta": 0.42, "option_gamma": 0.03,
                    "option_vega": 0.11, "option_theta": -0.05,
                    "strike_time": EXP_NEAR,
                })
        return {"rows": rows}

    def list_option_expiries(underlying):
        return {"rows": [{"strike_time": EXP_NEAR,
                          "option_expiry_date_distance": 21}]}

    def get_option_chain(underlying, expiry, option_type="ALL"):
        bare = underlying.split(".", 1)[-1]
        ymd = expiry.replace("-", "")[2:]
        rows = []
        for i in range(n_strikes_each_side * 2):
            # strikes straddling spot, $1 apart
            k = spot - n_strikes_each_side + i + 0.5
            for cp in ("C", "P"):
                rows.append({
                    "code": f"US.{bare}{ymd}{cp}{int(k * 1000)}",
                    "strike_price": k,
                    "option_type": "CALL" if cp == "C" else "PUT",
                })
        return {"rows": rows}

    return get_quote, list_option_expiries, get_option_chain


class _PgCapture:
    """Context-manager factory standing in for store.postgres.cursor():
    records every (sql, params) execute."""

    def __init__(self):
        self.executes: list[tuple[str, tuple]] = []

    def __call__(self):
        cur = MagicMock()
        cur.execute.side_effect = lambda sql, params=None: self.executes.append(
            (sql, params))
        ctx = MagicMock()
        ctx.__enter__.return_value = cur
        ctx.__exit__.return_value = False
        return ctx


@pytest.fixture()
def pg(monkeypatch):
    cap = _PgCapture()
    monkeypatch.setattr("trading_agent.store.postgres.cursor", cap)
    return cap


@pytest.fixture()
def emits(monkeypatch):
    out: list[dict] = []
    monkeypatch.setattr("trading_agent.events.emit",
                        lambda **kw: out.append(kw) or 1)
    return out


@pytest.fixture()
def frozen_today(monkeypatch):
    """Pin the snapshot date so EXP_NEAR's DTE is deterministic."""
    class _DT:
        @staticmethod
        def now(tz=None):
            from datetime import datetime, timezone
            return datetime(2026, 6, 12, 21, 15, tzinfo=timezone.utc)
    monkeypatch.setattr(m, "datetime", _DT)
    return TODAY


def _wire(monkeypatch, market, watchlist=None, journal_rows=None):
    monkeypatch.setattr(m, "_market_fns", lambda: market)
    monkeypatch.setattr(m, "_watchlist_underlyings", lambda: watchlist or [])
    monkeypatch.setattr("trading_agent.store.postgres.get_open_journal_trades",
                        lambda: journal_rows)


# ---------------------------------------------------------------------------
# Row construction + ON CONFLICT
# ---------------------------------------------------------------------------

def test_rows_inserted_with_on_conflict_upsert(monkeypatch, pg, emits, frozen_today):
    market = _fake_market(spot=100.0)
    _wire(monkeypatch, market)

    assert m.main(["--underlyings", "AAPL"]) == 0

    inserts = [(sql, p) for sql, p in pg.executes if "INSERT INTO option_chain_snapshots" in sql]
    # 12 strikes each side x 2 sides, single expiry.
    assert len(inserts) == 24
    sql = inserts[0][0]
    # Re-run idempotency lives in the SQL: upsert on the snapshot-day key.
    assert "ON CONFLICT (snapshot_date, code) DO UPDATE" in sql
    assert "captured_at = NOW()" in sql

    (snap_date, underlying, spot, code, expiry, strike, opt_type,
     bid, ask, last, volume, oi, iv, delta, gamma, vega, theta) = inserts[0][1]
    assert snap_date == TODAY
    assert underlying == "AAPL"
    assert spot == 100.0
    assert code.startswith("US.AAPL") and ("C" in code or "P" in code)
    assert expiry == date.fromisoformat(EXP_NEAR)
    assert opt_type in ("CALL", "PUT")
    assert (bid, ask, last) == (1.10, 1.30, 1.20)
    assert (volume, oi) == (250, 1234)
    # Broker percent -> stored fraction.
    assert iv == pytest.approx(0.285)
    assert (delta, gamma, vega, theta) == (0.42, 0.03, 0.11, -0.05)
    assert strike is not None and abs(strike - 100.0) <= m.STRIKES_EACH_SIDE

    # Exactly one audit row, info severity, with totals.
    assert len(emits) == 1
    assert emits[0]["severity"] == 0
    assert emits[0]["payload"]["rows"] == 24
    assert emits[0]["payload"]["failures"] == {}


def test_strike_selection_takes_nearest_each_side(monkeypatch, pg, emits, frozen_today):
    """Of 40 strikes per side only the 12 nearest spot (per side) are quoted."""
    calls: dict = {}
    market = _fake_market(spot=100.0, n_strikes_each_side=40, calls=calls)
    _wire(monkeypatch, market)

    assert m.main(["--underlyings", "AAPL"]) == 0

    option_batches = [b for b in calls["quote_batches"] if b != ["US.AAPL"]]
    assert len(option_batches) == 1
    codes = option_batches[0]
    assert len(codes) == 2 * m.STRIKES_EACH_SIDE
    strikes = sorted(_code_parts(c)[1] for c in codes)
    # All selected strikes hug spot: within 6 dollars (12 strikes, $1 grid,
    # straddling 100) — the $60-away wings must not appear.
    assert all(abs(k - 100.0) < 6.5 for k in strikes)


def test_nan_greeks_become_none_not_nan(monkeypatch, pg, emits, frozen_today):
    """Moomoo pads missing greeks with NaN; NaN must not reach Postgres."""
    get_quote, expiries, chain = _fake_market(spot=100.0)

    def quote_with_nan(symbols):
        out = get_quote(symbols)
        for r in out["rows"]:
            if "option_delta" in r:
                r["option_delta"] = float("nan")
                r["option_implied_volatility"] = float("nan")
        return out

    _wire(monkeypatch, (quote_with_nan, expiries, chain))
    assert m.main(["--underlyings", "AAPL"]) == 0

    inserts = [p for sql, p in pg.executes if "INSERT INTO option_chain_snapshots" in sql]
    assert inserts
    for params in inserts:
        iv, delta = params[12], params[13]
        assert iv is None and delta is None
        assert not any(isinstance(v, float) and math.isnan(v) for v in params)


# ---------------------------------------------------------------------------
# Expiry window
# ---------------------------------------------------------------------------

def test_expiry_dte_filter_and_cap(frozen_today):
    """Only 7-90 DTE expiries survive; more than 4 get thinned but the
    nearest and farthest in-window expiries are always kept."""
    dtes = [1, 7, 14, 30, 45, 60, 90, 120]
    rows = [{"strike_time": (TODAY + timedelta(days=d)).isoformat()} for d in dtes]
    picked = m._pick_expiries(rows, TODAY)
    assert len(picked) == m.MAX_EXPIRIES_PER_UNDERLYING
    assert (TODAY + timedelta(days=7)).isoformat() in picked     # nearest in window
    assert (TODAY + timedelta(days=90)).isoformat() in picked    # farthest in window
    assert (TODAY + timedelta(days=1)).isoformat() not in picked
    assert (TODAY + timedelta(days=120)).isoformat() not in picked


# ---------------------------------------------------------------------------
# Underlying set
# ---------------------------------------------------------------------------

def test_underlying_set_unions_watchlist_and_open_journal(monkeypatch):
    monkeypatch.setattr(m, "_watchlist_underlyings", lambda: ["SPY", "AAPL"])
    monkeypatch.setattr(
        "trading_agent.store.postgres.get_open_journal_trades",
        lambda: [{"symbol": "US.QQQ260626C734000"}, {"symbol": "US.SPY"}],
    )
    assert m.build_underlying_set() == ["AAPL", "QQQ", "SPY"]


def test_underlying_set_journal_unreadable_falls_back_to_watchlist(monkeypatch):
    """None from the store = PG unreachable; the cache run must still cover
    the watchlist instead of dying."""
    monkeypatch.setattr(m, "_watchlist_underlyings", lambda: ["SPY"])
    monkeypatch.setattr(
        "trading_agent.store.postgres.get_open_journal_trades", lambda: None)
    assert m.build_underlying_set() == ["SPY"]


def test_underlyings_override_skips_sourcing(monkeypatch):
    def _boom(*a):
        raise AssertionError("must not source watchlist/journal when overridden")
    monkeypatch.setattr(m, "_watchlist_underlyings", _boom)
    monkeypatch.setattr(
        "trading_agent.store.postgres.get_open_journal_trades", _boom)
    assert m.build_underlying_set("qqq, aapl") == ["AAPL", "QQQ"]


# ---------------------------------------------------------------------------
# Failure isolation + all-failed escalation
# ---------------------------------------------------------------------------

def test_one_underlying_failing_does_not_kill_the_run(monkeypatch, pg, emits, frozen_today):
    market = _fake_market(spot=100.0, fail_spot_for={"AAPL"})
    _wire(monkeypatch, market)

    assert m.main(["--underlyings", "AAPL,QQQ"]) == 0

    inserts = [p for sql, p in pg.executes if "INSERT INTO option_chain_snapshots" in sql]
    assert len(inserts) == 24                      # QQQ snapped fine
    assert all(p[1] == "QQQ" for p in inserts)

    assert len(emits) == 1
    payload = emits[0]["payload"]
    assert "AAPL" in payload["failures"]
    assert payload["per_underlying"] == {"QQQ": 24}
    assert emits[0]["severity"] == 1               # partial failure = warn


def test_every_underlying_failing_emits_sev2_and_exits_nonzero(monkeypatch, pg, emits, frozen_today):
    """All names failing means OpenD is down — sev-2 so the operator looks."""
    market = _fake_market(fail_spot_for={"AAPL", "QQQ"})
    _wire(monkeypatch, market)

    assert m.main(["--underlyings", "AAPL,QQQ"]) == 1

    assert not [p for sql, p in pg.executes if "option_chain_snapshots" in sql]
    assert len(emits) == 1
    assert emits[0]["severity"] == 2
    assert emits[0]["payload"]["all_failed"] is True
    assert set(emits[0]["payload"]["failures"]) == {"AAPL", "QQQ"}


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def test_dry_run_fetches_but_writes_nothing(monkeypatch, emits, frozen_today, capsys):
    market = _fake_market(spot=100.0)
    _wire(monkeypatch, market)

    def _no_db():
        raise AssertionError("dry-run must not open a Postgres cursor")
    monkeypatch.setattr("trading_agent.store.postgres.cursor", _no_db)

    assert m.main(["--dry-run", "--underlyings", "AAPL"]) == 0

    assert emits == []                             # emit() is also a DB write
    summary = json.loads(capsys.readouterr().out)
    assert summary["dry_run"] is True
    assert summary["rows"] == 24
    assert summary["per_underlying"] == {"AAPL": 24}
