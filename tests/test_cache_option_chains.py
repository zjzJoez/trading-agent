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


def _wire(monkeypatch, market, watchlist=None, journal_rows=None,
          shadow=None):
    monkeypatch.setattr(m, "_market_fns", lambda: market)
    monkeypatch.setattr(m, "_watchlist_underlyings", lambda: watchlist or [])
    monkeypatch.setattr(m, "_shadow_underlyings", lambda: shadow or [])
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


def _exp_rows(dtes):
    return [{"strike_time": (TODAY + timedelta(days=d)).isoformat()} for d in dtes]


def test_core_pick_guarantees_sleeve_entry_band_expiry():
    """Dense Mon/Wed/Fri weekly calendar: the 4 spread picks land near DTE
    {7, ~34, ~62, 90} only PROBABILISTICALLY inside 30-45; core names must
    be GUARANTEED one — the M1-0.4 replay gate counts days on it."""
    weekly_dtes = [d for d in range(1, 101)
                   if (TODAY + timedelta(days=d)).weekday() in (0, 2, 4)]
    rows = _exp_rows(weekly_dtes)
    picked = m._pick_expiries(rows, TODAY, core=True)
    dtes = [(date.fromisoformat(e) - TODAY).days for e in picked]
    assert any(30 <= d <= 45 for d in dtes)
    assert len(picked) <= m.CORE_MAX_EXPIRIES
    # Spread anchors survive the forcing.
    assert min(dtes) == min(d for d in weekly_dtes if d >= m.DTE_MIN)
    assert max(dtes) == max(d for d in weekly_dtes if d <= m.DTE_MAX)


def test_core_pick_pins_nearest_37_on_thin_calendar():
    """Thin calendar where the spread picks {7,21,55,90} all miss 30-45:
    core forcing adds the expiry nearest SLEEVE_DTE_TARGET; non-core
    keeps the plain spread pick."""
    dtes = [7, 14, 21, 28, 37, 55, 75, 90]
    rows = _exp_rows(dtes)
    exp_37 = (TODAY + timedelta(days=37)).isoformat()
    assert exp_37 not in m._pick_expiries(rows, TODAY)            # spread misses it
    picked = m._pick_expiries(rows, TODAY, core=True)
    assert exp_37 in picked
    assert len(picked) == 5


def test_held_expiry_forced_even_outside_dte_window():
    """An OPEN journal position's expiry is kept whenever the broker still
    lists it — including DTE < 7, where the nightly mark matters most —
    but an expiry the broker no longer lists is ignored."""
    dtes = [4, 7, 14, 21, 28, 37, 55, 75, 90]
    rows = _exp_rows(dtes)
    exp_4 = (TODAY + timedelta(days=4)).isoformat()
    exp_14 = (TODAY + timedelta(days=14)).isoformat()
    unlisted = (TODAY + timedelta(days=33)).isoformat()

    base = m._pick_expiries(rows, TODAY)
    assert exp_4 not in base and exp_14 not in base

    picked = m._pick_expiries(rows, TODAY, core=True, held={exp_4, exp_14, unlisted})
    assert exp_4 in picked and exp_14 in picked
    assert unlisted not in picked
    assert len(picked) <= m.CORE_MAX_EXPIRIES


def test_held_option_expiries_parsed_from_journal(monkeypatch):
    monkeypatch.setattr(
        "trading_agent.store.postgres.get_open_journal_trades",
        lambda: [{"symbol": "US.QQQ260814C600000"},
                 {"symbol": "US.QQQ260918P550000"},
                 {"symbol": "US.SPY"},          # stock — no expiry
                 {"symbol": "garbage"},
                 {}])
    assert m._held_option_expiries() == {
        "QQQ": {"2026-08-14", "2026-09-18"}}
    # Unreachable store (None) degrades to empty, never raises.
    monkeypatch.setattr(
        "trading_agent.store.postgres.get_open_journal_trades", lambda: None)
    assert m._held_option_expiries() == {}


def test_main_snapshots_held_and_sleeve_band_expiries(monkeypatch, pg, emits, frozen_today):
    """End-to-end: a held QQQ contract's expiry (DTE 14, not a spread pick)
    and the ~37 DTE sleeve-band expiry both land in the snapshot."""
    calls: dict = {}
    get_quote, _default_exp, chain = _fake_market(spot=100.0, calls=calls)
    dtes = [7, 14, 21, 28, 37, 55, 75, 90]

    def expiries_fn(underlying):
        return {"rows": [{"strike_time": (TODAY + timedelta(days=d)).isoformat()}
                         for d in dtes]}

    held_symbol = "US.QQQ" + (TODAY + timedelta(days=14)).strftime("%y%m%d") + "C100000"
    _wire(monkeypatch, (get_quote, expiries_fn, chain),
          journal_rows=[{"symbol": held_symbol}])

    assert m.main(["--underlyings", "QQQ"]) == 0

    inserts = [p for sql, p in pg.executes
               if "INSERT INTO option_chain_snapshots" in sql]
    snapped_expiries = {p[4] for p in inserts}
    # 4 spread picks (7,21,55,90) + forced 37 (sleeve band) + forced 14 (held).
    assert snapped_expiries == {TODAY + timedelta(days=d)
                                for d in (7, 14, 21, 37, 55, 90)}
    assert len(snapped_expiries) == m.CORE_MAX_EXPIRIES
    # Core strike window: 30 strikes/side inside ±15% x 2 sides = 60/expiry.
    assert len(inserts) == 6 * 60


# ---------------------------------------------------------------------------
# Underlying set
# ---------------------------------------------------------------------------

def test_underlying_set_unions_watchlist_and_open_journal(monkeypatch):
    monkeypatch.setattr(m, "_watchlist_underlyings", lambda: ["SPY", "AAPL"])
    monkeypatch.setattr(m, "_shadow_underlyings", lambda: [])
    monkeypatch.setattr(
        "trading_agent.store.postgres.get_open_journal_trades",
        lambda: [{"symbol": "US.QQQ260626C734000"}, {"symbol": "US.SPY"}],
    )
    # IWM arrives via CORE_UNDERLYINGS; SPY/QQQ dedupe against it.
    assert m.build_underlying_set() == ["SPY", "QQQ", "IWM", "AAPL"]


def test_underlying_set_includes_todays_shadow_book(monkeypatch):
    """A SHADOW_ONLY proposal's underlying can be in neither the watchlist
    nor the open-trade set (nothing was opened) — counterfactual replay
    still needs its chain priced tonight."""
    monkeypatch.setattr(m, "_watchlist_underlyings", lambda: ["SPY"])
    monkeypatch.setattr(m, "_shadow_underlyings", lambda: ["CRNX"])
    monkeypatch.setattr(
        "trading_agent.store.postgres.get_open_journal_trades", lambda: [])
    assert m.build_underlying_set() == ["SPY", "QQQ", "IWM", "CRNX"]


def test_shadow_underlyings_store_failure_is_empty(monkeypatch):
    """Best-effort: an unreachable shadow_proposals store loses one day of
    shadow coverage, it must not kill the cache run."""
    def _boom():
        raise ConnectionError("no postgres on this box")
    monkeypatch.setattr("trading_agent.store.postgres.cursor", _boom)
    assert m._shadow_underlyings() == []


def test_underlying_set_journal_unreadable_falls_back_to_watchlist(monkeypatch):
    """None from the store = PG unreachable; the cache run must still cover
    the watchlist instead of dying."""
    monkeypatch.setattr(m, "_watchlist_underlyings", lambda: ["SPY"])
    monkeypatch.setattr(m, "_shadow_underlyings", lambda: [])
    monkeypatch.setattr(
        "trading_agent.store.postgres.get_open_journal_trades", lambda: None)
    assert m.build_underlying_set() == ["SPY", "QQQ", "IWM"]


def test_underlyings_override_skips_sourcing(monkeypatch):
    def _boom(*a):
        raise AssertionError("must not source watchlist/journal when overridden")
    monkeypatch.setattr(m, "_watchlist_underlyings", _boom)
    monkeypatch.setattr(m, "_shadow_underlyings", _boom)
    monkeypatch.setattr(
        "trading_agent.store.postgres.get_open_journal_trades", _boom)
    assert m.build_underlying_set("qqq, aapl") == ["AAPL", "QQQ"]


# ---------------------------------------------------------------------------
# Core (sleeve 1) index set — always cached, wider strike window
# ---------------------------------------------------------------------------

def test_core_set_present_with_empty_watchlist(monkeypatch):
    """Sleeve 1 needs SPY/QQQ/IWM chain history daily even when the rotating
    watchlist, journal, and shadow book contain none of them."""
    monkeypatch.setattr(m, "_watchlist_underlyings", lambda: [])
    monkeypatch.setattr(m, "_shadow_underlyings", lambda: [])
    monkeypatch.setattr(
        "trading_agent.store.postgres.get_open_journal_trades", lambda: [])
    assert m.build_underlying_set() == ["SPY", "QQQ", "IWM"]


def test_core_window_constants_cover_sleeve():
    """The snapshot window must span the sleeve-1 trade band: expiries out to
    ~60 DTE (entries at 30-45 plus the roll tail) and strikes to ±15% of
    spot for the index names."""
    assert m.DTE_MIN <= 30 and m.DTE_MAX >= 60
    assert m.CORE_STRIKE_PCT >= 0.15
    assert set(m.CORE_UNDERLYINGS) == {"SPY", "QQQ", "IWM"}


def test_core_underlying_gets_pct_of_spot_strike_window(monkeypatch, pg, emits, frozen_today):
    """A core index name keeps every strike within ±15% of spot — well past
    the nearest-12 band a watchlist single name gets."""
    calls: dict = {}
    market = _fake_market(spot=100.0, n_strikes_each_side=40, calls=calls)
    _wire(monkeypatch, market)

    assert m.main(["--underlyings", "SPY"]) == 0

    option_batches = [b for b in calls["quote_batches"] if b != ["US.SPY"]]
    codes = [c for b in option_batches for c in b]
    strikes = sorted(_code_parts(c)[1] for c in codes)
    # $1 grid straddling 100 → 30 strikes per side inside ±15%, both sides.
    assert len(codes) == 60
    assert strikes[0] == 85.5 and strikes[-1] == 114.5
    assert all(abs(k - 100.0) <= 15.0 for k in strikes)
    # And the window genuinely widened past the nearest-12 default.
    assert any(abs(k - 100.0) > 6.5 for k in strikes)

    inserts = [p for sql, p in pg.executes
               if "INSERT INTO option_chain_snapshots" in sql]
    assert len(inserts) == 60


def test_core_quote_batches_respect_snapshot_cap(monkeypatch, pg, emits, frozen_today):
    """The widened core window can exceed moomoo's per-request snapshot cap;
    option quotes must go out in chunks of at most _QUOTE_BATCH_MAX."""
    calls: dict = {}
    market = _fake_market(spot=1000.0, n_strikes_each_side=400, calls=calls)
    _wire(monkeypatch, market)

    assert m.main(["--underlyings", "QQQ"]) == 0

    option_batches = [b for b in calls["quote_batches"] if b != ["US.QQQ"]]
    assert len(option_batches) > 1
    assert all(len(b) <= m._QUOTE_BATCH_MAX for b in option_batches)
    codes = [c for b in option_batches for c in b]
    # ±15% of 1000 on a $1 grid → 300 strikes per side, none lost to chunking.
    assert len(codes) == len(set(codes)) == 600
    inserts = [p for sql, p in pg.executes
               if "INSERT INTO option_chain_snapshots" in sql]
    assert len(inserts) == 600


def test_failed_quote_chunk_keeps_other_chunks(monkeypatch, pg, emits, frozen_today):
    """A transient OpenD hiccup on chunk 2 of 2 must not discard chunk 1:
    the fetched quotes are inserted, the expiry is counted partial, and the
    run escalates to warn instead of silently halving core coverage."""
    calls: dict = {}
    get_quote, expiries, chain = _fake_market(
        spot=1000.0, n_strikes_each_side=400, calls=calls)
    seen = {"option_batches": 0}

    def flaky_quote(symbols):
        symbols = list(symbols)
        if not symbols[0].split(".", 1)[-1].isalpha():   # option batch
            seen["option_batches"] += 1
            if seen["option_batches"] == 2:
                raise RuntimeError("OpenD hiccup on chunk 2")
        return get_quote(symbols)

    _wire(monkeypatch, (flaky_quote, expiries, chain))

    assert m.main(["--underlyings", "QQQ"]) == 0

    inserts = [p for sql, p in pg.executes
               if "INSERT INTO option_chain_snapshots" in sql]
    # ±15% of 1000 → 600 codes → two chunks of 300; chunk 2 died, chunk 1
    # survives instead of the whole expiry being dropped.
    assert len(inserts) == 300
    assert len(emits) == 1
    payload = emits[0]["payload"]
    assert payload["failures"] == {}                     # not a failed name
    assert payload["partial_expiries"] == {"QQQ": 1}
    assert payload["per_underlying"] == {"QQQ": 300}
    assert emits[0]["severity"] == 1                     # partial data = warn


def test_one_core_failure_does_not_skip_other_core(monkeypatch, pg, emits, frozen_today):
    """Sourced (no override) run: SPY's spot failing must still leave QQQ and
    IWM fully snapped — one core name down cannot blank the sleeve's data."""
    market = _fake_market(spot=100.0, fail_spot_for={"SPY"})
    _wire(monkeypatch, market)
    monkeypatch.setattr(
        "trading_agent.store.postgres.get_open_journal_trades", lambda: [])

    assert m.main([]) == 0

    inserts = [p for sql, p in pg.executes
               if "INSERT INTO option_chain_snapshots" in sql]
    # Default fake grid spans ±20 → 30 strikes/side inside ±15% x 2 sides = 60.
    assert {p[1] for p in inserts} == {"QQQ", "IWM"}
    assert len(emits) == 1
    payload = emits[0]["payload"]
    assert set(payload["failures"]) == {"SPY"}
    assert payload["per_underlying"] == {"IWM": 60, "QQQ": 60}
    assert emits[0]["severity"] == 1


# ---------------------------------------------------------------------------
# Failure isolation + all-failed escalation
# ---------------------------------------------------------------------------

def test_one_underlying_failing_does_not_kill_the_run(monkeypatch, pg, emits, frozen_today):
    market = _fake_market(spot=100.0, fail_spot_for={"AAPL"})
    _wire(monkeypatch, market)

    assert m.main(["--underlyings", "AAPL,MSFT"]) == 0

    inserts = [p for sql, p in pg.executes if "INSERT INTO option_chain_snapshots" in sql]
    assert len(inserts) == 24                      # MSFT snapped fine
    assert all(p[1] == "MSFT" for p in inserts)

    assert len(emits) == 1
    payload = emits[0]["payload"]
    assert "AAPL" in payload["failures"]
    assert payload["per_underlying"] == {"MSFT": 24}
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


# ---------------------------------------------------------------------------
# Core-first ordering + the core-empty tripwire (2026-07-25)
# ---------------------------------------------------------------------------

class TestCoreOrderingAndTripwire:
    """The 7/22-24 runs wrote ~1000 rows/night across 85 names while SPY alone
    yields ~1100: the broker's quota drains mid-run and later names return
    EMPTY without raising. Alphabetically SPY/QQQ/IWM were last, so the
    sleeve's feed got the leftovers (QQQ: 1 day in 3 weeks)."""

    def test_core_underlyings_come_first(self, monkeypatch):
        from trading_agent.jobs import cache_option_chains as job
        monkeypatch.setattr(job, "_watchlist_underlyings",
                            lambda: ["AAPL", "AAON", "ZM", "SPY"])
        monkeypatch.setattr(job, "_open_trade_underlyings", lambda: ["IBM"])
        monkeypatch.setattr(job, "_shadow_underlyings", lambda: ["NVDA"])
        out = job.build_underlying_set()
        assert out[:3] == list(job.CORE_UNDERLYINGS)
        # tail stays deduped + sorted, and never repeats a core name
        assert out[3:] == sorted({"AAPL", "AAON", "ZM", "IBM", "NVDA"})
        assert out.count("SPY") == 1

    def test_explicit_override_is_still_a_pure_override(self, monkeypatch):
        from trading_agent.jobs import cache_option_chains as job
        monkeypatch.setattr(job, "_watchlist_underlyings", lambda: ["AAPL"])
        monkeypatch.setattr(job, "_open_trade_underlyings", lambda: [])
        monkeypatch.setattr(job, "_shadow_underlyings", lambda: [])
        assert job.build_underlying_set("nvda,aapl") == ["AAPL", "NVDA"]

    def _run_main(self, monkeypatch, per_underlying_rows):
        """Drive main() with a stubbed snapshot_underlying."""
        from trading_agent.jobs import cache_option_chains as job
        monkeypatch.setattr(job, "_watchlist_underlyings", lambda: ["AAPL"])
        monkeypatch.setattr(job, "_open_trade_underlyings", lambda: [])
        monkeypatch.setattr(job, "_shadow_underlyings", lambda: [])
        monkeypatch.setattr(job, "_held_option_expiries", lambda: {})
        monkeypatch.setattr(job, "_insert_rows", lambda rows: None)
        monkeypatch.setattr(
            job, "snapshot_underlying",
            lambda u, today, held: ([("row",)] * per_underlying_rows.get(u, 0), 0))
        emitted = []
        monkeypatch.setattr(job.events, "emit",
                            lambda **kw: emitted.append(kw))
        pushes = []
        monkeypatch.setattr("trading_agent.notify.send",
                            lambda *a, **kw: pushes.append((a, kw)))
        rc = job.main([])
        return rc, emitted, pushes

    def test_core_empty_is_sev_error_and_pushes(self, monkeypatch):
        from trading_agent import events as ev
        rc, emitted, pushes = self._run_main(
            monkeypatch, {"SPY": 100, "IWM": 100, "AAPL": 50})  # QQQ -> 0
        payload = emitted[0]["payload"]
        assert payload["core_empty"] == ["QQQ"]
        assert emitted[0]["severity"] == ev.SEV_ERROR
        assert len(pushes) == 1 and "QQQ" in pushes[0][1]["title"]
        assert rc == 0  # a hole is an alert, not a job failure

    def test_all_core_present_is_quiet(self, monkeypatch):
        from trading_agent import events as ev
        rc, emitted, pushes = self._run_main(
            monkeypatch, {"SPY": 100, "QQQ": 100, "IWM": 100, "AAPL": 0})
        payload = emitted[0]["payload"]
        assert payload["core_empty"] == []
        assert emitted[0]["severity"] == ev.SEV_INFO
        assert pushes == []
