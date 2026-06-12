"""Exit-fill confirmation state machine (Phase 0.3).

Pins the core measurement-integrity property: a journal close happens ONLY
at a broker-confirmed dealt price, net of fees — never at placement time.
"""
from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_journal_db(tmp_path: Path, monkeypatch):
    from trading_agent import config as config_mod
    from trading_agent import db as db_mod

    db_file = tmp_path / "trader_test.db"
    new_cfg = dataclasses.replace(
        config_mod.CONFIG, db_path=db_file, data_dir=tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(db_mod, "CONFIG", new_cfg)
    db_mod.migrate(db_file)
    yield db_file


@pytest.fixture(autouse=True)
def _no_postgres(monkeypatch):
    """fill_confirm tries Postgres first in _iter_pending — force its lazy
    imports to fail so tests exercise the SQLite store deterministically.

    trading_agent.events imports store.postgres at module top, so import it
    (and the store) FIRST; only fresh imports are blocked afterwards. emit()
    itself never raises — it degrades to stderr when Postgres is down.
    """
    import sys

    import trading_agent.events  # noqa: F401 — resolve before blocking
    monkeypatch.setitem(sys.modules, "trading_agent.store.postgres", None)


def _open_trade(symbol="US.AAPL260626C325000", qty=2, entry=16.76,
                fees=2.0) -> int:
    from trading_agent.db import connection
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO trades (symbol, asset_type, side, qty, entry_price, "
            "opened_at, outcome, broker_order_id, is_paper, provenance, fees) "
            "VALUES (?, 'OPT', 'BUY', ?, ?, ?, 'OPEN', '9001', 1, 'agent', ?)",
            (symbol, qty, entry, datetime.now(UTC).isoformat(), fees),
        )
        return cur.lastrowid


def _pending_state(trade_id: int, order_id="EX-1", qty=2,
                   requested=15.50) -> dict:
    from trading_agent.exits.fill_confirm import (
        build_pending_state,
        write_pending_close,
    )
    state = build_pending_state(
        order_id=order_id, symbol="US.AAPL260626C325000",
        action="EXIT_STOP", reason="stop hit", qty=qty, side="SELL",
        asset_type="OPT", requested_exit_price=requested,
        quoted_bid=15.60, quoted_ask=16.40, thesis_id=None, source="sqlite",
    )
    assert write_pending_close(trade_id, state)
    return state


def test_pending_state_round_trips_through_sqlite(tmp_journal_db):
    from trading_agent.db import connection
    from trading_agent.exits.fill_confirm import pending_close_trade_ids_sqlite

    tid = _open_trade()
    _pending_state(tid)
    assert tid in pending_close_trade_ids_sqlite()
    with connection() as conn:
        row = conn.execute(
            "SELECT exit_order_id, exit_state_json, outcome FROM trades "
            "WHERE id = ?", (tid,)).fetchone()
    assert row["exit_order_id"] == "EX-1"
    assert row["outcome"] == "OPEN"  # NOT closed at placement
    assert json.loads(row["exit_state_json"])["requested_exit_price"] == 15.50


def test_filled_exit_closes_at_dealt_price_net_of_fees(tmp_journal_db):
    from trading_agent.db import connection
    from trading_agent.exits.fill_confirm import finalize_pending_exits

    tid = _open_trade(qty=2, entry=16.76, fees=2.0)
    _pending_state(tid, order_id="EX-1", qty=2)

    # Broker dealt 15.40 — NOT the 15.50 we requested, and not any mark.
    order_map = {"EX-1": {"order_status": "FILLED_ALL",
                          "dealt_avg_price": 15.40, "dealt_qty": 2}}
    stats = finalize_pending_exits(order_map, run_id="t", trigger="test")
    assert stats["confirmed"] == 1

    with connection() as conn:
        row = conn.execute(
            "SELECT outcome, exit_price, pnl, pnl_recomputed, fees, "
            "exit_order_id FROM trades WHERE id = ?", (tid,)).fetchone()
    assert row["outcome"] == "LOSS"
    assert row["exit_price"] == pytest.approx(15.40)
    # gross (15.40-16.76)×2×100 = -272; fees 2 entry + 2 exit = 4 → -276
    assert row["pnl"] == pytest.approx(-276.0)
    assert row["pnl_recomputed"] == pytest.approx(-276.0)
    assert row["fees"] == pytest.approx(4.0)
    assert row["exit_order_id"] is None  # pending state cleared


class _FakeMoomoo:
    """Minimal moomoo server stand-in for the state-machine tests."""

    placed: list[dict] = []
    cancelled: list[str] = []
    next_order_id = "EX-NEW"
    positions_rows: list[dict] = []
    quote_row = {"bid_price": 14.80, "ask_price": 15.20}

    @classmethod
    def reset(cls):
        cls.placed, cls.cancelled = [], []
        cls.next_order_id = "EX-NEW"
        cls.positions_rows = []
        cls.quote_row = {"bid_price": 14.80, "ask_price": 15.20}

    @classmethod
    def cancel_paper_order(cls, order_id):
        cls.cancelled.append(order_id)
        return {"rows": []}

    @classmethod
    def place_paper_option_order(cls, **kw):
        cls.placed.append(kw)
        return {"rows": [{"order_id": cls.next_order_id}]}

    @classmethod
    def place_paper_order(cls, **kw):
        cls.placed.append(kw)
        return {"rows": [{"order_id": cls.next_order_id}]}

    @classmethod
    def get_quote(cls, symbols):
        return {"rows": [cls.quote_row]}

    @classmethod
    def get_positions(cls):
        return {"rows": cls.positions_rows}


@pytest.fixture()
def fake_moomoo(monkeypatch):
    import sys
    _FakeMoomoo.reset()
    monkeypatch.setitem(
        sys.modules, "trading_agent.mcp_servers.moomoo.server", _FakeMoomoo)
    return _FakeMoomoo


def test_dead_exit_order_replaces_remainder(tmp_journal_db, fake_moomoo):
    """Dead order, nothing dealt → a fresh order for the full remainder is
    placed in the same pass; the journal row stays OPEN, guard stays armed."""
    from trading_agent.db import connection
    from trading_agent.exits.fill_confirm import finalize_pending_exits

    tid = _open_trade(qty=2)
    _pending_state(tid, order_id="EX-2", qty=2)
    order_map = {"EX-2": {"order_status": "CANCELLED_ALL", "dealt_qty": 0}}
    stats = finalize_pending_exits(order_map, run_id="t", trigger="test")
    assert stats["dead"] == 1
    assert fake_moomoo.placed and fake_moomoo.placed[0]["contracts"] == 2
    with connection() as conn:
        row = conn.execute(
            "SELECT outcome, exit_order_id FROM trades WHERE id = ?",
            (tid,)).fetchone()
    assert row["outcome"] == "OPEN"            # position NOT falsely closed
    assert row["exit_order_id"] == "EX-NEW"    # replacement is tracked


def test_cancelled_part_books_leg_then_blends_final_close(tmp_journal_db, fake_moomoo):
    """CANCELLED_PART: the dealt portion enters the books as a leg, the
    remainder is re-placed, and the final close blends both fills."""
    import json as _json

    from trading_agent.db import connection
    from trading_agent.exits.fill_confirm import finalize_pending_exits

    tid = _open_trade(qty=2, entry=16.76, fees=2.0)
    _pending_state(tid, order_id="EX-2", qty=2)

    # Tick 1: 1 of 2 dealt at 15.00, order cancelled → leg booked, 1 re-placed
    fake_moomoo.next_order_id = "EX-9"
    order_map = {"EX-2": {"order_status": "CANCELLED_PART",
                          "dealt_qty": 1, "dealt_avg_price": 15.00}}
    stats = finalize_pending_exits(order_map, run_id="t", trigger="test")
    assert stats["dead"] == 1
    assert fake_moomoo.placed[0]["contracts"] == 1  # remainder only
    with connection() as conn:
        row = conn.execute(
            "SELECT outcome, exit_state_json FROM trades WHERE id = ?",
            (tid,)).fetchone()
    assert row["outcome"] == "OPEN"
    state = _json.loads(row["exit_state_json"])
    assert state["dealt_legs"] == [{"qty": 1, "price": 15.00}]

    # Tick 2: the replacement fills 1 @ 14.00 → blended exit (15+14)/2 = 14.50
    order_map = {"EX-9": {"order_status": "FILLED_ALL",
                          "dealt_qty": 1, "dealt_avg_price": 14.00}}
    stats = finalize_pending_exits(order_map, run_id="t", trigger="test")
    assert stats["confirmed"] == 1
    with connection() as conn:
        row = conn.execute(
            "SELECT outcome, exit_price, pnl FROM trades WHERE id = ?",
            (tid,)).fetchone()
    assert row["outcome"] == "LOSS"
    assert row["exit_price"] == pytest.approx(14.50)
    # gross (14.50-16.76)×2×100 = -452; fees 2 entry + 2 exit → -456
    assert row["pnl"] == pytest.approx(-456.0)


def test_filled_part_is_never_cancelled(tmp_journal_db, fake_moomoo):
    """A partially-filled working order must not be repriced/cancelled even
    when stale — legs only enter the books from terminal states."""
    from trading_agent.exits import fill_confirm as fc

    tid = _open_trade(qty=2)
    state = _pending_state(tid, order_id="EX-3", qty=2)
    state["placed_at"] = "2020-01-01T00:00:00+00:00"  # ancient
    assert fc.write_pending_close(tid, state)
    order_map = {"EX-3": {"order_status": "FILLED_PART",
                          "dealt_qty": 1, "dealt_avg_price": 15.0}}
    stats = fc.finalize_pending_exits(order_map, run_id="t", trigger="test")
    assert stats["waiting"] == 1
    assert fake_moomoo.cancelled == []
    assert fake_moomoo.placed == []


def test_vanished_order_settles_unconfirmed_when_flat(tmp_journal_db, fake_moomoo):
    """Order absent from the today-only order map for MAX_MISSING_TICKS and
    the broker position is flat → settle at the requested price, FLAGGED
    unconfirmed (no livelock, no phantom open position)."""
    from trading_agent.db import connection
    from trading_agent.exits.fill_confirm import finalize_pending_exits

    tid = _open_trade(qty=2)
    _pending_state(tid, order_id="EX-GONE", qty=2, requested=15.50)
    fake_moomoo.positions_rows = []  # flat at the broker

    stats = finalize_pending_exits({}, run_id="t", trigger="test")
    assert stats["waiting"] == 1     # miss #1 — tolerated
    stats = finalize_pending_exits({}, run_id="t", trigger="test")
    assert stats["unresolved"] == 1  # miss #2 — reconciled via positions

    with connection() as conn:
        row = conn.execute(
            "SELECT outcome, exit_price FROM trades WHERE id = ?",
            (tid,)).fetchone()
    assert row["outcome"] in ("WIN", "LOSS", "SCRATCH")
    assert row["exit_price"] == pytest.approx(15.50)


def test_vanished_order_replaces_when_still_held(tmp_journal_db, fake_moomoo):
    """Order vanished but the position is still held → the dead order is
    replaced; nothing is journaled."""
    from trading_agent.db import connection
    from trading_agent.exits.fill_confirm import finalize_pending_exits

    tid = _open_trade(qty=2)
    _pending_state(tid, order_id="EX-GONE", qty=2)
    fake_moomoo.positions_rows = [
        {"code": "US.AAPL260626C325000", "qty": 2}]

    finalize_pending_exits({}, run_id="t", trigger="test")   # miss #1
    stats = finalize_pending_exits({}, run_id="t", trigger="test")  # miss #2
    assert stats["replaced"] == 1
    assert fake_moomoo.placed and fake_moomoo.placed[0]["contracts"] == 2
    with connection() as conn:
        row = conn.execute(
            "SELECT outcome FROM trades WHERE id = ?", (tid,)).fetchone()
    assert row["outcome"] == "OPEN"


def test_find_adoptable_close_order():
    from trading_agent.exits.fill_confirm import find_adoptable_close_order
    order_map = {
        "1": {"code": "US.AAPL", "trd_side": "SELL",
              "order_status": "FILLED_ALL", "qty": 2},   # done — not adoptable
        "2": {"code": "US.AAPL", "trd_side": "SELL",
              "order_status": "SUBMITTED", "qty": 2},    # live — adoptable
        "3": {"code": "US.MSFT", "trd_side": "SELL",
              "order_status": "SUBMITTED", "qty": 1},    # wrong symbol
    }
    assert find_adoptable_close_order(order_map, "US.AAPL", "SELL") == ("2", 2.0)
    assert find_adoptable_close_order(order_map, "US.NVDA", "SELL") is None


def test_pending_lookup_returns_none_on_store_error(tmp_path, monkeypatch):
    """Fail-closed contract: an unreadable store returns None (unknown),
    never an empty set (which the caller would read as 'no pending exits')."""
    import dataclasses

    from trading_agent import config as config_mod
    from trading_agent import db as db_mod
    from trading_agent.exits.fill_confirm import pending_close_trade_ids_sqlite

    # Point at a directory that is not a database → connect/select fails
    bad = tmp_path / "not_a_db"
    bad.mkdir()
    new_cfg = dataclasses.replace(config_mod.CONFIG, db_path=bad)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(db_mod, "CONFIG", new_cfg)
    assert pending_close_trade_ids_sqlite() is None


def test_resting_young_order_waits(tmp_journal_db):
    from trading_agent.exits.fill_confirm import finalize_pending_exits

    tid = _open_trade()
    _pending_state(tid, order_id="EX-3")
    # Order resting, just placed → no reprice yet
    order_map = {"EX-3": {"order_status": "SUBMITTED"}}
    stats = finalize_pending_exits(order_map, run_id="t", trigger="test")
    assert stats["waiting"] == 1
    assert stats["confirmed"] == 0 and stats["repriced"] == 0


def test_stale_resting_order_repriced(tmp_journal_db, monkeypatch):
    from trading_agent.db import connection
    from trading_agent.exits import fill_confirm as fc

    tid = _open_trade()
    state = _pending_state(tid, order_id="EX-4")
    # Age the order past the reprice window
    state["placed_at"] = "2020-01-01T00:00:00+00:00"
    assert fc.write_pending_close(tid, state)

    calls = {}

    class _FakeMoomoo:
        @staticmethod
        def cancel_paper_order(order_id):
            calls["cancelled"] = order_id
            return {"rows": []}

        @staticmethod
        def place_paper_option_order(**kw):
            calls["placed"] = kw
            return {"rows": [{"order_id": "EX-5"}]}

        @staticmethod
        def place_paper_order(**kw):  # pragma: no cover - OPT path used
            return {"rows": [{"order_id": "EX-5"}]}

        @staticmethod
        def get_quote(symbols):
            return {"rows": [{"bid_price": 14.80, "ask_price": 15.20}]}

    import sys
    monkeypatch.setitem(
        sys.modules, "trading_agent.mcp_servers.moomoo.server", _FakeMoomoo)

    order_map = {"EX-4": {"order_status": "SUBMITTED"}}
    stats = fc.finalize_pending_exits(order_map, run_id="t", trigger="test")
    assert stats["repriced"] == 1
    assert calls["cancelled"] == "EX-4"
    # New limit = fresh bid 14.80 × 0.98 = 14.50
    assert calls["placed"]["price"] == pytest.approx(14.50, abs=0.011)

    with connection() as conn:
        row = conn.execute(
            "SELECT exit_order_id, exit_state_json FROM trades WHERE id = ?",
            (tid,)).fetchone()
    new_state = json.loads(row["exit_state_json"])
    assert row["exit_order_id"] == "EX-5"
    assert new_state["attempts"] == 1


def test_close_trade_flags_self_reported_pnl_mismatch(tmp_journal_db):
    """The id-8 case: caller-supplied pnl contradicting leg arithmetic."""
    from trading_agent.db import connection
    from trading_agent.mcp_servers.journal.server import close_trade

    tid = _open_trade(qty=4, entry=16.76, fees=4.0)
    res = close_trade(trade_id=tid, exit_price=16.09, outcome="LOSS",
                      pnl=-136.0)
    # legs: (16.09-16.76)×4×100 = -268 gross; fees 4+4 → recomputed -276
    assert res["pnl_recomputed"] == pytest.approx(-276.0)
    assert res["pnl_mismatch"] is True
    with connection() as conn:
        row = conn.execute(
            "SELECT pnl, pnl_recomputed, pnl_mismatch FROM trades "
            "WHERE id = ?", (tid,)).fetchone()
    assert row["pnl"] == pytest.approx(-136.0)       # original preserved
    assert row["pnl_recomputed"] == pytest.approx(-276.0)
    assert row["pnl_mismatch"] == 1


def test_close_trade_computes_net_pnl_when_omitted(tmp_journal_db):
    from trading_agent.mcp_servers.journal.server import close_trade

    tid = _open_trade(qty=2, entry=2.00, fees=2.0)
    res = close_trade(trade_id=tid, exit_price=3.00, outcome="WIN")
    # gross (3-2)×2×100 = 200; fees 2 entry + 2 exit = 4 → 196
    assert res["pnl"] == pytest.approx(196.0)
    assert res["pnl_mismatch"] is False


def test_close_trade_is_idempotent(tmp_journal_db):
    """A second close (MCP retry, double-fire) must not accumulate phantom
    exit fees or rewrite pnl — it returns the existing close untouched."""
    from trading_agent.db import connection
    from trading_agent.mcp_servers.journal.server import close_trade

    tid = _open_trade(qty=2, entry=2.00, fees=2.0)
    first = close_trade(trade_id=tid, exit_price=3.00, outcome="WIN")
    second = close_trade(trade_id=tid, exit_price=1.00, outcome="LOSS")
    assert second.get("error") == "already closed"
    with connection() as conn:
        row = conn.execute(
            "SELECT outcome, pnl, fees, exit_price FROM trades WHERE id = ?",
            (tid,)).fetchone()
    assert row["outcome"] == "WIN"                    # first close stands
    assert row["pnl"] == pytest.approx(first["pnl"])
    assert row["fees"] == pytest.approx(first["fees"])  # no fee accumulation
    assert row["exit_price"] == pytest.approx(3.00)


def test_virtual_close_charges_exit_half_spread(tmp_journal_db):
    """A virtual position closes at a MARK — close_trade must charge the
    exit-side half-spread on top of fees (entry side was charged at record
    time). 4% of 3.00 × 2 × 100 = $24 + $2 exit fees."""
    from trading_agent.db import connection
    from trading_agent.mcp_servers.journal.server import close_trade

    from trading_agent import execution_costs as ec
    ec.reset_calibration_cache()

    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO trades (symbol, asset_type, side, qty, entry_price, "
            "opened_at, outcome, broker_order_id, is_paper, provenance, fees) "
            "VALUES ('US.GLD260522C445000', 'OPT', 'BUY', 2, 2.00, "
            "'2026-06-01T00:00:00+00:00', 'OPEN', 'VIRTUAL-abc123', 1, "
            "'virtual', 2.0)")
        tid = cur.lastrowid
    res = close_trade(trade_id=tid, exit_price=3.00, outcome="WIN")
    # gross 200 − entry fees 2 − exit fees 2 − exit half-spread 24 = 172
    assert res["pnl"] == pytest.approx(172.0)
