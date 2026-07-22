"""Two-leg (combo) pending-close state machine (M1-0.1 lane A).

Core invariants pinned here:
  * the journal row settles ONLY when BOTH legs are fully dealt — never
    while any leg has a live broker order;
  * settlement price is net-of-legs (net_debit = short_avg − long_avg) with
    TWO-leg exit fees;
  * a dead leg is re-placed alone (sized target − dealt, never over-close);
  * trims complete by clearing pending and leaving the row OPEN;
  * a guard block on a leg re-placement alerts (a guard blocking a combo
    close leg is always a bug).
"""
from __future__ import annotations

import pytest


def _combo_state(*, short_oid="BTC-1", long_oid="STC-1", units=2.0,
                 settle_mode="full", net_credit=1.2,
                 short_px=0.66, long_px=0.19,
                 broker_qty_before=None) -> dict:
    from trading_agent.exits.fill_confirm import build_pending_combo_state
    state = build_pending_combo_state(
        symbol="US.SPY261016P00100000",
        action="EXIT_TARGET",
        reason="spread value <= 50% of net credit",
        units=units,
        settle_mode=settle_mode,
        net_credit=net_credit,
        short_leg="US.SPY261016P00100000",
        long_leg="US.SPY261016P00095000",
        short_order_id=short_oid,
        long_order_id=long_oid,
        short_price=short_px,
        long_price=long_px,
        short_bid=0.60, short_ask=0.65,
        long_bid=0.20, long_ask=0.24,
        thesis_id=9,
        source="postgres",
        broker_qty_before=broker_qty_before,
    )
    return state


def _legs(state):
    short = next(x for x in state["legs"] if x["leg"] == "short_close")
    long = next(x for x in state["legs"] if x["leg"] == "long_close")
    return short, long


class _Capture:
    def __init__(self):
        self.events: list[dict] = []
        self.settles: list[tuple] = []
        self.writes: list[dict] = []
        self.cleared: list[int] = []

    def emit(self, **kw):
        self.events.append(kw)

    def types(self):
        return [e.get("event_type") for e in self.events]


@pytest.fixture()
def cap(monkeypatch):
    from trading_agent.exits import fill_confirm as fc
    c = _Capture()
    monkeypatch.setattr(fc, "emit", c.emit)
    monkeypatch.setattr(
        fc, "write_pending_close",
        lambda tid, state: (c.writes.append(dict(state)) or True))
    monkeypatch.setattr(
        fc, "_clear_pending", lambda tid, state: c.cleared.append(tid))
    return c


def _patch_settle(monkeypatch, cap, ok=True, net=42.0):
    from trading_agent.exits import fill_confirm as fc

    def _settle(trade_id, state, final_qty, final_price, *, unconfirmed=False,
                n_legs=1):
        cap.settles.append((trade_id, final_qty, final_price, unconfirmed, n_legs))
        return ok, net
    monkeypatch.setattr(fc, "_settle", _settle)


class _FakeMoomoo:
    placed: list[dict] = []
    cancelled: list[str] = []
    next_order_id = "LEG-NEW"
    blocked = False
    positions_rows: list[dict] = []

    @classmethod
    def reset(cls):
        cls.placed, cls.cancelled = [], []
        cls.next_order_id = "LEG-NEW"
        cls.blocked = False
        cls.positions_rows = []

    @classmethod
    def place_paper_option_order(cls, **kw):
        cls.placed.append(kw)
        if cls.blocked:
            return {"order_blocked": True, "violations": ["R_x"]}
        return {"rows": [{"order_id": cls.next_order_id}]}

    @classmethod
    def cancel_paper_order(cls, order_id):
        cls.cancelled.append(order_id)
        return {"rows": []}

    @classmethod
    def get_quote(cls, symbols):
        return {"rows": [{"bid_price": 0.58, "ask_price": 0.68}]}

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


def test_both_legs_filled_settles_net_debit_with_two_leg_fees(cap, monkeypatch):
    from trading_agent.exits import fill_confirm as fc
    _patch_settle(monkeypatch, cap)
    state = _combo_state()
    order_map = {
        "BTC-1": {"order_status": "FILLED_ALL",
                  "dealt_qty": 2, "dealt_avg_price": 0.64},
        "STC-1": {"order_status": "FILLED_ALL",
                  "dealt_qty": 2, "dealt_avg_price": 0.21},
    }
    fc._finalize_combo_close(7, state, order_map, "r", "t")
    assert len(cap.settles) == 1
    trade_id, qty, price, unconfirmed, n_legs = cap.settles[0]
    assert trade_id == 7
    assert qty == pytest.approx(2.0)
    # M1-0.2 clamp: dealt 0.64 beats the quoted ask 0.65 on the BTC leg →
    # honest short close = 0.65; dealt 0.21 beats the quoted bid 0.20 on
    # the STC leg → honest long close = 0.20. net_debit = 0.45, raw 0.43.
    assert price == pytest.approx(0.65 - 0.20)   # honest net debit
    assert unconfirmed is False
    assert n_legs == 2                            # 双腿双向 fee accounting
    types = cap.types()
    assert "position_closed" in types
    closed = next(e for e in cap.events
                  if e.get("event_type") == "position_closed")
    assert closed["payload"]["combo"] is True
    assert closed["payload"]["exit_price"] == pytest.approx(0.45)
    assert closed["payload"]["net_debit_raw"] == pytest.approx(0.43)


def test_net_pnl_two_leg_fee_arithmetic():
    """gross (net_credit − net_debit)×qty×100 minus 2-leg fees each side."""
    from trading_agent.exits.fill_confirm import _net_pnl_for_close
    net, exit_fees = _net_pnl_for_close(
        entry_price=1.2, dealt_exit_price=0.55, qty=1.0, asset_type="OPT",
        side_opened="SELL", entry_fees=None, n_legs=2)
    # gross (1.2-0.55)×1×100 = 65; fees: 2 legs × $1 × entry + exit = 4
    assert exit_fees == pytest.approx(2.0)
    assert net == pytest.approx(61.0)


def test_net_pnl_default_n_legs_matches_pre_change_fees():
    """No-behavior-change lock: default n_legs=1 is bit-identical to the
    pre-combo fee model for existing single-leg callers."""
    from trading_agent.exits.fill_confirm import _net_pnl_for_close
    net, exit_fees = _net_pnl_for_close(
        entry_price=16.76, dealt_exit_price=15.40, qty=2.0, asset_type="OPT",
        side_opened="BUY", entry_fees=2.0)
    assert exit_fees == pytest.approx(2.0)
    assert net == pytest.approx(-276.0)  # matches the single-leg pin test


def test_dead_leg_replaced_alone(cap, fake_moomoo, monkeypatch):
    """Long-leg close died; short leg filled — ONLY the long leg is
    re-placed, sized to its remainder, and nothing settles."""
    from trading_agent.exits import fill_confirm as fc
    _patch_settle(monkeypatch, cap)
    state = _combo_state()
    order_map = {
        "BTC-1": {"order_status": "FILLED_ALL",
                  "dealt_qty": 2, "dealt_avg_price": 0.64},
        "STC-1": {"order_status": "CANCELLED_ALL", "dealt_qty": 0},
    }
    fc._finalize_combo_close(7, state, order_map, "r", "t")
    assert cap.settles == []                     # one leg not dealt → no settle
    assert len(fake_moomoo.placed) == 1
    placed = fake_moomoo.placed[0]
    assert placed["option_symbol"] == "US.SPY261016P00095000"
    assert placed["side"] == "SELL"
    assert placed["contracts"] == 2
    short, long = _legs(state)
    assert short["dealt_legs"] == [{"qty": 2.0, "price": 0.64}]
    assert long["order_id"] == "LEG-NEW"
    assert cap.writes                            # progress persisted


def test_partial_terminal_books_leg_then_remainder(cap, fake_moomoo, monkeypatch):
    """CANCELLED_PART on a leg: the dealt portion enters that leg's books,
    the remainder is re-placed (target − dealt, never over-close)."""
    from trading_agent.exits import fill_confirm as fc
    _patch_settle(monkeypatch, cap)
    state = _combo_state()
    order_map = {
        "BTC-1": {"order_status": "CANCELLED_PART",
                  "dealt_qty": 1, "dealt_avg_price": 0.65},
        "STC-1": {"order_status": "SUBMITTED"},
    }
    fc._finalize_combo_close(7, state, order_map, "r", "t")
    short, _long = _legs(state)
    assert short["dealt_legs"] == [{"qty": 1.0, "price": 0.65}]
    assert len(fake_moomoo.placed) == 1
    assert fake_moomoo.placed[0]["contracts"] == 1   # remainder only
    assert fake_moomoo.placed[0]["side"] == "BUY"
    assert "exit_partial_terminal" in cap.types()
    assert cap.settles == []


def test_never_settles_with_one_leg_live(cap, monkeypatch):
    """Short filled, long still resting → wait. The journal row is NEVER
    settled while any leg has a live broker order."""
    from trading_agent.exits import fill_confirm as fc
    _patch_settle(monkeypatch, cap)
    state = _combo_state()
    order_map = {
        "BTC-1": {"order_status": "FILLED_ALL",
                  "dealt_qty": 2, "dealt_avg_price": 0.64},
        "STC-1": {"order_status": "SUBMITTED"},
    }
    fc._finalize_combo_close(7, state, order_map, "r", "t")
    assert cap.settles == []
    assert "position_closed" not in cap.types()
    _short, long = _legs(state)
    assert long["order_id"] == "STC-1"           # still tracked


def test_filled_part_leg_is_never_cancelled(cap, fake_moomoo, monkeypatch):
    from trading_agent.exits import fill_confirm as fc
    _patch_settle(monkeypatch, cap)
    state = _combo_state()
    short, _long = _legs(state)
    short["placed_at"] = "2020-01-01T00:00:00+00:00"  # ancient
    order_map = {
        "BTC-1": {"order_status": "FILLED_PART",
                  "dealt_qty": 1, "dealt_avg_price": 0.65},
        "STC-1": {"order_status": "SUBMITTED"},
    }
    fc._finalize_combo_close(7, state, order_map, "r", "t")
    assert fake_moomoo.cancelled == []
    assert fake_moomoo.placed == []


def test_vanished_leg_reconciles_per_leg_position(cap, fake_moomoo, monkeypatch):
    """Long-leg order vanished from the today-only map; leg flat at the
    broker after MAX_MISSING_TICKS → that leg settles UNCONFIRMED at its
    requested price and the flag propagates to the row settle."""
    from trading_agent.exits import fill_confirm as fc
    _patch_settle(monkeypatch, cap)
    fake_moomoo.positions_rows = []              # flat — the close filled
    state = _combo_state()
    order_map = {
        "BTC-1": {"order_status": "FILLED_ALL",
                  "dealt_qty": 2, "dealt_avg_price": 0.64},
        # STC-1 absent
    }
    fc._finalize_combo_close(7, state, order_map, "r", "t")   # miss #1
    assert cap.settles == []
    fc._finalize_combo_close(7, state, {}, "r", "t")          # miss #2
    assert len(cap.settles) == 1
    _tid, qty, price, unconfirmed, n_legs = cap.settles[0]
    assert unconfirmed is True
    # long settled at its requested price 0.19 (below the quoted bid 0.20 →
    # clamp no-op); short dealt 0.64 clamps up to the quoted ask 0.65.
    assert price == pytest.approx(0.65 - 0.19)
    assert "combo_close_leg_unconfirmed" in cap.types()


def test_vanished_leg_still_held_replaces(cap, fake_moomoo, monkeypatch):
    from trading_agent.exits import fill_confirm as fc
    _patch_settle(monkeypatch, cap)
    fake_moomoo.positions_rows = [
        {"code": "US.SPY261016P00095000", "qty": 2}]
    state = _combo_state()
    order_map = {
        "BTC-1": {"order_status": "FILLED_ALL",
                  "dealt_qty": 2, "dealt_avg_price": 0.64},
    }
    fc._finalize_combo_close(7, state, order_map, "r", "t")   # miss #1
    fc._finalize_combo_close(7, state, {}, "r", "t")          # miss #2
    assert cap.settles == []
    assert len(fake_moomoo.placed) == 1
    assert fake_moomoo.placed[0]["option_symbol"] == "US.SPY261016P00095000"


def test_trim_completion_leaves_row_open(cap, monkeypatch):
    """settle_mode 'trim': pending clears, journal row stays OPEN (no
    _settle), trim P&L is event-only (combo_trimmed) — C8."""
    from trading_agent.exits import fill_confirm as fc
    _patch_settle(monkeypatch, cap)
    state = _combo_state(units=1.0, settle_mode="trim")
    order_map = {
        "BTC-1": {"order_status": "FILLED_ALL",
                  "dealt_qty": 1, "dealt_avg_price": 0.60},
        "STC-1": {"order_status": "FILLED_ALL",
                  "dealt_qty": 1, "dealt_avg_price": 0.20},
    }
    fc._finalize_combo_close(7, state, order_map, "r", "t")
    assert cap.settles == []                     # row NOT closed
    assert cap.cleared == [7]                    # pending cleared
    trimmed = next(e for e in cap.events
                   if e.get("event_type") == "combo_trimmed")
    # M1-0.2 clamp: BTC dealt 0.60 < quoted ask 0.65 → honest 0.65; STC
    # dealt 0.20 = quoted bid → no-op. net_debit 0.45 (raw 0.40).
    # realized = (1.2 − 0.45)×1×100 − 2-leg exit fees (2×$1) = 73
    assert trimmed["payload"]["net_debit"] == pytest.approx(0.45)
    assert trimmed["payload"]["net_debit_raw"] == pytest.approx(0.40)
    assert trimmed["payload"]["realized"] == pytest.approx(73.0)


def test_long_leg_replacement_passes_guard_or_alerts(cap, fake_moomoo, monkeypatch):
    """A guard block on a combo close-leg re-placement is always a bug —
    sev-2 exit_order_blocked_by_guard, alerted once per leg."""
    from trading_agent.exits import fill_confirm as fc
    _patch_settle(monkeypatch, cap)
    fake_moomoo.blocked = True
    state = _combo_state(long_oid=None)          # long leg needs placement
    order_map = {
        "BTC-1": {"order_status": "FILLED_ALL",
                  "dealt_qty": 2, "dealt_avg_price": 0.64},
    }
    fc._finalize_combo_close(7, state, order_map, "r", "t")
    blocked = [e for e in cap.events
               if e.get("event_type") == "exit_order_blocked_by_guard"]
    assert len(blocked) == 1
    assert blocked[0]["severity"] == 2
    # second tick: alerted once, not spammed
    fc._finalize_combo_close(7, state, order_map, "r", "t")
    blocked = [e for e in cap.events
               if e.get("event_type") == "exit_order_blocked_by_guard"]
    assert len(blocked) == 1
    assert cap.settles == []


def test_negative_qty_short_leg_not_settled_as_filled(cap, fake_moomoo,
                                                      monkeypatch):
    """Broker reports the short via NEGATIVE qty: the short leg is STILL
    held — the vanished BTC must be re-placed, never falsely booked as
    dealt-unconfirmed (which would settle the row while a naked short
    remains at the broker — the inverse orphan)."""
    from trading_agent.exits import fill_confirm as fc
    _patch_settle(monkeypatch, cap)
    fake_moomoo.positions_rows = [
        {"code": "US.SPY261016P00100000", "qty": -2,
         "position_side": "SHORT"}]
    state = _combo_state()
    order_map = {
        "STC-1": {"order_status": "FILLED_ALL",
                  "dealt_qty": 2, "dealt_avg_price": 0.21},
        # BTC-1 absent (vanished)
    }
    fc._finalize_combo_close(7, state, order_map, "r", "t")   # miss #1
    fc._finalize_combo_close(7, state, {}, "r", "t")          # miss #2
    assert "combo_close_leg_unconfirmed" not in cap.types()
    assert cap.settles == []
    assert len(fake_moomoo.placed) == 1                       # re-placed
    assert fake_moomoo.placed[0]["option_symbol"] == "US.SPY261016P00100000"
    assert fake_moomoo.placed[0]["side"] == "BUY"


def test_trim_vanished_leg_qty_reduced_books_unconfirmed(cap, fake_moomoo,
                                                         monkeypatch):
    """TRIM: the leg order fills late then vanishes from the today-only
    map. The position is never flat after a partial close, so presence
    alone would re-place and over-trim — the broker qty already being
    reduced by the trim amount is the 'filled' signal."""
    from trading_agent.exits import fill_confirm as fc
    _patch_settle(monkeypatch, cap)
    # trim 1 of 2: short leg already reduced 2 → 1 at the broker
    fake_moomoo.positions_rows = [
        {"code": "US.SPY261016P00100000", "qty": 1,
         "position_side": "SHORT"},
        {"code": "US.SPY261016P00095000", "qty": 1,
         "position_side": "LONG"}]
    state = _combo_state(units=1.0, settle_mode="trim",
                         broker_qty_before=2.0)
    order_map = {
        "STC-1": {"order_status": "FILLED_ALL",
                  "dealt_qty": 1, "dealt_avg_price": 0.20},
        # BTC-1 absent (filled after the day's last tick)
    }
    fc._finalize_combo_close(7, state, order_map, "r", "t")   # miss #1
    fc._finalize_combo_close(7, state, {}, "r", "t")          # miss #2
    assert fake_moomoo.placed == [], "re-placing would over-trim one unit"
    assert "combo_close_leg_unconfirmed" in cap.types()
    trimmed = [e for e in cap.events
               if e.get("event_type") == "combo_trimmed"]
    assert len(trimmed) == 1
    assert trimmed[0]["payload"]["unconfirmed"] is True
    assert cap.cleared == [7]                    # trim completes, row OPEN


def test_trim_vanished_leg_qty_intact_replaces(cap, fake_moomoo, monkeypatch):
    """TRIM: order vanished with the broker qty UNREDUCED → it died
    invisibly; re-place the leg close for the trim amount."""
    from trading_agent.exits import fill_confirm as fc
    _patch_settle(monkeypatch, cap)
    fake_moomoo.positions_rows = [
        {"code": "US.SPY261016P00100000", "qty": 2,
         "position_side": "SHORT"},
        {"code": "US.SPY261016P00095000", "qty": 2,
         "position_side": "LONG"}]
    state = _combo_state(units=1.0, settle_mode="trim",
                         broker_qty_before=2.0)
    order_map = {
        "STC-1": {"order_status": "FILLED_ALL",
                  "dealt_qty": 1, "dealt_avg_price": 0.20},
    }
    fc._finalize_combo_close(7, state, order_map, "r", "t")   # miss #1
    fc._finalize_combo_close(7, state, {}, "r", "t")          # miss #2
    assert "combo_close_leg_unconfirmed" not in cap.types()
    assert len(fake_moomoo.placed) == 1
    assert fake_moomoo.placed[0]["option_symbol"] == "US.SPY261016P00100000"
    assert fake_moomoo.placed[0]["contracts"] == 1            # trim amount


def test_trim_vanished_leg_ambiguous_qty_alerts_and_waits(cap, fake_moomoo,
                                                          monkeypatch):
    """Neither expected state matches (manual interference?) — never guess
    with broker orders: sev-2 once, retry next tick."""
    from trading_agent.exits import fill_confirm as fc
    _patch_settle(monkeypatch, cap)
    fake_moomoo.positions_rows = [
        {"code": "US.SPY261016P00100000", "qty": 5,   # neither 1 nor 2
         "position_side": "SHORT"}]
    state = _combo_state(units=1.0, settle_mode="trim",
                         broker_qty_before=2.0)
    order_map = {
        "STC-1": {"order_status": "FILLED_ALL",
                  "dealt_qty": 1, "dealt_avg_price": 0.20},
    }
    fc._finalize_combo_close(7, state, order_map, "r", "t")   # miss #1
    fc._finalize_combo_close(7, state, {}, "r", "t")          # miss #2
    fc._finalize_combo_close(7, state, {}, "r", "t")          # miss #3
    assert fake_moomoo.placed == []
    assert cap.settles == []
    ambiguous = [e for e in cap.events
                 if e.get("event_type") == "combo_close_leg_ambiguous"]
    assert len(ambiguous) == 1 and ambiguous[0]["severity"] == 2


@pytest.fixture()
def twin_db(tmp_path, monkeypatch):
    """Real tmp SQLite (both stores' twin-close path); Postgres dark."""
    import dataclasses
    from trading_agent import config as config_mod
    from trading_agent import db as db_mod
    db_file = tmp_path / "trader_test.db"
    new_cfg = dataclasses.replace(config_mod.CONFIG, db_path=db_file)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(db_mod, "CONFIG", new_cfg)
    db_mod.migrate(db_file)

    def _boom():
        raise ConnectionError("no postgres in tests")
    monkeypatch.setattr("trading_agent.store.postgres.cursor", _boom)
    yield db_file


def _insert_sqlite_twin(short: str, long: str, qty: float = 2.0) -> int:
    import json as _json
    from datetime import datetime, timezone
    from trading_agent.db import connection
    now = datetime.now(timezone.utc).isoformat()
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO trades (symbol, asset_type, side, qty, entry_price, "
            "opened_at, outcome, is_paper) VALUES (?, 'OPT', 'SELL', ?, 1.2, "
            "?, 'OPEN', 1)", (short, qty, now))
        tid = int(cur.lastrowid)
        conn.execute(
            "INSERT INTO market_snapshots (trade_id, taken_at, payload) "
            "VALUES (?, ?, ?)",
            (tid, now, _json.dumps({
                "combo": True, "short_leg": short, "long_leg": long,
                "net_credit": 1.2, "width": 5.0, "max_loss": 760.0,
                "contracts": qty, "broker_order_ids": ["L1", "S1"]})))
        conn.commit()
    return tid


def test_full_settle_closes_sqlite_twin_and_frees_wing(cap, twin_db,
                                                       monkeypatch):
    """An engine combo close settles the PG row AND the SQLite twin the
    posttool hook dual-wrote — leaving the twin OPEN keeps the wing in
    open_combo_long_legs forever, letting a future SELL on that exact
    symbol classify as intent='close' and bypass R_NO_SHORT_OPEN."""
    from trading_agent.combo import open_combo_long_legs
    from trading_agent.db import connection
    from trading_agent.exits import fill_confirm as fc
    _patch_settle(monkeypatch, cap, ok=True, net=146.0)
    short, long = "US.SPY261016P00100000", "US.SPY261016P00095000"
    tid = _insert_sqlite_twin(short, long)
    assert long in open_combo_long_legs()
    state = _combo_state()
    order_map = {
        "BTC-1": {"order_status": "FILLED_ALL",
                  "dealt_qty": 2, "dealt_avg_price": 0.64},
        "STC-1": {"order_status": "FILLED_ALL",
                  "dealt_qty": 2, "dealt_avg_price": 0.21},
    }
    fc._finalize_combo_close(7, state, order_map, "r", "t")
    assert len(cap.settles) == 1                 # PG row settled
    with connection() as conn:
        row = conn.execute(
            "SELECT outcome, exit_price, pnl FROM trades WHERE id = ?",
            (tid,)).fetchone()
    assert row["outcome"] == "WIN"
    # honest net debit: 0.64 clamps up to ask 0.65, 0.21 down to bid 0.20
    assert row["exit_price"] == pytest.approx(0.45)
    assert row["pnl"] == pytest.approx(146.0)
    assert long not in open_combo_long_legs()    # wing freed


def test_trim_settle_leaves_sqlite_twin_open(cap, twin_db, monkeypatch):
    """A TRIM must NOT close the twin — the residual units stay open in
    both stores."""
    from trading_agent.db import connection
    from trading_agent.exits import fill_confirm as fc
    _patch_settle(monkeypatch, cap)
    short, long = "US.SPY261016P00100000", "US.SPY261016P00095000"
    tid = _insert_sqlite_twin(short, long)
    state = _combo_state(units=1.0, settle_mode="trim",
                         broker_qty_before=2.0)
    order_map = {
        "BTC-1": {"order_status": "FILLED_ALL",
                  "dealt_qty": 1, "dealt_avg_price": 0.60},
        "STC-1": {"order_status": "FILLED_ALL",
                  "dealt_qty": 1, "dealt_avg_price": 0.20},
    }
    fc._finalize_combo_close(7, state, order_map, "r", "t")
    with connection() as conn:
        row = conn.execute(
            "SELECT outcome FROM trades WHERE id = ?", (tid,)).fetchone()
    assert row["outcome"] == "OPEN"


def test_finalize_dispatches_combo_states(monkeypatch):
    """finalize_pending_exits routes combo-shaped states to the two-leg
    machine and leaves single-leg handling untouched."""
    from trading_agent.exits import fill_confirm as fc
    seen = []
    monkeypatch.setattr(
        fc, "_iter_pending",
        lambda: [(3, {"combo": True, "legs": [], "source": "postgres"})])
    monkeypatch.setattr(
        fc, "_finalize_combo_close",
        lambda tid, state, om, r, t: seen.append(tid))
    stats = fc.finalize_pending_exits({}, "r", "t")
    assert seen == [3]
    assert stats["combo"] == 1
    assert stats["confirmed"] == 0
