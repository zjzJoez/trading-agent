"""M1-0.2 — combo paper fills are cost-honest on the FULL round trip.

The moomoo SIMULATE environment books limit orders at the caller's limit
price, so a mid-priced entry "fills" at mid and never pays the spread.
The fix is the "fill no better than the touch" clamp, applied at BOTH ends:

  * ENTRY (_sync_combo_entry): honest_short = min(dealt, entry_bid),
    honest_long = max(dealt, entry_ask) — quotes captured from the R5d
    guard snapshots, echoed through place_paper_option_combo and persisted
    in the combo payload by posttool_fill_capture;
  * CLOSE (_finalize_combo_close): honest_short_close = max(dealt, ask),
    honest_long_close = min(dealt, bid) — quotes persisted in the pending
    combo state at placement time;
  * either side missing a quote → charge the calibrated per-underlying
    half-spread off the dealt price instead.

The round-trip test drives the REAL pipeline end to end: posttool capture
(SQLite twin + Postgres dual-write) → entry fill sync (honest credit) →
two-leg close finalize (honest debit) → journal close, asserting that the
final journal P&L equals

    (net_credit_honest − net_debit_honest) × 100 × contracts − fees

with fees = entry 2-leg + exit 2-leg, and that the total modeled friction
decomposes into exactly fees + the two clamp-model spread charges.
"""
from __future__ import annotations

import dataclasses
import io
import json
from datetime import UTC

import pytest

SHORT = "US.SPY261016P00100000"
LONG = "US.SPY261016P00095000"

ENTRY_QUOTES = {"short": {"bid": 1.10, "ask": 1.30},
                "long": {"bid": 0.28, "ask": 0.40}}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

@pytest.fixture()
def combo_db(tmp_path, monkeypatch):
    """tmp SQLite for the posttool twin + isolated cost calibration (none →
    library defaults: $1/contract/side, 4% half-spread of mark)."""
    from trading_agent import config as config_mod
    from trading_agent import db as db_mod
    from trading_agent import execution_costs as ec
    from trading_agent.hooks import posttool_fill_capture as pf

    db_file = tmp_path / "trader_test.db"
    new_cfg = dataclasses.replace(
        config_mod.CONFIG, db_path=db_file, data_dir=tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(db_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(pf, "AUDIT_PATH", tmp_path / "hook_audit.log")
    db_mod.migrate(db_file)
    ec.reset_calibration_cache()
    yield tmp_path
    ec.reset_calibration_cache()


class _FakePg:
    """Minimal stateful journal_trades twin for the exact SQL this round
    trip issues (insert → entry sync UPDATE → close SELECT/UPDATE)."""

    def __init__(self):
        self.row: dict | None = None
        self.close_params: tuple | None = None
        self._pending_select: tuple | None = None

    def cursor(self):
        fake = self

        class _Cur:
            def __enter__(cur):
                return cur

            def __exit__(cur, *a):
                return False

            def execute(cur, sql, params=None):
                fake._execute(sql, params or ())

            def fetchone(cur):
                return fake._pending_select

        return _Cur()

    def _execute(self, sql, params):
        if "INSERT INTO journal_trades" in sql:
            symbol, qty, entry_price, boid, _plan, fill_json = params
            self.row = {
                "id": 1, "symbol": symbol, "qty": float(qty),
                "entry_price": float(entry_price), "asset_type": "OPT",
                "side": "SELL", "thesis_id": 4, "outcome": "OPEN",
                "broker_fill_json": json.loads(fill_json),
            }
        elif "SET entry_price" in sql:
            honest, avg, raw, honest2, trade_id = params
            assert self.row is not None and self.row["outcome"] == "OPEN"
            self.row["entry_price"] = float(honest)
            self.row["broker_fill_json"].update({
                "status": "FILLED_ALL", "combo_fill_synced": True,
                "avg_fill_price": float(avg),
                "entry_credit_raw": float(raw),
                "entry_credit_honest": float(honest2),
            })
        elif "SELECT entry_price, qty, asset_type, side, thesis_id" in sql:
            r = self.row
            self._pending_select = None if r is None or r["outcome"] != "OPEN" else (
                r["entry_price"], r["qty"], r["asset_type"], r["side"],
                r["thesis_id"], r["broker_fill_json"])
        elif "SET exit_price" in sql:
            self.close_params = params
            self.row["outcome"] = params[1]
            self.row["exit_price"] = float(params[0])
        # theses updates / pending-state writes+clears: ignored


def _insert_thesis(ticker: str = "SPY") -> int:
    """trades.thesis_id is a FK → theses.id; production always has one."""
    from datetime import datetime

    from trading_agent.db import connection
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO theses (created_at, ticker, direction, thesis_text, "
            "invalidation, status) VALUES (?, ?, 'LONG', 't', 'i', 'open')",
            (datetime.now(UTC).isoformat(), ticker),
        )
        conn.commit()
        return int(cur.lastrowid)


def _run_posttool(monkeypatch, *, contracts=2, net_credit=0.86,
                  leg_quotes=ENTRY_QUOTES) -> int:
    """Drive the REAL posttool capture path with a combo tool response."""
    from trading_agent.hooks import posttool_fill_capture as pf
    thesis_id = _insert_thesis("SPY")
    payload = {
        "tool_name": "mcp__moomoo-mcp__place_paper_option_combo",
        "tool_input": {
            "short_leg_symbol": SHORT, "long_leg_symbol": LONG,
            "contracts": contracts,
            "strategy_label": "credit_put_spread_30_45",
            "thesis_id": thesis_id,
        },
        "tool_response": {
            "combo": True, "thesis_id": thesis_id,
            "strategy_label": "credit_put_spread_30_45",
            "net_credit": net_credit, "width": 5.0,
            "max_loss": (5.0 - net_credit) * 100 * contracts,
            "leg_quotes": leg_quotes,
            "combo_rows": [
                {"order_id": "L1", "code": LONG},
                {"order_id": "S1", "code": SHORT},
            ],
        },
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    return pf.main()


def _sync_entry(fake_pg, *, short_dealt, long_dealt):
    """Run the REAL _sync_combo_entry against the fake PG row."""
    from trading_agent.combo import combo_meta
    from trading_agent.graph.nodes import intraday_nodes as inode
    meta = combo_meta(fake_pg.row["broker_fill_json"]["combo"])
    assert meta is not None
    qty = fake_pg.row["qty"]
    order_map = {
        "L1": {"code": LONG, "order_status": "FILLED_ALL",
               "dealt_qty": qty, "dealt_avg_price": long_dealt},
        "S1": {"code": SHORT, "order_status": "FILLED_ALL",
               "dealt_qty": qty, "dealt_avg_price": short_dealt},
    }
    inode._sync_combo_entry(fake_pg.cursor, 1, SHORT, meta, order_map,
                            run_id="r", trigger="t")


def _pending_close_state(units, *, short_bid=0.50, short_ask=0.60,
                         long_bid=0.10, long_ask=0.16,
                         net_credit=0.70, settle_mode="full"):
    from trading_agent.exits.fill_confirm import build_pending_combo_state
    return build_pending_combo_state(
        symbol=SHORT, action="EXIT_TARGET", reason="pt", units=units,
        settle_mode=settle_mode, net_credit=net_credit,
        short_leg=SHORT, long_leg=LONG,
        short_order_id="BTC-1", long_order_id="STC-1",
        short_price=round(short_ask * 1.02, 2),
        long_price=round(long_bid * 0.98, 2),
        short_bid=short_bid, short_ask=short_ask,
        long_bid=long_bid, long_ask=long_ask,
        thesis_id=4, source="postgres", broker_qty_before=units,
    )


def _finalize_close(fake_pg, monkeypatch, state, *, short_dealt, long_dealt,
                    events):
    from trading_agent.exits import fill_confirm as fc
    units = state["qty"]
    order_map = {
        "BTC-1": {"order_status": "FILLED_ALL", "dealt_qty": units,
                  "dealt_avg_price": short_dealt},
        "STC-1": {"order_status": "FILLED_ALL", "dealt_qty": units,
                  "dealt_avg_price": long_dealt},
    }
    monkeypatch.setattr(fc, "emit",
                        lambda **kw: events.append(kw))
    monkeypatch.setattr(fc, "_notify_exit_filled", lambda *a, **k: None)
    monkeypatch.setattr("trading_agent.store.postgres.cursor",
                        fake_pg.cursor)
    fc._finalize_combo_close(1, state, order_map, "r", "t")


# ---------------------------------------------------------------------------
# The mandatory ROUND-TRIP test
# ---------------------------------------------------------------------------

def test_full_round_trip_is_cost_honest(combo_db, monkeypatch):
    """Open (posttool capture) → entry sync (honest clamp) → close finalize
    (honest clamp) → journal close. Asserts the audit's full identity:

      entry: dealt at the caller's MID limits (short 1.20, long 0.34;
             raw credit 0.86) vs touch bid 1.10 / ask 0.40 → honest
             credit = 1.10 − 0.40 = 0.70 (entry spread charge $32 on 2x);
      close: dealt at MID (short 0.55, long 0.13; raw debit 0.42) vs touch
             ask 0.60 / bid 0.10 → honest debit = 0.50 (exit spread $16);
      fees:  2 legs × 2 contracts × $1 per side, both directions = $8;
      P&L:   (0.70 − 0.50) × 100 × 2 − 8 = $32
           = frictionless mid P&L $88 − total modeled friction $56.
    """
    contracts = 2
    fake_pg = _FakePg()
    monkeypatch.setattr("trading_agent.store.postgres.cursor",
                        fake_pg.cursor)

    # -- open: REAL posttool capture path (SQLite twin + PG dual-write)
    assert _run_posttool(monkeypatch, contracts=contracts) == 0
    assert fake_pg.row is not None
    entry_fees = fake_pg.row["broker_fill_json"]["entry_fees"]
    assert entry_fees == pytest.approx(2.0 * contracts)    # 2 legs × $1
    assert fake_pg.row["broker_fill_json"]["combo"]["leg_quotes"] == ENTRY_QUOTES

    # -- entry fill sync: mid-priced limits "filled" at mid → clamp to touch
    _sync_entry(fake_pg, short_dealt=1.20, long_dealt=0.34)
    blob = fake_pg.row["broker_fill_json"]
    assert blob["entry_credit_raw"] == pytest.approx(0.86)
    assert blob["entry_credit_honest"] == pytest.approx(0.70)
    assert fake_pg.row["entry_price"] == pytest.approx(0.70)

    # -- close: both legs deal at mid → clamp to touch, settle the row
    events: list[dict] = []
    state = _pending_close_state(float(contracts))
    _finalize_close(fake_pg, monkeypatch, state,
                    short_dealt=0.55, long_dealt=0.13, events=events)

    assert fake_pg.close_params is not None, "row must journal-close"
    p = fake_pg.close_params
    net_debit_honest, outcome = float(p[0]), p[1]
    exit_fees, net_pnl = float(p[7]), float(p[8])
    assert net_debit_honest == pytest.approx(0.50)
    assert exit_fees == pytest.approx(2.0 * contracts)     # 2 legs × $1

    # Total modeled costs = entry fees ×2 legs + exit fees ×2 legs
    #                     + spread charges per the clamp model.
    entry_spread = (0.86 - 0.70) * 100 * contracts         # $32
    exit_spread = (0.50 - 0.42) * 100 * contracts          # $16
    total_costs = entry_fees + exit_fees + entry_spread + exit_spread
    assert total_costs == pytest.approx(56.0)
    frictionless_mid_pnl = (0.86 - 0.42) * 100 * contracts  # $88
    assert net_pnl == pytest.approx(frictionless_mid_pnl - total_costs)

    # Final journal P&L = (credit_honest − debit_honest) × 100 × n − fees.
    assert net_pnl == pytest.approx(
        (0.70 - 0.50) * 100 * contracts - entry_fees - exit_fees)
    assert net_pnl == pytest.approx(32.0)
    assert outcome == "WIN"
    assert fake_pg.row["outcome"] == "WIN"

    closed = next(e for e in events
                  if e.get("event_type") == "position_closed")
    assert closed["payload"]["exit_price"] == pytest.approx(0.50)
    assert closed["payload"]["net_debit_raw"] == pytest.approx(0.42)

    # -- the SQLite twin the posttool wrote settles with the SAME numbers
    from trading_agent.db import connection
    with connection() as conn:
        twin = conn.execute(
            "SELECT outcome, exit_price, pnl FROM trades "
            "WHERE symbol = ?", (SHORT,)).fetchone()
    assert twin["outcome"] == "WIN"
    assert twin["exit_price"] == pytest.approx(0.50)
    assert twin["pnl"] == pytest.approx(32.0)


# ---------------------------------------------------------------------------
# Clamp unit tests — entry side
# ---------------------------------------------------------------------------

def test_entry_clamp_mid_priced_fill_clamps_to_touch(combo_db):
    from trading_agent.graph.nodes.intraday_nodes import _honest_entry_leg
    # SOLD short at "mid" 1.20 but the bid was 1.10 → honest receipt 1.10
    assert _honest_entry_leg(1.20, 1.10, is_short=True,
                             underlying="SPY") == pytest.approx(1.10)
    # BOUGHT long at "mid" 0.34 but the ask was 0.40 → honest cost 0.40
    assert _honest_entry_leg(0.34, 0.40, is_short=False,
                             underlying="SPY") == pytest.approx(0.40)


def test_entry_clamp_worse_than_touch_is_noop(combo_db):
    """A dealt price WORSE than the touch is kept — the clamp only ever
    removes flattery, never adds it."""
    from trading_agent.graph.nodes.intraday_nodes import _honest_entry_leg
    assert _honest_entry_leg(1.05, 1.10, is_short=True,
                             underlying="SPY") == pytest.approx(1.05)
    assert _honest_entry_leg(0.45, 0.40, is_short=False,
                             underlying="SPY") == pytest.approx(0.45)


def test_entry_clamp_missing_quote_charges_calibrated_half_spread(combo_db):
    """No touch persisted → dealt ∓ half_spread_cost(dealt)/100 per
    contract-point (defaults: 4% of mark)."""
    from trading_agent.graph.nodes.intraday_nodes import _honest_entry_leg
    assert _honest_entry_leg(1.20, None, is_short=True,
                             underlying="SPY") == pytest.approx(1.20 - 0.048)
    assert _honest_entry_leg(0.34, None, is_short=False,
                             underlying="SPY") == pytest.approx(0.34 + 0.0136)


def test_entry_clamp_fallback_uses_per_underlying_calibration(combo_db):
    """The missing-quote fallback consults the M1-0.3 per-underlying
    calibrated spread (SPY 1% full → 0.5% half) instead of the global 4%."""
    from trading_agent import execution_costs as ec
    from trading_agent.graph.nodes.intraday_nodes import _honest_entry_leg
    (combo_db / "execution_costs.json").write_text(json.dumps({
        "per_underlying": {"SPY": {"median_spread_pct_mid": 0.01}},
    }))
    ec.reset_calibration_cache()
    assert _honest_entry_leg(1.20, None, is_short=True,
                             underlying="SPY") == pytest.approx(1.20 - 0.006)
    # Uncalibrated name → global 4% fallback
    assert _honest_entry_leg(1.20, None, is_short=True,
                             underlying="MRVL") == pytest.approx(1.20 - 0.048)


def test_sync_combo_entry_persists_raw_and_honest(combo_db, monkeypatch):
    """The journal row must carry BOTH the raw dealt credit and the honest
    clamped credit (entry_price = honest)."""
    fake_pg = _FakePg()
    monkeypatch.setattr("trading_agent.store.postgres.cursor",
                        fake_pg.cursor)
    assert _run_posttool(monkeypatch, contracts=1) == 0
    _sync_entry(fake_pg, short_dealt=1.20, long_dealt=0.34)
    blob = fake_pg.row["broker_fill_json"]
    assert blob["combo_fill_synced"] is True
    assert blob["entry_credit_raw"] == pytest.approx(0.86)
    assert blob["entry_credit_honest"] == pytest.approx(0.70)
    assert blob["avg_fill_price"] == pytest.approx(0.70)
    assert fake_pg.row["entry_price"] == pytest.approx(0.70)


def test_sync_combo_entry_missing_quotes_falls_back(combo_db, monkeypatch):
    """Payload without leg_quotes (pre-M1-0.2 row) → both legs charged the
    calibrated half-spread off their dealt prices."""
    fake_pg = _FakePg()
    monkeypatch.setattr("trading_agent.store.postgres.cursor",
                        fake_pg.cursor)
    assert _run_posttool(monkeypatch, contracts=1, leg_quotes=None) == 0
    _sync_entry(fake_pg, short_dealt=1.20, long_dealt=0.34)
    blob = fake_pg.row["broker_fill_json"]
    assert blob["entry_credit_raw"] == pytest.approx(0.86)
    # honest = (1.20 − 0.048) − (0.34 + 0.0136) = 0.7984
    assert blob["entry_credit_honest"] == pytest.approx(0.7984)
    assert fake_pg.row["entry_price"] == pytest.approx(0.7984)


# ---------------------------------------------------------------------------
# Clamp unit tests — close side
# ---------------------------------------------------------------------------

def test_close_clamp_marketable_limit_is_noop(combo_db):
    """Today's marketable close (BTC at ask×1.02, STC at bid×0.98) already
    crosses the spread — the clamp must not double-charge it."""
    from trading_agent.exits.fill_confirm import _honest_close_leg
    btc = {"quoted_bid": 0.50, "quoted_ask": 0.60, "symbol": SHORT}
    stc = {"quoted_bid": 0.10, "quoted_ask": 0.16, "symbol": LONG}
    assert _honest_close_leg(btc, 0.61, is_buy=True) == pytest.approx(0.61)
    assert _honest_close_leg(stc, 0.098, is_buy=False) == pytest.approx(0.098)


def test_close_clamp_better_than_touch_clamps(combo_db):
    """A SIMULATE fill better than the placement touch is clamped to it."""
    from trading_agent.exits.fill_confirm import _honest_close_leg
    btc = {"quoted_bid": 0.50, "quoted_ask": 0.60, "symbol": SHORT}
    stc = {"quoted_bid": 0.10, "quoted_ask": 0.16, "symbol": LONG}
    assert _honest_close_leg(btc, 0.55, is_buy=True) == pytest.approx(0.60)
    assert _honest_close_leg(stc, 0.13, is_buy=False) == pytest.approx(0.10)


def test_close_clamp_missing_quote_charges_half_spread(combo_db):
    """Close leg placed off the bare mark (no quote persisted) → charge the
    calibrated half-spread off the dealt price, per-underlying aware."""
    from trading_agent import execution_costs as ec
    from trading_agent.exits.fill_confirm import _honest_close_leg
    btc = {"quoted_bid": None, "quoted_ask": None, "symbol": SHORT}
    stc = {"quoted_bid": None, "quoted_ask": None, "symbol": LONG}
    assert _honest_close_leg(btc, 0.55, is_buy=True) == pytest.approx(
        0.55 + 0.55 * 0.04)
    assert _honest_close_leg(stc, 0.13, is_buy=False) == pytest.approx(
        0.13 - 0.13 * 0.04)
    # per-underlying calibration wins over the global figure
    (combo_db / "execution_costs.json").write_text(json.dumps({
        "per_underlying": {"SPY": {"median_spread_pct_mid": 0.01}},
    }))
    ec.reset_calibration_cache()
    assert _honest_close_leg(btc, 0.55, is_buy=True) == pytest.approx(
        0.55 + 0.55 * 0.005)


# ---------------------------------------------------------------------------
# _close_sqlite n_legs=1 hazard guard
# ---------------------------------------------------------------------------

def test_settle_refuses_sqlite_sourced_combo(monkeypatch):
    """_close_sqlite has no n_legs and would silently charge single-leg
    fees on a 2-leg close. Combos are only ever built source='postgres';
    a sqlite-sourced combo state must refuse to settle (fail closed)."""
    from trading_agent.exits import fill_confirm as fc

    def _boom(*a, **k):
        raise AssertionError("_close_sqlite must never see a combo state")
    monkeypatch.setattr(fc, "_close_sqlite", _boom)
    monkeypatch.setattr(fc, "_close_pg", _boom)
    state = {"combo": True, "source": "sqlite", "qty": 1.0}
    ok, net = fc._settle(9, state, 1.0, 0.5, n_legs=2)
    assert (ok, net) == (False, 0.0)


def test_settle_sqlite_single_leg_still_routes(monkeypatch):
    """Non-combo sqlite states keep their existing path."""
    from trading_agent.exits import fill_confirm as fc
    seen = []
    monkeypatch.setattr(
        fc, "_close_sqlite",
        lambda tid, state, price, qty, unconfirmed=False:
            (seen.append((tid, price, qty)) or (True, 5.0)))
    state = {"source": "sqlite", "qty": 1.0}
    ok, net = fc._settle(9, state, 1.0, 0.5)
    assert ok and net == 5.0
    assert seen == [(9, 0.5, 1.0)]
