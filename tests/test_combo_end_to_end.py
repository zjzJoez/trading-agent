"""End-to-end combo exit simulation (M1-0.1 acceptance):

simulated Postgres combo row (+canonical payload) → refresh (stubbed quotes
for BOTH legs) → detect (net-of-legs mark, trigger fires) → route (atomic
two-leg close, short first) → finalize (both legs dealt → journal close).

Asserts both legs close atomically through the combo tool (never the
single-leg path) and that journal P&L equals the net-of-legs arithmetic
minus modeled costs (2-leg fees each way).
"""
from __future__ import annotations

from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest


def _syms(days=40):
    exp = (datetime.now(UTC).date()
           + timedelta(days=days)).strftime("%y%m%d")
    return f"US.SPY{exp}P{100_000:08d}", f"US.SPY{exp}P{95_000:08d}"


def _base_state(**overrides) -> dict:
    s = {
        "run_id": "e2e-combo", "trigger": "intraday_monitor",
        "ts": "2026-07-20T00:00:00Z",
        "budget": {"run_usd_cap": 1.0, "spent_usd": 0.0, "model_calls": 0},
        "watchlist": [], "market_data": {}, "candidates": [], "research": {},
        "sizing": {}, "order": {}, "fill": {}, "journal": {}, "learning": {},
        "notifications": [], "errors": [],
    }
    s.update(overrides)
    return s


class _Sim:
    """One simulated combo trade wired through the three intraday nodes."""

    def __init__(self, *, days=40, net_credit=1.2, width=5.0, qty=2,
                 short_quote=None, long_quote=None, downsize_labels=None,
                 regime_label="RANGE_LOW_VOL"):
        from trading_agent.llm.schemas import combo_exit_plan
        self.short_sym, self.long_sym = _syms(days)
        self.qty = qty
        self.net_credit = net_credit
        self.regime_label = regime_label
        self.short_quote = short_quote or {
            "last_price": 0.62, "bid_price": 0.60, "ask_price": 0.65}
        self.long_quote = long_quote or {
            "last_price": 0.22, "bid_price": 0.20, "ask_price": 0.24}
        plan = combo_exit_plan(net_credit, width).model_dump()
        if downsize_labels:
            plan["regime_rules"]["downsize_50_on_labels"] = downsize_labels
        self.payload = {
            "combo": True, "short_leg": self.short_sym,
            "long_leg": self.long_sym, "net_credit": net_credit,
            "width": width, "max_loss": (width - net_credit) * 100 * qty,
            "contracts": float(qty), "broker_order_ids": ["L1", "S1"],
        }
        self.journal_row = {
            "trade_id": 15, "symbol": self.short_sym, "asset_type": "OPT",
            "side": "SELL", "qty": float(qty), "entry_price": net_credit,
            "stop": None, "target": None, "exit_plan": plan,
            "mfe_so_far": None, "scale_rungs_taken": 0,
            "regime_downsized_at_label": None, "opened_at": None,
            "thesis_id": 4, "strategy_label": "credit_put_spread_30_45",
            "combo": self.payload, "direction": "SHORT",
            "thesis_text": "t", "invalidation": "i", "thesis_status": "open",
        }
        self.broker_positions = [
            {"symbol": self.short_sym, "asset_type": "OPT", "qty": qty,
             "entry_price": 2.05, "mark": 0.9},
            {"symbol": self.long_sym, "asset_type": "OPT", "qty": qty,
             "entry_price": 0.85, "mark": 0.3},
        ]
        self.close_calls: list[dict] = []
        self.single_leg_orders: list[dict] = []
        self.pending_writes: list[tuple] = []
        self.events: list[dict] = []
        self.pending_ids: set[int] = set()

    def _fake_close_tool(self, **kw):
        self.close_calls.append(kw)
        return {"combo_close": True, "close_rows": [],
                "short_close_order_id": "BTC-1",
                "long_close_order_id": "STC-1"}

    def run_intraday_tick(self):
        """refresh → detect → route with everything external stubbed."""
        state = _base_state(
            positions=[dict(p) for p in self.broker_positions],
            regime={"label": self.regime_label, "confidence": 0.9, "gate": {}},
        )
        quote_rows = [
            dict(self.short_quote, code=self.short_sym),
            dict(self.long_quote, code=self.long_sym),
        ]
        pg_cursor = MagicMock()
        pg_cursor.__enter__ = lambda s: pg_cursor
        pg_cursor.__exit__ = MagicMock(return_value=False)
        pg_cursor.fetchall.return_value = []
        self.pg_cursor = pg_cursor
        with ExitStack() as stack:
            stack.enter_context(patch(
                "trading_agent.graph.nodes.intraday_nodes.emit",
                side_effect=lambda **kw: self.events.append(kw)))
            stack.enter_context(patch(
                "trading_agent.events.emit", return_value=1))
            stack.enter_context(patch(
                "trading_agent.mcp_servers.moomoo.server.get_quote",
                return_value={"rows": quote_rows}))
            stack.enter_context(patch(
                "trading_agent.mcp_servers.moomoo.server.get_orders",
                return_value={"rows": []}))
            stack.enter_context(patch(
                "trading_agent.store.postgres.get_open_journal_trades",
                return_value=[self.journal_row]))
            stack.enter_context(patch(
                "trading_agent.store.postgres.cursor",
                return_value=pg_cursor))
            stack.enter_context(patch(
                "trading_agent.exits.fill_confirm.pending_close_trade_ids_pg",
                side_effect=lambda: set(self.pending_ids)))
            stack.enter_context(patch(
                "trading_agent.graph.nodes.intraday_nodes._alerted_today",
                return_value=False))
            stack.enter_context(patch(
                "trading_agent.mcp_servers.moomoo.server."
                "close_paper_option_combo",
                side_effect=self._fake_close_tool))
            stack.enter_context(patch(
                "trading_agent.mcp_servers.moomoo.server."
                "place_paper_option_order",
                side_effect=lambda **kw: self.single_leg_orders.append(kw)))
            stack.enter_context(patch(
                "trading_agent.mcp_servers.moomoo.server.place_paper_order",
                side_effect=lambda **kw: self.single_leg_orders.append(kw)))
            stack.enter_context(patch(
                "trading_agent.exits.fill_confirm.write_pending_close",
                side_effect=lambda tid, st: (
                    self.pending_writes.append((tid, st)) or True)))
            stack.enter_context(patch("trading_agent.notify.send"))

            from trading_agent.graph.nodes.intraday_nodes import (
                detect_exit_triggers,
                refresh_quotes_and_greeks,
                route_exit_or_hold,
            )
            r1 = refresh_quotes_and_greeks(state)
            state.update(r1)
            r2 = detect_exit_triggers(state)
            state.update(r2)
            route_exit_or_hold(state)
        self.decisions = r2["journal"]["exit_decisions"]
        return state

    def finalize(self, *, short_dealt, long_dealt, entry_fees=None):
        """Run the fill-confirm pass against both close orders FILLED_ALL.

        Returns the captured journal-close UPDATE params (or None)."""
        assert self.pending_writes, "route must have written a pending state"
        trade_id, pending = self.pending_writes[-1]
        units = pending["qty"]
        if entry_fees is None:
            entry_fees = units * 1.0 * 2  # 2-leg entry fees at $1/contract
        order_map = {
            "BTC-1": {"order_status": "FILLED_ALL", "dealt_qty": units,
                      "dealt_avg_price": short_dealt},
            "STC-1": {"order_status": "FILLED_ALL", "dealt_qty": units,
                      "dealt_avg_price": long_dealt},
        }
        pg_cursor = MagicMock()
        pg_cursor.__enter__ = lambda s: pg_cursor
        pg_cursor.__exit__ = MagicMock(return_value=False)
        pg_cursor.fetchone.return_value = (
            self.net_credit, self.qty, "OPT", "SELL", 4,
            {"entry_fees": entry_fees})
        with ExitStack() as stack:
            stack.enter_context(patch(
                "trading_agent.exits.fill_confirm.emit",
                side_effect=lambda **kw: self.events.append(kw)))
            stack.enter_context(patch(
                "trading_agent.store.postgres.cursor",
                return_value=pg_cursor))
            stack.enter_context(patch("trading_agent.notify.send"))
            from trading_agent.exits.fill_confirm import _finalize_combo_close
            _finalize_combo_close(trade_id, pending, order_map, "r", "t")
        closes = [c for c in pg_cursor.execute.call_args_list
                  if c.args and "SET exit_price" in c.args[0]]
        return closes[0].args[1] if closes else None


def test_e2e_50pct_profit_take_closes_atomically_with_net_of_legs_pnl():
    # spread value = 0.62 − 0.22 = 0.40 <= 0.60 = 0.5 × 1.2 → PT fires
    sim = _Sim()
    sim.run_intraday_tick()
    assert sim.decisions[0]["action"] == "EXIT_TARGET"
    assert sim.decisions[0]["close_units"] == 2

    # atomic close, short leg first, per-leg marketable limits
    assert len(sim.close_calls) == 1
    call = sim.close_calls[0]
    assert call["short_leg_symbol"] == sim.short_sym
    assert call["long_leg_symbol"] == sim.long_sym
    assert call["contracts"] == 2
    assert call["short_price"] == pytest.approx(round(0.65 * 1.02, 2))
    assert call["long_price"] == pytest.approx(round(0.20 * 0.98, 2))
    assert sim.single_leg_orders == []  # never the single-leg path

    # both legs deal → settle: net_debit = 0.64 − 0.21 = 0.43
    params = sim.finalize(short_dealt=0.64, long_dealt=0.21)
    assert params is not None, "full close must journal once both legs deal"
    dealt_price, outcome = params[0], params[1]
    net = params[8]
    assert dealt_price == pytest.approx(0.43)
    # gross (1.2 − 0.43) × 2 × 100 = 154; fees 4 entry + 4 exit (2 legs × 2
    # contracts × $1 each way) → 146
    assert net == pytest.approx(146.0)
    assert outcome == "WIN"


def test_e2e_21dte_force_close_fires_on_short_leg_dte():
    # 15 DTE, spread 0.70 (no PT) → EXIT_DTE_HARD full close
    sim = _Sim(days=15,
               short_quote={"last_price": 0.90, "bid_price": 0.88,
                            "ask_price": 0.93},
               long_quote={"last_price": 0.20, "bid_price": 0.18,
                           "ask_price": 0.22})
    sim.run_intraday_tick()
    assert sim.decisions[0]["action"] == "EXIT_DTE_HARD"
    assert len(sim.close_calls) == 1
    assert sim.close_calls[0]["contracts"] == 2
    assert sim.single_leg_orders == []
    # settle at net_debit 0.95 − 0.15 = 0.80
    params = sim.finalize(short_dealt=0.95, long_dealt=0.15)
    net = params[8]
    # gross (1.2 − 0.80) × 2 × 100 = 80; fees 4 + 4 → 72
    assert net == pytest.approx(72.0)
    assert params[1] == "WIN"


def test_e2e_p0b_regime_trim_closes_one_whole_unit_and_stays_open():
    sim = _Sim(downsize_labels=["VOLATILE_TRANSITION"],
               regime_label="VOLATILE_TRANSITION",
               short_quote={"last_price": 0.90, "bid_price": 0.88,
                            "ask_price": 0.93},
               long_quote={"last_price": 0.20, "bid_price": 0.18,
                           "ask_price": 0.22})
    sim.run_intraday_tick()
    dec = sim.decisions[0]
    assert dec["action"] == "EXIT_REGIME_DOWNSIZE"
    assert dec["close_units"] == 1              # floor(2 × 0.5), whole units
    assert len(sim.close_calls) == 1
    assert sim.close_calls[0]["contracts"] == 1
    assert sim.single_leg_orders == []
    _tid, pending = sim.pending_writes[-1]
    assert pending["settle_mode"] == "trim"
    # trim stamped the downsize guard immediately
    stamps = [c for c in sim.pg_cursor.execute.call_args_list
              if c.args and "regime_downsized_at_label" in c.args[0]]
    assert len(stamps) == 1

    # both legs deal → trim completes: row stays OPEN, event-only P&L
    params = sim.finalize(short_dealt=0.95, long_dealt=0.15)
    assert params is None                       # NO journal close
    trimmed = [e for e in sim.events
               if e.get("event_type") == "combo_trimmed"]
    assert len(trimmed) == 1
    p = trimmed[0]["payload"]
    assert p["net_debit"] == pytest.approx(0.80)
    # realized = (1.2 − 0.80) × 1 × 100 − 2-leg exit fees ($2) = 38
    assert p["realized"] == pytest.approx(38.0)


def test_e2e_p0b_trim_qty1_skips_and_stamps_without_any_order():
    sim = _Sim(qty=1, downsize_labels=["VOLATILE_TRANSITION"],
               regime_label="VOLATILE_TRANSITION",
               short_quote={"last_price": 0.90, "bid_price": 0.88,
                            "ask_price": 0.93},
               long_quote={"last_price": 0.20, "bid_price": 0.18,
                           "ask_price": 0.22})
    sim.broker_positions[0]["qty"] = 1
    sim.broker_positions[1]["qty"] = 1
    sim.journal_row["qty"] = 1.0
    sim.run_intraday_tick()
    dec = sim.decisions[0]
    assert dec["action"] == "HOLD"
    assert dec["stamp_regime_label"] == "VOLATILE_TRANSITION"
    assert sim.close_calls == []                # no order of any kind
    assert sim.single_leg_orders == []
    assert sim.pending_writes == []
    stamps = [c for c in sim.pg_cursor.execute.call_args_list
              if c.args and "regime_downsized_at_label" in c.args[0]]
    assert len(stamps) == 1
    skipped = [e for e in sim.events
               if e.get("event_type") == "combo_trim_skipped_single_unit"]
    assert len(skipped) == 1


def test_excursions_mark_combo_net_of_legs():
    """MAE/MFE for a combo row must be computed against the SPREAD mid
    (short − long), in units of the net credit — never one leg's price."""
    from trading_agent.learning import excursion as exc

    short_sym, long_sym = _syms(40)
    payload = {
        "combo": True, "short_leg": short_sym, "long_leg": long_sym,
        "net_credit": 1.2, "width": 5.0, "max_loss": 760.0,
        "contracts": 2.0, "broker_order_ids": ["L1", "S1"],
    }
    pg_cursor = MagicMock()
    pg_cursor.__enter__ = lambda s: pg_cursor
    pg_cursor.__exit__ = MagicMock(return_value=False)
    # id, symbol, entry, stop, mae, mfe, combo
    pg_cursor.fetchall.return_value = [
        (15, short_sym, 1.2, None, None, None, payload)]

    mids = {short_sym: 0.62, long_sym: 0.22}
    with patch.object(exc, "cursor", return_value=pg_cursor), \
         patch.object(exc, "_get_latest_mid",
                      side_effect=lambda s: mids.get(s)):
        updates = exc.update_excursions_once()
    assert len(updates) == 1
    u = updates[0]
    # spread mid = 0.40; R_unit = 5% × 1.2 = 0.06 (implicit stop);
    # SHORT polarity: realized_R = (1.2 − 0.40)/0.06 — the spread decaying
    # BELOW the net credit is favorable, so it must record as positive MFE.
    assert u.realized_R_now == pytest.approx((1.2 - 0.40) / 0.06)
    assert u.mfe_so_far == pytest.approx((1.2 - 0.40) / 0.06)
    write = [c for c in pg_cursor.execute.call_args_list
             if c.args and "SET mae_so_far" in c.args[0]]
    assert len(write) == 1
    assert write[0].args[1][3] == pytest.approx(0.40)   # last_quote_price


def test_degraded_combo_payload_holds_and_places_nothing():
    """M1-0.1 degrade contract: a PG combo row whose combo payload is
    present-but-unparseable must HOLD + alert — NEVER fall through to
    single-leg management (which would BUY back the short leg alone and
    orphan the wing)."""
    sim = _Sim()
    sim.journal_row["combo"] = {"combo": True}   # legs/net_credit stripped
    sim.run_intraday_tick()
    short_decs = [d for d in sim.decisions if d["symbol"] == sim.short_sym]
    assert len(short_decs) == 1
    assert short_decs[0]["action"] == "HOLD"
    assert short_decs[0]["reason"] == "combo_payload_unparseable"
    # nothing reached the broker via ANY path
    assert sim.close_calls == []
    assert sim.single_leg_orders == []
    assert sim.pending_writes == []
    alerts = [e for e in sim.events
              if e.get("event_type") == "combo_payload_unparseable"]
    assert len(alerts) == 1 and alerts[0]["severity"] == 2


def test_route_backstop_refuses_single_leg_short_close():
    """Even with a fabricated EXIT decision, route must refuse a single-leg
    close on a SHORT-direction plan with no combo meta (degraded combo)."""
    from trading_agent.graph.nodes.intraday_nodes import route_exit_or_hold
    short_sym, _long_sym = _syms(40)
    events: list[dict] = []
    placed: list[dict] = []
    state = _base_state(
        positions=[{"symbol": short_sym, "asset_type": "OPT", "qty": 2,
                    "entry_price": 1.2, "mark": 0.9, "bid": 0.85,
                    "ask": 0.95,
                    "exit_plan": {"direction": "SHORT"}}],
        journal={"exit_decisions": [
            {"symbol": short_sym, "action": "EXIT_STOP",
             "exit_qty_factor": 1.0, "reason": "fabricated"}]},
    )
    with ExitStack() as stack:
        stack.enter_context(patch(
            "trading_agent.graph.nodes.intraday_nodes.emit",
            side_effect=lambda **kw: events.append(kw)))
        stack.enter_context(patch(
            "trading_agent.mcp_servers.moomoo.server.place_paper_option_order",
            side_effect=lambda **kw: placed.append(kw)))
        stack.enter_context(patch(
            "trading_agent.mcp_servers.moomoo.server.place_paper_order",
            side_effect=lambda **kw: placed.append(kw)))
        stack.enter_context(patch(
            "trading_agent.mcp_servers.moomoo.server.get_orders",
            return_value={"rows": []}))
        stack.enter_context(patch(
            "trading_agent.store.postgres.get_open_journal_trades",
            return_value=[{"symbol": short_sym, "trade_id": 15,
                           "thesis_id": 4}]))
        stack.enter_context(patch(
            "trading_agent.exits.fill_confirm.pending_close_trade_ids_pg",
            return_value=set()))
        stack.enter_context(patch("trading_agent.notify.send"))
        route_exit_or_hold(state)
    assert placed == [], "single-leg close on a degraded combo row placed!"
    refused = [e for e in events
               if e.get("event_type") == "combo_single_leg_close_refused"]
    assert len(refused) == 1 and refused[0]["severity"] == 2


def test_combo_qty_mismatch_during_pending_close_holds_quietly():
    """Leg qtys diverging while a combo close is in flight is a NORMAL
    pending state (trim legs filling across a tick boundary) — HOLD with
    no sev-2 page."""
    sim = _Sim()
    sim.pending_ids = {15}
    sim.broker_positions[0]["qty"] = 1      # short trimmed 2→1
    sim.broker_positions[1]["qty"] = 2      # long STC still working
    sim.run_intraday_tick()
    dec = next(d for d in sim.decisions if d["symbol"] == sim.short_sym)
    assert dec["action"] == "HOLD"
    assert dec["reason"] == "combo_close_in_flight"
    mismatches = [e for e in sim.events
                  if e.get("event_type") == "combo_leg_qty_mismatch"]
    assert mismatches == []
    # route also placed nothing (trade already pending)
    assert sim.close_calls == []


def test_combo_qty_mismatch_without_pending_close_still_pages():
    sim = _Sim()
    sim.broker_positions[0]["qty"] = 1
    sim.run_intraday_tick()
    dec = next(d for d in sim.decisions if d["symbol"] == sim.short_sym)
    assert dec["action"] == "HOLD"
    assert dec["reason"] == "combo_leg_qty_mismatch"
    mismatches = [e for e in sim.events
                  if e.get("event_type") == "combo_leg_qty_mismatch"]
    assert len(mismatches) == 1 and mismatches[0]["severity"] == 2


def test_residual_wing_during_pending_close_no_manual_alert():
    """Short leg closed before the wing: the residual wing surfacing alone
    at the broker mid-close must not fire the manual-trade alert."""
    sim = _Sim()
    sim.pending_ids = {15}
    sim.broker_positions = [sim.broker_positions[1]]   # only the wing left
    sim.run_intraday_tick()
    wing_dec = next(d for d in sim.decisions
                    if d["symbol"] == sim.long_sym)
    assert wing_dec["action"] == "HOLD"
    assert wing_dec["reason"] == "combo_close_in_flight_residual_leg"
    no_plan = [e for e in sim.events
               if e.get("event_type") == "position_no_exit_plan"]
    assert no_plan == []


def test_eod_mark_to_market_marks_combo_net_of_legs_short_polarity():
    """EOD marks: a combo row is marked short_last − long_last with SHORT
    polarity — the short leg's price alone with LONG polarity reports a
    healthy credit spread as a loss (sign-inverted AND wing-blind)."""
    from trading_agent.graph.nodes import eod_nodes
    short_sym, long_sym = _syms(40)
    trade = {
        "symbol": short_sym, "entry_price": 1.2, "qty": 2.0, "side": "SELL",
        "stop": None, "target": None, "thesis_id": 4,
        "strategy_label": "credit_put_spread_30_45",
        "combo": {"combo": True, "short_leg": short_sym,
                  "long_leg": long_sym, "net_credit": 1.2, "width": 5.0,
                  "max_loss": 760.0, "contracts": 2.0,
                  "broker_order_ids": ["L1", "S1"]},
    }
    events: list[dict] = []
    state = _base_state(journal={"open_trades": [trade]})
    with ExitStack() as stack:
        stack.enter_context(patch(
            "trading_agent.graph.nodes.eod_nodes.emit",
            side_effect=lambda **kw: events.append(kw)))
        stack.enter_context(patch(
            "trading_agent.mcp_servers.moomoo.server.get_quote",
            return_value={"rows": [
                {"code": short_sym, "last_price": 0.62},
                {"code": long_sym, "last_price": 0.22},
            ]}))
        out = eod_nodes.mark_to_market(state)
    marked = out["journal"]["marked_positions"]
    assert len(marked) == 1
    assert marked[0]["combo"] is True
    assert marked[0]["mark"] == pytest.approx(0.40)
    # SHORT polarity: (1.2 − 0.40) × 2 × 100 = +160 (the spread DECAYING is
    # the win) — the old single-leg math reported (0.62−1.2)×2×100 = −116.
    assert marked[0]["unrealized_pnl"] == pytest.approx(160.0)
    assert out["journal"]["total_unrealized_pnl"] == pytest.approx(160.0)


def test_eod_mark_to_market_skips_combo_with_unquotable_leg():
    from trading_agent.graph.nodes import eod_nodes
    short_sym, long_sym = _syms(40)
    trade = {
        "symbol": short_sym, "entry_price": 1.2, "qty": 2.0, "side": "SELL",
        "combo": {"combo": True, "short_leg": short_sym,
                  "long_leg": long_sym, "net_credit": 1.2,
                  "contracts": 2.0},
    }
    events: list[dict] = []
    state = _base_state(journal={"open_trades": [trade]})
    with ExitStack() as stack:
        stack.enter_context(patch(
            "trading_agent.graph.nodes.eod_nodes.emit",
            side_effect=lambda **kw: events.append(kw)))
        stack.enter_context(patch(
            "trading_agent.mcp_servers.moomoo.server.get_quote",
            return_value={"rows": [
                {"code": short_sym, "last_price": 0.62},   # wing unquoted
            ]}))
        out = eod_nodes.mark_to_market(state)
    assert out["journal"]["marked_positions"] == []
    assert out["journal"]["total_unrealized_pnl"] == 0.0
    computed = next(e for e in events
                    if e.get("event_type") == "marks_computed")
    assert computed["payload"]["combo_skipped_unquotable"] == 1


def test_excursions_skip_combo_when_leg_unquotable():
    from trading_agent.learning import excursion as exc

    short_sym, long_sym = _syms(40)
    payload = {
        "combo": True, "short_leg": short_sym, "long_leg": long_sym,
        "net_credit": 1.2, "width": 5.0, "max_loss": 760.0,
        "contracts": 2.0, "broker_order_ids": ["L1", "S1"],
    }
    pg_cursor = MagicMock()
    pg_cursor.__enter__ = lambda s: pg_cursor
    pg_cursor.__exit__ = MagicMock(return_value=False)
    pg_cursor.fetchall.return_value = [
        (15, short_sym, 1.2, None, None, None, payload)]

    mids = {short_sym: 0.62}  # long leg unquotable
    with patch.object(exc, "cursor", return_value=pg_cursor), \
         patch.object(exc, "_get_latest_mid",
                      side_effect=lambda s: mids.get(s)):
        updates = exc.update_excursions_once()
    assert updates == []
    writes = [c for c in pg_cursor.execute.call_args_list
              if c.args and "SET mae_so_far" in c.args[0]]
    assert writes == []
