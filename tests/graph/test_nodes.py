"""Behavioral unit tests for the new Phase 2.5+ graph nodes.

All tests use mocks for external I/O (Postgres, moomoo, LLM router, ntfy)
so they run without any live infrastructure.

Coverage:
  - health_nodes: opend_health, postgres_health, ntfy_health
  - intraday_nodes: refresh_quotes_and_greeks, detect_exit_triggers, route_exit_or_hold
  - eod_nodes: reconcile_journal, mark_to_market, persist_daily_marks
  - premarket_nodes: collect_watchlist_data, rank_candidates, ntfy_scan_digest
  - trade_nodes: persist_veto, ntfy_risk_block, persist_defer, ntfy_defer
  - trade_nodes: regime_execution_gate soak gate, deterministic_sizing soak cap
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> dict:
    s = {
        "run_id": "test-run-001",
        "trigger": "healthcheck",
        "ts": "2026-05-08T00:00:00Z",
        "budget": {"run_usd_cap": 1.0, "spent_usd": 0.0, "model_calls": 0},
        "watchlist": [],
        "market_data": {},
        "candidates": [],
        "research": {},
        "sizing": {},
        "order": {},
        "fill": {},
        "journal": {},
        "learning": {},
        "notifications": [],
        "errors": [],
    }
    s.update(overrides)
    return s


def _patch_emit(module: str = "trading_agent.events"):
    """Patch emit in a specific module (avoids local-binding issues)."""
    return patch(f"{module}.emit", return_value=1)


def _patch_all_emits():
    """Patch emit across all node modules simultaneously."""
    from contextlib import ExitStack
    modules = [
        "trading_agent.graph.nodes.trade_nodes",
        "trading_agent.graph.nodes.intraday_nodes",
        "trading_agent.graph.nodes.eod_nodes",
        "trading_agent.graph.nodes.premarket_nodes",
        "trading_agent.graph.nodes.health_nodes",
        "trading_agent.events",
    ]
    class _MultiPatch:
        def __enter__(self):
            self._stack = ExitStack()
            for m in modules:
                self._stack.enter_context(patch(f"{m}.emit", return_value=1))
            return self
        def __exit__(self, *a):
            return self._stack.__exit__(*a)
    return _MultiPatch()


# ---------------------------------------------------------------------------
# health_nodes
# ---------------------------------------------------------------------------

class TestOpendHealth:
    def test_healthy_emits_ok(self):
        state = _base_state()
        with _patch_all_emits():
            with patch("trading_agent.mcp_servers.moomoo.server.get_account_info",
                       return_value={"rows": [{"total_assets": 100_000}]}):
                from trading_agent.graph.nodes.health_nodes import opend_health
                result = opend_health(state)
        assert result == {}

    def test_unhealthy_emits_alert(self):
        state = _base_state()
        with _patch_all_emits():
            with patch("trading_agent.mcp_servers.moomoo.server.get_account_info",
                       side_effect=ConnectionError("timeout")):
                with patch("trading_agent.graph.nodes.health_nodes._alert_ops") as mock_alert:
                    from trading_agent.graph.nodes.health_nodes import opend_health
                    opend_health(state)
                    mock_alert.assert_called_once()


class TestPostgresHealth:
    def test_healthy(self):
        state = _base_state()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = (1,)
        with _patch_all_emits():
            with patch("trading_agent.store.postgres.cursor", return_value=mock_cursor):
                from trading_agent.graph.nodes.health_nodes import postgres_health
                result = postgres_health(state)
        assert result == {}

    def test_unreachable_alerts(self):
        state = _base_state()
        with _patch_all_emits():
            with patch("trading_agent.store.postgres.cursor", side_effect=Exception("conn refused")):
                with patch("trading_agent.graph.nodes.health_nodes._alert_ops") as mock_alert:
                    from trading_agent.graph.nodes.health_nodes import postgres_health
                    postgres_health(state)
                    mock_alert.assert_called_once()


class TestNtfyHealth:
    def test_sends_heartbeat(self):
        state = _base_state()
        with _patch_all_emits():
            with patch("trading_agent.notify.send") as mock_send:
                from trading_agent.graph.nodes.health_nodes import ntfy_health
                ntfy_health(state)
                mock_send.assert_called_once()
                call_kwargs = mock_send.call_args[1] if mock_send.call_args[1] else {}
                call_args = mock_send.call_args[0] if mock_send.call_args[0] else []
                topic = call_kwargs.get("topic") or (call_args[0] if call_args else None)
                assert topic == "ops"


# ---------------------------------------------------------------------------
# intraday_nodes
# ---------------------------------------------------------------------------

class TestRefreshQuotesAndGreeks:
    def _make_pos(self, symbol="US.AAPL", asset_type="STK"):
        return {
            "symbol": symbol, "asset_type": asset_type,
            "underlying": "US.AAPL", "qty": 100,
            "entry_price": 200.0, "mark": 200.0,
        }

    def test_stock_quote_refreshed(self):
        state = _base_state(
            trigger="intraday_monitor",
            positions=[self._make_pos("US.AAPL", "STK")],
        )
        with _patch_all_emits():
            with patch("trading_agent.mcp_servers.moomoo.server.get_quote",
                       return_value={"rows": [{"code": "US.AAPL", "last_price": 210.0}]}):
                from trading_agent.graph.nodes.intraday_nodes import refresh_quotes_and_greeks
                result = refresh_quotes_and_greeks(state)
        assert result["positions"][0]["mark"] == 210.0

    def test_option_greeks_refreshed(self):
        opt_sym = "US.AAPL260117C00200000"
        state = _base_state(
            trigger="intraday_monitor",
            positions=[self._make_pos(opt_sym, "OPT")],
        )
        with _patch_all_emits():
            with patch("trading_agent.mcp_servers.moomoo.server.get_quote",
                       return_value={"rows": [{
                           "code": opt_sym, "last_price": 5.5,
                           "imp_volatility": 0.35,
                           "delta": 0.45, "gamma": 0.02,
                           "vega": 0.1, "theta": -0.05,
                       }]}):
                from trading_agent.graph.nodes.intraday_nodes import refresh_quotes_and_greeks
                result = refresh_quotes_and_greeks(state)
        pos = result["positions"][0]
        assert pos["mark"] == 5.5
        assert pos["iv"] == 0.35
        assert pos["delta"] == 0.45

    def test_empty_positions_returns_empty(self):
        state = _base_state(trigger="intraday_monitor", positions=[])
        with _patch_all_emits():
            from trading_agent.graph.nodes.intraday_nodes import refresh_quotes_and_greeks
            result = refresh_quotes_and_greeks(state)
        assert result == {}


class TestDetectExitTriggers:
    def _make_pos(self):
        return {
            "symbol": "US.SPY260117C00700000",
            "asset_type": "OPT", "qty": 2,
            "entry_price": 3.0, "mark": 1.5,
            "stop": 1.0, "target": 6.0, "age_minutes": 120,
        }

    def test_hold_decision(self):
        from trading_agent.llm.schemas import ExitMonitorOutput
        state = _base_state(
            trigger="intraday_monitor",
            positions=[self._make_pos()],
            regime={"label": "BULL_TREND", "confidence": 0.9, "gate": {}},
        )
        mock_res = MagicMock()
        mock_res.parsed = ExitMonitorOutput(action="HOLD", exit_qty_factor=0.0, reason="thesis intact")
        with _patch_all_emits():
            with patch("trading_agent.mcp_servers.journal.server.get_open_positions_with_thesis",
                       return_value={"rows": []}):
                with patch("trading_agent.llm.get_router") as mock_router:
                    mock_router.return_value.call.return_value = mock_res
                    from trading_agent.graph.nodes.intraday_nodes import detect_exit_triggers
                    result = detect_exit_triggers(state)
        decisions = result["journal"]["exit_decisions"]
        assert len(decisions) == 1
        assert decisions[0]["action"] == "HOLD"

    def test_exit_stop_decision(self):
        from trading_agent.llm.schemas import ExitMonitorOutput
        state = _base_state(
            trigger="intraday_monitor",
            positions=[self._make_pos()],
            regime={"label": "BEAR_TREND", "confidence": 0.8, "gate": {}},
        )
        mock_res = MagicMock()
        mock_res.parsed = ExitMonitorOutput(action="EXIT_STOP", exit_qty_factor=1.0, reason="stop hit")
        with _patch_all_emits():
            with patch("trading_agent.mcp_servers.journal.server.get_open_positions_with_thesis",
                       return_value={"rows": []}):
                with patch("trading_agent.llm.get_router") as mock_router:
                    mock_router.return_value.call.return_value = mock_res
                    from trading_agent.graph.nodes.intraday_nodes import detect_exit_triggers
                    result = detect_exit_triggers(state)
        decisions = result["journal"]["exit_decisions"]
        assert decisions[0]["action"] == "EXIT_STOP"

    def test_llm_failure_defaults_to_hold(self):
        state = _base_state(
            trigger="intraday_monitor",
            positions=[self._make_pos()],
            regime={"label": "BULL_TREND", "confidence": 0.9, "gate": {}},
        )
        with _patch_all_emits():
            with patch("trading_agent.mcp_servers.journal.server.get_open_positions_with_thesis",
                       return_value={"rows": []}):
                with patch("trading_agent.llm.get_router") as mock_router:
                    mock_router.return_value.call.side_effect = RuntimeError("LLM timeout")
                    from trading_agent.graph.nodes.intraday_nodes import detect_exit_triggers
                    result = detect_exit_triggers(state)
        decisions = result["journal"]["exit_decisions"]
        assert decisions[0]["action"] == "HOLD"


class TestRouteExitOrHold:
    def test_all_hold_returns_empty(self):
        state = _base_state(
            trigger="intraday_monitor",
            journal={"exit_decisions": [
                {"symbol": "US.AAPL", "action": "HOLD", "exit_qty_factor": 0.0, "reason": "ok"},
            ]},
        )
        with _patch_all_emits():
            from trading_agent.graph.nodes.intraday_nodes import route_exit_or_hold
            result = route_exit_or_hold(state)
        assert result == {}

    def test_exit_stock_places_normal_order(self):
        state = _base_state(
            trigger="intraday_monitor",
            journal={"exit_decisions": [
                {"symbol": "US.AAPL", "action": "EXIT_STOP", "exit_qty_factor": 1.0, "reason": "stop"},
            ]},
            positions=[{
                "symbol": "US.AAPL", "asset_type": "STK",
                "qty": 10, "entry_price": 200.0, "mark": 180.0,
                "thesis_id": 42, "strategy_label": "momentum_long",
            }],
        )
        mock_order = {"thesis_id": 7, "rows": [{"order_id": "ORD123", "order_status": "SUBMITTED"}]}
        with _patch_all_emits():
            with patch("trading_agent.mcp_servers.journal.server.get_open_positions_with_thesis",
                       return_value={"rows": [{"symbol": "US.AAPL", "trade_id": 7}]}):
                with patch("trading_agent.mcp_servers.moomoo.server.place_paper_order",
                           return_value=mock_order) as mock_place:
                    with patch("trading_agent.mcp_servers.journal.server.close_trade"):
                        with patch("trading_agent.notify.send"):
                            from trading_agent.graph.nodes.intraday_nodes import route_exit_or_hold
                            route_exit_or_hold(state)
            mock_place.assert_called_once()
            call_kwargs = mock_place.call_args[1] if mock_place.call_args[1] else {}
            assert call_kwargs.get("side") == "SELL"
            assert call_kwargs.get("thesis_id") == 7  # journal trade_id used as link

    def test_exit_option_uses_option_placer(self):
        opt_sym = "US.AAPL260117C00200000"
        state = _base_state(
            trigger="intraday_monitor",
            journal={"exit_decisions": [
                {"symbol": opt_sym, "action": "EXIT_TARGET",
                 "exit_qty_factor": 1.0, "reason": "target hit"},
            ]},
            positions=[{
                "symbol": opt_sym, "asset_type": "OPT",
                "qty": 2, "entry_price": 3.0, "mark": 6.5,
                "thesis_id": 9, "strategy_label": "earnings_long_call",
                "delta": 0.52, "dte": 20,
            }],
        )
        mock_order = {"thesis_id": 15, "rows": [{"order_id": "OPT456", "order_status": "SUBMITTED"}]}
        with _patch_all_emits():
            with patch("trading_agent.mcp_servers.journal.server.get_open_positions_with_thesis",
                       return_value={"rows": [{"symbol": opt_sym, "trade_id": 15}]}):
                with patch("trading_agent.mcp_servers.moomoo.server.place_paper_option_order",
                           return_value=mock_order) as mock_place_opt:
                    with patch("trading_agent.mcp_servers.journal.server.close_trade") as mock_close:
                        with patch("trading_agent.notify.send"):
                            from trading_agent.graph.nodes.intraday_nodes import route_exit_or_hold
                            route_exit_or_hold(state)
            mock_place_opt.assert_called_once()
            mock_close.assert_called_once_with(trade_id=15, exit_price=6.5, outcome="WIN", pnl=pytest.approx(700.0))

    def test_order_failure_does_not_journal(self):
        state = _base_state(
            trigger="intraday_monitor",
            journal={"exit_decisions": [
                {"symbol": "US.SPY", "action": "EXIT_STOP", "exit_qty_factor": 1.0, "reason": "stop"},
            ]},
            positions=[{
                "symbol": "US.SPY", "asset_type": "STK",
                "qty": 5, "entry_price": 500.0, "mark": 480.0,
                "thesis_id": 3, "strategy_label": "spy_short",
            }],
        )
        with _patch_all_emits():
            with patch("trading_agent.mcp_servers.journal.server.get_open_positions_with_thesis",
                       return_value={"rows": [{"symbol": "US.SPY", "trade_id": 99}]}):
                with patch("trading_agent.mcp_servers.moomoo.server.place_paper_order",
                           side_effect=ConnectionError("OpenD down")):
                    with patch("trading_agent.mcp_servers.journal.server.close_trade") as mock_close:
                        with patch("trading_agent.notify.send"):
                            from trading_agent.graph.nodes.intraday_nodes import route_exit_or_hold
                            route_exit_or_hold(state)
        mock_close.assert_not_called()

    def _mock_pg_cursor(self, fetchall_rows: list[tuple]):
        """Build a Postgres cursor mock returning the given rows from fetchall."""
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: mock_cur
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchall.return_value = fetchall_rows
        mock_cur.execute = MagicMock()
        return mock_cur

    def test_no_journal_entry_skips_close(self):
        """Position exists at broker but not in journal — must NOT close.
        Both SQLite and Postgres lookups return empty."""
        state = _base_state(
            trigger="intraday_monitor",
            journal={"exit_decisions": [
                {"symbol": "US.MANUAL", "action": "EXIT_CAUTIOUS", "exit_qty_factor": 0.5, "reason": "no thesis"},
            ]},
            positions=[{
                "symbol": "US.MANUAL", "asset_type": "OPT",
                "qty": 1, "entry_price": 5.0, "mark": 4.5,
                "thesis_id": None, "strategy_label": None,
            }],
        )
        with _patch_all_emits():
            with patch("trading_agent.mcp_servers.journal.server.get_open_positions_with_thesis",
                       return_value={"rows": []}):  # empty SQLite
                with patch("trading_agent.store.postgres.cursor",
                           return_value=self._mock_pg_cursor([])):  # empty Postgres
                    with patch("trading_agent.mcp_servers.moomoo.server.place_paper_order") as mock_place:
                        with patch("trading_agent.mcp_servers.moomoo.server.place_paper_option_order") as mock_place_opt:
                            with patch("trading_agent.mcp_servers.journal.server.close_trade") as mock_close:
                                with patch("trading_agent.notify.send"):
                                    from trading_agent.graph.nodes.intraday_nodes import route_exit_or_hold
                                    route_exit_or_hold(state)
        # No order should be placed, no journal close, no ntfy
        mock_place.assert_not_called()
        mock_place_opt.assert_not_called()
        mock_close.assert_not_called()

    def test_postgres_fallback_resolves_trade_id(self):
        """When SQLite is empty but Postgres journal_trades has the symbol,
        the resolver must fall back to Postgres, place the order, and close
        the trade via Postgres UPDATE (not the SQLite MCP)."""
        opt_sym = "US.QQQ260605C665000"
        state = _base_state(
            trigger="intraday_monitor",
            journal={"exit_decisions": [
                {"symbol": opt_sym, "action": "EXIT_TARGET", "exit_qty_factor": 1.0, "reason": "target hit"},
            ]},
            positions=[{
                "symbol": opt_sym, "asset_type": "OPT",
                "qty": 1, "entry_price": 18.51, "mark": 25.00,
                "strategy_label": "earnings_long_call",
            }],
        )
        mock_order = {"thesis_id": 42, "rows": [{"order_id": "QQQ789", "order_status": "SUBMITTED"}]}
        pg_cursor = self._mock_pg_cursor([(42, opt_sym)])

        with _patch_all_emits():
            with patch("trading_agent.mcp_servers.journal.server.get_open_positions_with_thesis",
                       return_value={"rows": []}):  # SQLite empty
                with patch("trading_agent.store.postgres.cursor", return_value=pg_cursor):
                    with patch("trading_agent.mcp_servers.moomoo.server.place_paper_option_order",
                               return_value=mock_order) as mock_place_opt:
                        with patch("trading_agent.mcp_servers.journal.server.close_trade") as mock_sqlite_close:
                            with patch("trading_agent.notify.send"):
                                from trading_agent.graph.nodes.intraday_nodes import route_exit_or_hold
                                route_exit_or_hold(state)

        # Order placed with the Postgres-resolved trade_id (42)
        mock_place_opt.assert_called_once()
        opt_kwargs = mock_place_opt.call_args[1]
        assert opt_kwargs.get("thesis_id") == 42

        # SQLite close_trade should NOT be called (trade not in SQLite)
        mock_sqlite_close.assert_not_called()

        # Postgres cursor should have received an UPDATE journal_trades call
        executes = [c.args[0] for c in pg_cursor.execute.call_args_list if c.args]
        update_called = any("UPDATE journal_trades" in q for q in executes)
        assert update_called, f"expected UPDATE journal_trades in executes: {executes!r}"


# ---------------------------------------------------------------------------
# eod_nodes
# ---------------------------------------------------------------------------

class TestReconcileJournal:
    def test_matched_positions(self):
        state = _base_state(trigger="eod_review")
        with _patch_all_emits():
            with patch("trading_agent.mcp_servers.moomoo.server.get_account_info",
                       return_value={"rows": [{"total_assets": 100_000, "cash": 50_000}]}):
                with patch("trading_agent.mcp_servers.moomoo.server.get_positions",
                           return_value={"rows": [{"code": "US.AAPL", "qty": 10, "cost_price": 200}]}):
                    with patch("trading_agent.mcp_servers.journal.server.get_open_positions_with_thesis",
                               return_value={"rows": [{"symbol": "US.AAPL", "trade_id": 1}]}):
                        from trading_agent.graph.nodes.eod_nodes import reconcile_journal
                        result = reconcile_journal(state)
        reconcile = result["journal"]["reconcile"]
        assert "US.AAPL" in reconcile["matched"]
        assert reconcile["discrepancy_count"] == 0

    def test_discrepancy_detected(self):
        state = _base_state(trigger="eod_review")
        with _patch_all_emits():
            with patch("trading_agent.mcp_servers.moomoo.server.get_account_info",
                       return_value={"rows": []}):
                with patch("trading_agent.mcp_servers.moomoo.server.get_positions",
                           return_value={"rows": [{"code": "US.NVDA", "qty": 5, "cost_price": 900}]}):
                    with patch("trading_agent.mcp_servers.journal.server.get_open_positions_with_thesis",
                               return_value={"rows": []}):
                        from trading_agent.graph.nodes.eod_nodes import reconcile_journal
                        result = reconcile_journal(state)
        reconcile = result["journal"]["reconcile"]
        assert "US.NVDA" in reconcile["only_in_moomoo"]
        assert reconcile["discrepancy_count"] == 1


class TestMarkToMarket:
    def test_stock_marks_updated(self):
        state = _base_state(
            trigger="eod_review",
            journal={"open_trades": [
                {"trade_id": 1, "symbol": "US.AAPL", "entry_price": 200.0, "qty": 10,
                 "side": "BUY", "stop": 190.0, "target": 220.0},
            ]},
        )
        with _patch_all_emits():
            with patch("trading_agent.mcp_servers.moomoo.server.get_quote",
                       return_value={"rows": [{"code": "US.AAPL", "last_price": 205.0}]}):
                from trading_agent.graph.nodes.eod_nodes import mark_to_market
                result = mark_to_market(state)
        pos = result["journal"]["marked_positions"][0]
        assert pos["mark"] == 205.0
        assert pos["unrealized_pnl"] == pytest.approx(50.0)  # (205-200)*10


class TestPersistDailyMarks:
    def test_inserts_portfolio_marks_row(self):
        state = _base_state(
            trigger="eod_review",
            account={"equity": 100_500.0, "cash": 50_000.0},
            journal={"marked_positions": [
                {"symbol": "US.AAPL", "asset_type": "STK", "qty": 10,
                 "entry_price": 200.0, "mark": 205.0, "unrealized_pnl": 50.0},
            ], "total_unrealized_pnl": 50.0},
        )
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = (42,)
        with _patch_all_emits():
            with patch("trading_agent.store.postgres.cursor", return_value=mock_cursor):
                from trading_agent.graph.nodes.eod_nodes import persist_daily_marks
                persist_daily_marks(state)
        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "INSERT INTO portfolio_marks" in sql


# ---------------------------------------------------------------------------
# premarket_nodes
# ---------------------------------------------------------------------------

class TestCollectWatchlistData:
    def test_uses_provided_watchlist(self):
        state = _base_state(
            trigger="premarket_scan",
            watchlist=["AAPL", "NVDA"],
        )
        with _patch_all_emits():
            with patch("trading_agent.mcp_servers.moomoo.server.get_quote",
                       return_value={"rows": [{"last_price": 200.0, "volume": 1_000_000,
                                               "change_rate": 0.01}]}):
                with patch("trading_agent.mcp_servers.edgar.server.get_recent_filings_for_ticker",
                           return_value={"filings": []}):
                    from trading_agent.graph.nodes.premarket_nodes import collect_watchlist_data
                    result = collect_watchlist_data(state)
        assert set(result["watchlist"]) == {"AAPL", "NVDA"}
        assert "AAPL" in result["market_data"]

    def test_falls_back_to_default_watchlist_when_empty(self):
        state = _base_state(trigger="premarket_scan", watchlist=[])
        with _patch_all_emits():
            with patch("trading_agent.mcp_servers.moomoo.server.get_quote",
                       return_value={"rows": [{"last_price": 500.0}]}):
                with patch("trading_agent.mcp_servers.edgar.server.get_recent_filings_for_ticker",
                           return_value={"filings": []}):
                    from trading_agent.graph.nodes.premarket_nodes import collect_watchlist_data
                    result = collect_watchlist_data(state)
        assert len(result["watchlist"]) > 0
        assert "SPY" in result["watchlist"]


class TestRankCandidates:
    def test_llm_candidates_returned(self):
        from trading_agent.llm.schemas import ScoutOutput, ScoutCandidate
        state = _base_state(
            trigger="premarket_scan",
            watchlist=["AAPL", "NVDA"],
            market_data={
                "AAPL": {"ticker": "AAPL", "last": 200.0, "change_pct": 0.02},
                "NVDA": {"ticker": "NVDA", "last": 900.0, "change_pct": -0.01},
            },
            regime={"label": "BULL_TREND", "confidence": 0.85, "gate": {"allow_new_entries": True}},
        )
        mock_output = ScoutOutput(
            candidates=[ScoutCandidate(ticker="AAPL", score=0.85, reason="momentum breakout")],
            skipped=[],
        )
        mock_res = MagicMock()
        mock_res.parsed = mock_output
        with _patch_all_emits():
            with patch("trading_agent.llm.get_router") as mock_router:
                mock_router.return_value.call.return_value = mock_res
                from trading_agent.graph.nodes.premarket_nodes import rank_candidates
                result = rank_candidates(state)
        assert result["candidates"][0]["ticker"] == "AAPL"
        assert result["candidates"][0]["score"] == 0.85

    def test_fallback_when_llm_fails(self):
        state = _base_state(
            trigger="premarket_scan",
            watchlist=["AAPL", "NVDA"],
            market_data={
                "AAPL": {"ticker": "AAPL", "last": 200.0, "change_pct": 0.05},
                "NVDA": {"ticker": "NVDA", "last": 900.0, "change_pct": 0.01},
            },
            regime={},
        )
        with _patch_all_emits():
            with patch("trading_agent.llm.get_router") as mock_router:
                mock_router.return_value.call.side_effect = RuntimeError("quota exceeded")
                from trading_agent.graph.nodes.premarket_nodes import rank_candidates
                result = rank_candidates(state)
        # Fallback: sorted by abs change_pct, AAPL (5%) beats NVDA (1%)
        assert len(result["candidates"]) > 0
        assert result["candidates"][0]["ticker"] == "AAPL"


class TestNtfyScanDigest:
    def _patch_dispatch_no_op(self):
        """Disable the candidate_entry subprocess fork in tests that don't care about it."""
        return patch("trading_agent.graph.nodes.premarket_nodes._dispatch_candidate_entry_if_eligible",
                     return_value=None)

    def test_sends_to_trades_topic(self):
        state = _base_state(
            trigger="premarket_scan",
            candidates=[{"ticker": "AAPL", "score": 0.9, "reason": "breakout"}],
            regime={"label": "BULL_TREND", "gate": {"allow_new_entries": True}},
        )
        with _patch_all_emits(), self._patch_dispatch_no_op():
            with patch("trading_agent.notify.send") as mock_send:
                from trading_agent.graph.nodes.premarket_nodes import ntfy_scan_digest
                ntfy_scan_digest(state)
        mock_send.assert_called_once()
        kwargs = mock_send.call_args[1] if mock_send.call_args[1] else {}
        assert kwargs.get("topic") == "trades"

    def test_dispatch_forks_when_eligible(self):
        """High-score top candidate + clean state → subprocess.Popen fires."""
        state = _base_state(
            trigger="premarket_scan",
            candidates=[{"ticker": "NVDA", "score": 0.85, "reason": "momentum breakout"}],
            regime={"label": "BULL_TREND", "gate": {"allow_new_entries": True}},
        )
        with _patch_all_emits():
            with patch("trading_agent.notify.send"):
                with patch("pathlib.Path.exists", return_value=False):  # halt flag absent
                    with patch("trading_agent.learning.soak.is_new_entry_allowed", return_value=True):
                        with patch("subprocess.Popen") as mock_popen:
                            mock_popen.return_value.pid = 12345
                            from trading_agent.graph.nodes.premarket_nodes import ntfy_scan_digest
                            ntfy_scan_digest(state)
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        # Verify CLI: python -m trading_agent.orchestrator --trigger=candidate_entry --ticker NVDA
        assert "trading_agent.orchestrator" in call_args
        assert "--trigger=candidate_entry" in call_args
        assert "NVDA" in call_args
        # Detached
        assert mock_popen.call_args[1].get("start_new_session") is True

    def test_dispatch_skipped_below_threshold(self):
        """Top candidate score < 0.6 must NOT trigger dispatch."""
        state = _base_state(
            trigger="premarket_scan",
            candidates=[{"ticker": "AAPL", "score": 0.45, "reason": "weak setup"}],
            regime={"label": "BULL_TREND", "gate": {"allow_new_entries": True}},
        )
        with _patch_all_emits():
            with patch("trading_agent.notify.send"):
                with patch("subprocess.Popen") as mock_popen:
                    from trading_agent.graph.nodes.premarket_nodes import ntfy_scan_digest
                    ntfy_scan_digest(state)
        mock_popen.assert_not_called()

    def test_dispatch_skipped_when_halt_flag(self):
        """Halt flag set → no dispatch even if score is high."""
        state = _base_state(
            trigger="premarket_scan",
            candidates=[{"ticker": "NVDA", "score": 0.9, "reason": "strong setup"}],
            regime={"label": "BULL_TREND", "gate": {"allow_new_entries": True}},
        )
        with _patch_all_emits():
            with patch("trading_agent.notify.send"):
                with patch("pathlib.Path.exists", return_value=True):  # halt flag PRESENT
                    with patch("subprocess.Popen") as mock_popen:
                        from trading_agent.graph.nodes.premarket_nodes import ntfy_scan_digest
                        ntfy_scan_digest(state)
        mock_popen.assert_not_called()

    def test_dispatch_skipped_when_regime_blocks(self):
        """Regime gate says no new entries → no dispatch."""
        state = _base_state(
            trigger="premarket_scan",
            candidates=[{"ticker": "NVDA", "score": 0.9, "reason": "strong setup"}],
            regime={"label": "CRISIS", "gate": {"allow_new_entries": False}},
        )
        with _patch_all_emits():
            with patch("trading_agent.notify.send"):
                with patch("pathlib.Path.exists", return_value=False):
                    with patch("trading_agent.learning.soak.is_new_entry_allowed", return_value=True):
                        with patch("subprocess.Popen") as mock_popen:
                            from trading_agent.graph.nodes.premarket_nodes import ntfy_scan_digest
                            ntfy_scan_digest(state)
        mock_popen.assert_not_called()

    def test_dispatch_skipped_when_soak_read_only(self):
        """Soak phase READ_ONLY → no dispatch."""
        state = _base_state(
            trigger="premarket_scan",
            candidates=[{"ticker": "NVDA", "score": 0.9, "reason": "strong setup"}],
            regime={"label": "BULL_TREND", "gate": {"allow_new_entries": True}},
        )
        with _patch_all_emits():
            with patch("trading_agent.notify.send"):
                with patch("pathlib.Path.exists", return_value=False):
                    with patch("trading_agent.learning.soak.is_new_entry_allowed", return_value=False):
                        with patch("subprocess.Popen") as mock_popen:
                            from trading_agent.graph.nodes.premarket_nodes import ntfy_scan_digest
                            ntfy_scan_digest(state)
        mock_popen.assert_not_called()


# ---------------------------------------------------------------------------
# trade_nodes — veto/defer notifications
# ---------------------------------------------------------------------------

class TestVetoDefer:
    def test_persist_veto_emits(self):
        state = _base_state(
            trigger="candidate_entry",
            proposal={"ticker": "AAPL", "proposal_id": "prop-001"},
            risk={"hard_violations": ["R1_exceed"], "reasons": ["too large"]},
        )
        with patch("trading_agent.graph.nodes.trade_nodes.emit", return_value=1) as mock_emit:
            from trading_agent.graph.nodes.trade_nodes import persist_veto
            result = persist_veto(state)
        assert result == {}
        mock_emit.assert_called_once()
        payload = mock_emit.call_args[1]["payload"]
        assert "R1_exceed" in payload["hard_violations"]

    def test_ntfy_risk_block_sends(self):
        state = _base_state(
            trigger="candidate_entry",
            proposal={"ticker": "TSLA", "proposal_id": "prop-002"},
            risk={"hard_violations": [], "reasons": ["max heat exceeded"]},
        )
        with _patch_all_emits():
            with patch("trading_agent.notify.send") as mock_send:
                from trading_agent.graph.nodes.trade_nodes import ntfy_risk_block
                ntfy_risk_block(state)
        mock_send.assert_called_once()
        kwargs = mock_send.call_args[1] if mock_send.call_args[1] else {}
        assert kwargs.get("topic") == "risk"

    def test_persist_defer_emits(self):
        state = _base_state(
            trigger="candidate_entry",
            proposal={"ticker": "META", "proposal_id": "prop-003"},
            risk={"reasons": ["regime CRISIS"]},
        )
        with patch("trading_agent.graph.nodes.trade_nodes.emit", return_value=1) as mock_emit:
            from trading_agent.graph.nodes.trade_nodes import persist_defer
            result = persist_defer(state)
        assert result == {}
        mock_emit.assert_called_once()

    def test_ntfy_defer_sends(self):
        state = _base_state(
            trigger="candidate_entry",
            proposal={"ticker": "NVDA", "proposal_id": "prop-004"},
            risk={"reasons": ["data quality degraded"]},
        )
        with _patch_all_emits():
            with patch("trading_agent.notify.send") as mock_send:
                from trading_agent.graph.nodes.trade_nodes import ntfy_defer
                ntfy_defer(state)
        mock_send.assert_called_once()
        kwargs = mock_send.call_args[1] if mock_send.call_args[1] else {}
        assert kwargs.get("topic") == "risk"


# ---------------------------------------------------------------------------
# trade_nodes — soak phase gates
# ---------------------------------------------------------------------------

class TestSoakGates:
    def test_read_only_blocks_entry(self):
        """When SOAK_PHASE=read_only, regime_execution_gate must zero qty."""
        state = _base_state(
            trigger="candidate_entry",
            proposal={"ticker": "AAPL", "qty": 5, "asset_type": "OPT"},
            regime={"label": "BULL_TREND", "gate": {"allow_new_entries": True, "size_multiplier": 1.0}},
        )
        with _patch_all_emits():
            with patch("trading_agent.learning.soak.current_phase",
                       return_value=__import__("trading_agent.learning.soak", fromlist=["SoakPhase"]).SoakPhase.READ_ONLY):
                with patch("trading_agent.learning.soak.is_new_entry_allowed", return_value=False):
                    from trading_agent.graph.nodes.trade_nodes import regime_execution_gate
                    result = regime_execution_gate(state)
        assert result["proposal"]["qty"] == 0.0

    def test_autonomous_allows_entry(self):
        """When SOAK_PHASE=autonomous (default), entry gate should not block."""
        state = _base_state(
            trigger="candidate_entry",
            proposal={"ticker": "AAPL", "qty": 2, "asset_type": "OPT"},
            regime={"label": "BULL_TREND", "gate": {"allow_new_entries": True, "size_multiplier": 1.0}},
        )
        with _patch_all_emits():
            with patch("trading_agent.learning.soak.current_phase",
                       return_value=__import__("trading_agent.learning.soak", fromlist=["SoakPhase"]).SoakPhase.AUTONOMOUS):
                with patch("trading_agent.learning.soak.is_new_entry_allowed", return_value=True):
                    from trading_agent.graph.nodes.trade_nodes import regime_execution_gate
                    result = regime_execution_gate(state)
        # qty unchanged at 2 (size_multiplier=1.0)
        assert result["proposal"]["qty"] == 2.0

    def test_tiny_paper_caps_qty(self):
        """In TINY_PAPER phase, deterministic_sizing must cap at 1 contract."""
        from trading_agent.sizing import SizingContext, ProposedTrade, SizingViolation
        state = _base_state(
            trigger="candidate_entry",
            proposal={
                "ticker": "SPY", "asset_type": "OPT", "direction": "LONG_CALL",
                "qty": 10, "entry_price": 5.0, "stop": 0.0,
                "strategy_label": "momentum_long_call", "option_delta": 0.45, "option_dte": 30,
            },
            account={"equity": 100_000.0},
            positions=[],
        )
        with _patch_all_emits():
            with patch("trading_agent.learning.soak.tiny_paper_qty_cap", return_value=1):
                with patch("trading_agent.sizing.check", return_value=[]):
                    with patch("trading_agent.sizing.blockers", return_value=[]):
                        with patch("trading_agent.sectors.known_count", return_value=10):
                            with patch("trading_agent.sectors.lookup", return_value="Technology"):
                                from trading_agent.graph.nodes.trade_nodes import deterministic_sizing
                                result = deterministic_sizing(state)
        assert result["sizing"]["approved_qty"] == 1.0
