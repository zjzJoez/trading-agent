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


class TestDispatchSilentDieWatchdog:
    """Regression for the 5/13 SPY systemd-cgroup kill.

    The watchdog runs from postgres_health every healthcheck tick. It finds
    candidate_entry_dispatched events without a matching run_start within
    the silent-die window and fires audible alerts.
    """

    def test_silent_die_emits_alert(self):
        """Dispatched but no run_start → emit + ntfy."""
        # Mock cursor returns: (ticker, age_seconds) row for SPY silent die.
        fake_cur = MagicMock()
        fake_cur.fetchall.return_value = [("SPY", 420)]  # 7 min old
        fake_ctx = MagicMock()
        fake_ctx.__enter__.return_value = fake_cur
        fake_ctx.__exit__.return_value = False

        emitted: list[dict] = []
        alerted: list[dict] = []
        with patch("trading_agent.store.postgres.cursor", return_value=fake_ctx):
            with patch("trading_agent.graph.nodes.health_nodes.emit",
                       side_effect=lambda **kw: emitted.append(kw)):
                with patch("trading_agent.graph.nodes.health_nodes._alert_ops",
                           side_effect=lambda **kw: alerted.append(kw)):
                    from trading_agent.graph.nodes.health_nodes import (
                        _check_dispatch_silent_die,
                    )
                    _check_dispatch_silent_die("test_run", "healthcheck")
        assert len(emitted) == 1
        assert emitted[0]["event_type"] == "candidate_entry_silent_die"
        assert emitted[0]["severity"] == 2
        assert emitted[0]["payload"]["ticker"] == "SPY"
        assert len(alerted) == 1
        assert "SPY" in alerted[0]["title"]

    def test_no_silent_dies_no_alert(self):
        """Empty result set → no emit, no ntfy."""
        fake_cur = MagicMock()
        fake_cur.fetchall.return_value = []
        fake_ctx = MagicMock()
        fake_ctx.__enter__.return_value = fake_cur
        fake_ctx.__exit__.return_value = False

        emitted: list[dict] = []
        alerted: list[dict] = []
        with patch("trading_agent.store.postgres.cursor", return_value=fake_ctx):
            with patch("trading_agent.graph.nodes.health_nodes.emit",
                       side_effect=lambda **kw: emitted.append(kw)):
                with patch("trading_agent.graph.nodes.health_nodes._alert_ops",
                           side_effect=lambda **kw: alerted.append(kw)):
                    from trading_agent.graph.nodes.health_nodes import (
                        _check_dispatch_silent_die,
                    )
                    _check_dispatch_silent_die("test_run", "healthcheck")
        assert emitted == []
        assert alerted == []


class TestLLMSchemaViolationWatchdog:
    """Watchdog fires when LLM channel produces enough garbage to be audible."""

    def test_above_threshold_alerts(self):
        """≥5 schema violations in window → emit + ntfy."""
        fake_cur = MagicMock()
        fake_cur.fetchall.return_value = [("trader_synthesizer", 4), ("news_analyst", 2)]
        fake_cur.fetchone.return_value = None  # no recent alert in cooldown window
        fake_ctx = MagicMock()
        fake_ctx.__enter__.return_value = fake_cur
        fake_ctx.__exit__.return_value = False

        emitted: list[dict] = []
        alerted: list[dict] = []
        with patch("trading_agent.store.postgres.cursor", return_value=fake_ctx):
            with patch("trading_agent.graph.nodes.health_nodes.emit",
                       side_effect=lambda **kw: emitted.append(kw)):
                with patch("trading_agent.graph.nodes.health_nodes._alert_ops",
                           side_effect=lambda **kw: alerted.append(kw)):
                    from trading_agent.graph.nodes.health_nodes import (
                        _check_llm_schema_violations,
                    )
                    _check_llm_schema_violations("test_run", "healthcheck")
        assert len(emitted) == 1
        assert emitted[0]["event_type"] == "llm_schema_violation_alert"
        assert emitted[0]["payload"]["total"] == 6
        assert "trader_synthesizer" in emitted[0]["payload"]["per_role"]
        assert len(alerted) == 1

    def test_below_threshold_silent(self):
        """3 violations in window (< threshold=5) → no alert."""
        fake_cur = MagicMock()
        fake_cur.fetchall.return_value = [("news_analyst", 3)]
        fake_ctx = MagicMock()
        fake_ctx.__enter__.return_value = fake_cur
        fake_ctx.__exit__.return_value = False

        emitted: list[dict] = []
        alerted: list[dict] = []
        with patch("trading_agent.store.postgres.cursor", return_value=fake_ctx):
            with patch("trading_agent.graph.nodes.health_nodes.emit",
                       side_effect=lambda **kw: emitted.append(kw)):
                with patch("trading_agent.graph.nodes.health_nodes._alert_ops",
                           side_effect=lambda **kw: alerted.append(kw)):
                    from trading_agent.graph.nodes.health_nodes import (
                        _check_llm_schema_violations,
                    )
                    _check_llm_schema_violations("test_run", "healthcheck")
        assert emitted == []
        assert alerted == []

    def test_cooldown_suppresses_repeat_alert(self):
        """Above threshold but recently alerted → suppress."""
        fake_cur = MagicMock()
        fake_cur.fetchall.return_value = [("trader_synthesizer", 8)]
        fake_cur.fetchone.return_value = (1,)  # recent alert found
        fake_ctx = MagicMock()
        fake_ctx.__enter__.return_value = fake_cur
        fake_ctx.__exit__.return_value = False

        emitted: list[dict] = []
        alerted: list[dict] = []
        with patch("trading_agent.store.postgres.cursor", return_value=fake_ctx):
            with patch("trading_agent.graph.nodes.health_nodes.emit",
                       side_effect=lambda **kw: emitted.append(kw)):
                with patch("trading_agent.graph.nodes.health_nodes._alert_ops",
                           side_effect=lambda **kw: alerted.append(kw)):
                    from trading_agent.graph.nodes.health_nodes import (
                        _check_llm_schema_violations,
                    )
                    _check_llm_schema_violations("test_run", "healthcheck")
        assert emitted == []
        assert alerted == []


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


class TestFailedUnitsCheck:
    """5/22 incident: persistent failed unit triggered hourly sev-2 ntfy
    spam until the operator complained. Cooldown logic added to suppress
    repeated alerts for the same unit set within FAILED_UNITS_NTFY_COOLDOWN_HOURS.
    """

    def _systemd_output(self, failed: list[str]) -> str:
        return "\n".join(
            f"{u} loaded failed failed Description for {u}" for u in failed
        )

    def test_ntfy_fires_first_time_seeing_failure(self):
        state = _base_state(trigger="healthcheck")
        fake_run = MagicMock(stdout=self._systemd_output(
            ["trading-agent-watchlist-refresh.service"]
        ))
        # No prior alert → cooldown lookup returns no row
        fake_cur = MagicMock()
        fake_cur.fetchone.return_value = None
        fake_ctx = MagicMock()
        fake_ctx.__enter__.return_value = fake_cur
        fake_ctx.__exit__.return_value = False

        sent: list = []
        with _patch_all_emits():
            with patch("subprocess.run", return_value=fake_run):
                with patch("trading_agent.store.postgres.cursor", return_value=fake_ctx):
                    with patch("trading_agent.notify.send",
                               side_effect=lambda **kw: sent.append(kw)):
                        from trading_agent.graph.nodes.health_nodes import (
                            failed_units_check,
                        )
                        failed_units_check(state)
        assert len(sent) == 1, "first-time failure should fire ntfy"
        assert "FAILED" in sent[0].get("title", "")

    def test_ntfy_suppressed_during_cooldown(self):
        state = _base_state(trigger="healthcheck")
        fake_run = MagicMock(stdout=self._systemd_output(
            ["trading-agent-watchlist-refresh.service"]
        ))
        # Prior alert exists → cooldown lookup returns (1,)
        fake_cur = MagicMock()
        fake_cur.fetchone.return_value = (1,)
        fake_ctx = MagicMock()
        fake_ctx.__enter__.return_value = fake_cur
        fake_ctx.__exit__.return_value = False

        sent: list = []
        with _patch_all_emits():
            with patch("subprocess.run", return_value=fake_run):
                with patch("trading_agent.store.postgres.cursor", return_value=fake_ctx):
                    with patch("trading_agent.notify.send",
                               side_effect=lambda **kw: sent.append(kw)):
                        from trading_agent.graph.nodes.health_nodes import (
                            failed_units_check,
                        )
                        failed_units_check(state)
        assert sent == [], (
            "repeated alert for same unit set within cooldown should be suppressed"
        )

    def test_no_failures_emits_clean(self):
        state = _base_state(trigger="healthcheck")
        fake_run = MagicMock(stdout="")
        sent: list = []
        with _patch_all_emits():
            with patch("subprocess.run", return_value=fake_run):
                with patch("trading_agent.notify.send",
                           side_effect=lambda **kw: sent.append(kw)):
                    from trading_agent.graph.nodes.health_nodes import (
                        failed_units_check,
                    )
                    failed_units_check(state)
        assert sent == [], "clean state should never ntfy"


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


class TestSyncFillStatus:
    """Sync stale journal SUBMITTED → real broker fill state (6/2 IBM/SNOW)."""

    def _pg(self, open_rows):
        cur = MagicMock()
        cur.__enter__ = lambda s: cur
        cur.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = open_rows
        cur.execute = MagicMock()
        return cur

    def test_filled_order_syncs_entry_price(self):
        # journal has trade #7 OPEN, broker_order_id 3112455
        pg = self._pg([(7, "US.SNOW260717C270000", "3112455", 33.80)])
        with _patch_all_emits():
            with patch("trading_agent.store.postgres.cursor", return_value=pg):
                with patch("trading_agent.mcp_servers.moomoo.server.get_orders",
                           return_value={"rows": [{"order_id": "3112455",
                                                   "order_status": "FILLED_ALL",
                                                   "dealt_qty": 2, "dealt_avg_price": 33.55}]}):
                    from trading_agent.graph.nodes.intraday_nodes import sync_fill_status
                    sync_fill_status(_base_state(trigger="intraday_monitor"))
        executes = [c.args[0] for c in pg.execute.call_args_list if c.args]
        assert any("UPDATE journal_trades" in q and "entry_price" in q for q in executes)

    def test_cancelled_order_marks_unfilled(self):
        pg = self._pg([(5, "US.NVDA", "3112342", 224.36)])
        with _patch_all_emits():
            with patch("trading_agent.store.postgres.cursor", return_value=pg):
                with patch("trading_agent.mcp_servers.moomoo.server.get_orders",
                           return_value={"rows": [{"order_id": "3112342",
                                                   "order_status": "CANCELLED_ALL"}]}):
                    from trading_agent.graph.nodes.intraday_nodes import sync_fill_status
                    sync_fill_status(_base_state(trigger="intraday_monitor"))
        executes = [c.args[0] for c in pg.execute.call_args_list if c.args]
        assert any("'UNFILLED'" in q for q in executes)

    def test_pending_order_left_alone(self):
        pg = self._pg([(5, "US.NVDA", "3112342", 224.36)])
        with _patch_all_emits():
            with patch("trading_agent.store.postgres.cursor", return_value=pg):
                with patch("trading_agent.mcp_servers.moomoo.server.get_orders",
                           return_value={"rows": [{"order_id": "3112342",
                                                   "order_status": "SUBMITTED"}]}):
                    from trading_agent.graph.nodes.intraday_nodes import sync_fill_status
                    sync_fill_status(_base_state(trigger="intraday_monitor"))
        # Only the SELECT ran; no UPDATE for a still-pending order
        executes = [c.args[0] for c in pg.execute.call_args_list if c.args]
        assert not any("UPDATE journal_trades" in q for q in executes)

    def test_broker_unreachable_skips_gracefully(self):
        pg = self._pg([(5, "US.NVDA", "3112342", 224.36)])
        with _patch_all_emits():
            with patch("trading_agent.store.postgres.cursor", return_value=pg):
                with patch("trading_agent.mcp_servers.moomoo.server.get_orders",
                           side_effect=RuntimeError("OpenD timeout")):
                    from trading_agent.graph.nodes.intraday_nodes import sync_fill_status
                    result = sync_fill_status(_base_state(trigger="intraday_monitor"))
        assert result == {}  # graceful no-op, never crashes the intraday loop


class TestDetectExitTriggers:
    """Detect-exit-triggers now drives the deterministic hard executor.

    Per-rule branch coverage of hard_exit_decision lives in
    tests/exits/test_hard_executor.py — these tests check the *integration*:
    state → enrichment → executor call → demotion guard → emit.
    """

    def _make_pos(self):
        # ITM call with plenty of DTE so the hard executor won't fire DTE
        # rules; the test controls entry/mark/stop/target via enrichment.
        return {
            "symbol": "US.SPY260817C00700000",
            "asset_type": "OPT", "qty": 2,
            "entry_price": 3.0, "mark": 1.5,
        }

    def _enrichment(self, stop=1.0, target=6.0, exit_plan=None, mfe_so_far=None):
        return {
            "US.SPY260817C00700000": {
                "trade_id": 3, "thesis_id": 4,
                "stop": stop, "target": target,
                "exit_plan": exit_plan, "mfe_so_far": mfe_so_far,
                "opened_at": None,
                "direction": "LONG_CALL", "thesis_text": "t", "invalidation": "i",
            }
        }

    def test_hold_when_mark_between_stop_and_target(self):
        # mark=1.5 between stop=1.0 and target=6.0 → no rule triggers → HOLD
        state = _base_state(
            trigger="intraday_monitor",
            positions=[self._make_pos()],
            regime={"label": "RANGE_LOW_VOL", "confidence": 0.9, "gate": {}},
        )
        with _patch_all_emits():
            with patch(
                "trading_agent.graph.nodes.intraday_nodes._load_journal_enrichment_by_symbol",
                return_value=self._enrichment(),
            ):
                with patch(
                    "trading_agent.mcp_servers.moomoo.server.get_quote",
                    return_value={"rows": []},
                ):
                    from trading_agent.graph.nodes.intraday_nodes import detect_exit_triggers
                    result = detect_exit_triggers(state)
        decisions = result["journal"]["exit_decisions"]
        assert len(decisions) == 1
        assert decisions[0]["action"] == "HOLD"

    def test_exit_stop_when_mark_breaches_stop(self):
        pos = self._make_pos()
        pos["mark"] = 0.95  # below stop=1.0
        state = _base_state(
            trigger="intraday_monitor",
            positions=[pos],
            regime={"label": "RANGE_LOW_VOL", "confidence": 0.8, "gate": {}},
        )
        with _patch_all_emits():
            with patch(
                "trading_agent.graph.nodes.intraday_nodes._load_journal_enrichment_by_symbol",
                return_value=self._enrichment(),
            ):
                with patch(
                    "trading_agent.mcp_servers.moomoo.server.get_quote",
                    return_value={"rows": []},
                ):
                    from trading_agent.graph.nodes.intraday_nodes import detect_exit_triggers
                    result = detect_exit_triggers(state)
        decisions = result["journal"]["exit_decisions"]
        assert decisions[0]["action"] == "EXIT_STOP"
        assert decisions[0]["exit_qty_factor"] == 1.0

    def test_exit_regime_when_crisis(self):
        # mark mid-range, but regime=CRISIS → executor fires P0
        state = _base_state(
            trigger="intraday_monitor",
            positions=[self._make_pos()],
            regime={"label": "CRISIS", "confidence": 0.9, "gate": {}},
        )
        with _patch_all_emits():
            with patch(
                "trading_agent.graph.nodes.intraday_nodes._load_journal_enrichment_by_symbol",
                return_value=self._enrichment(),
            ):
                with patch(
                    "trading_agent.mcp_servers.moomoo.server.get_quote",
                    return_value={"rows": []},
                ):
                    from trading_agent.graph.nodes.intraday_nodes import detect_exit_triggers
                    result = detect_exit_triggers(state)
        assert result["journal"]["exit_decisions"][0]["action"] == "EXIT_REGIME"

    def test_no_plan_holds_and_warns(self):
        """Position with no exit_plan AND no legacy stop/target → HOLD,
        plus a sev-1 event so ops can clean it up (likely a manual fill).
        First detection in the day fires; subsequent ticks dedup."""
        pos = self._make_pos()
        state = _base_state(
            trigger="intraday_monitor",
            positions=[pos],
            regime={"label": "RANGE_LOW_VOL", "confidence": 0.9, "gate": {}},
        )
        captured: list[dict] = []
        with patch(
            "trading_agent.graph.nodes.intraday_nodes.emit",
            side_effect=lambda **kw: captured.append(kw),
        ):
            with patch(
                "trading_agent.graph.nodes.intraday_nodes._load_journal_enrichment_by_symbol",
                return_value={  # missing stop+target+exit_plan
                    "US.SPY260817C00700000": {
                        "trade_id": 3, "thesis_id": 4,
                        "stop": None, "target": None, "exit_plan": None,
                    }
                },
            ):
                # Patch the dedup helper to return False (no prior alert today)
                # so the first-detection branch fires.
                with patch(
                    "trading_agent.graph.nodes.intraday_nodes._no_plan_alerted_today",
                    return_value=False,
                ):
                    with patch(
                        "trading_agent.mcp_servers.moomoo.server.get_quote",
                        return_value={"rows": []},
                    ):
                        from trading_agent.graph.nodes.intraday_nodes import detect_exit_triggers
                        result = detect_exit_triggers(state)
        decisions = result["journal"]["exit_decisions"]
        assert decisions[0]["action"] == "HOLD"
        assert "no_exit_plan" in decisions[0]["reason"]
        no_plan_events = [e for e in captured if e.get("event_type") == "position_no_exit_plan"]
        assert len(no_plan_events) == 1
        assert no_plan_events[0]["severity"] == 1
        # New: should mention manual trade interpretation
        assert "manual" in no_plan_events[0]["payload"].get("interpretation", "")

    def test_no_plan_suppressed_when_already_alerted_today(self):
        """Per-symbol per-day dedup: if we already alerted today, the
        decision still happens (HOLD) but no new sev-1 event fires."""
        pos = self._make_pos()
        state = _base_state(
            trigger="intraday_monitor",
            positions=[pos],
            regime={"label": "RANGE_LOW_VOL", "confidence": 0.9, "gate": {}},
        )
        captured: list[dict] = []
        with patch(
            "trading_agent.graph.nodes.intraday_nodes.emit",
            side_effect=lambda **kw: captured.append(kw),
        ):
            with patch(
                "trading_agent.graph.nodes.intraday_nodes._load_journal_enrichment_by_symbol",
                return_value={
                    "US.SPY260817C00700000": {
                        "trade_id": 3, "thesis_id": 4,
                        "stop": None, "target": None, "exit_plan": None,
                    }
                },
            ):
                # Dedup says: already alerted today → suppress
                with patch(
                    "trading_agent.graph.nodes.intraday_nodes._no_plan_alerted_today",
                    return_value=True,
                ):
                    with patch(
                        "trading_agent.mcp_servers.moomoo.server.get_quote",
                        return_value={"rows": []},
                    ):
                        from trading_agent.graph.nodes.intraday_nodes import detect_exit_triggers
                        detect_exit_triggers(state)
        no_plan_events = [e for e in captured if e.get("event_type") == "position_no_exit_plan"]
        assert no_plan_events == [], "second alert in same day should suppress"

    def test_partial_exit_on_qty_1_demoted_to_hold(self):
        """Regression: 5/12 NVDA EXIT_CAUTIOUS-but-actually-closed-100%
        bug. The hard executor's scale-out rung could emit factor=0.5
        on a 1-contract position; downstream route_exit_or_hold would
        ``max(1, round(0.5))`` and force a full close. Demote to HOLD."""
        from trading_agent.llm.schemas import ExitPlan, ScaleOutRung

        # qty=1, mark=12 (between stop=7 and full target=18); scale rung
        # at 12 with factor=0.5 → executor returns EXIT_TARGET factor 0.5
        pos = {
            "symbol": "US.NVDA260817C220000",
            "asset_type": "OPT", "qty": 1,  # ← single contract
            "entry_price": 10.0, "mark": 12.0,
        }
        plan = ExitPlan(
            hard_stop=7.0, hard_target=18.0,
            scale_out_ladder=[ScaleOutRung(at_mark=12.0, exit_factor=0.5)],
        )
        enrichment = {
            "US.NVDA260817C220000": {
                "trade_id": 5, "thesis_id": 6,
                "stop": 7.0, "target": 18.0,
                "exit_plan": plan.model_dump(),
                "mfe_so_far": None, "opened_at": None,
            }
        }
        state = _base_state(
            trigger="intraday_monitor",
            positions=[pos],
            regime={"label": "RANGE_LOW_VOL", "confidence": 0.7, "gate": {}},
        )
        with _patch_all_emits():
            with patch(
                "trading_agent.graph.nodes.intraday_nodes._load_journal_enrichment_by_symbol",
                return_value=enrichment,
            ):
                with patch(
                    "trading_agent.mcp_servers.moomoo.server.get_quote",
                    return_value={"rows": []},
                ):
                    from trading_agent.graph.nodes.intraday_nodes import detect_exit_triggers
                    result = detect_exit_triggers(state)
        dec = result["journal"]["exit_decisions"][0]
        assert dec["action"] == "HOLD", (
            "regression: partial-exit signal on qty=1 should demote to HOLD"
        )
        assert dec["exit_qty_factor"] == 0.0
        assert "demoted_from_EXIT_TARGET" in dec["reason"]

    def test_partial_exit_on_qty_2_not_demoted(self):
        """qty=2 × factor=0.5 = 1 contract → physically possible, keep."""
        from trading_agent.llm.schemas import ExitPlan, ScaleOutRung

        pos = {
            "symbol": "US.NVDA260817C220000",
            "asset_type": "OPT", "qty": 2,
            "entry_price": 10.0, "mark": 12.0,
        }
        plan = ExitPlan(
            hard_stop=7.0, hard_target=18.0,
            scale_out_ladder=[ScaleOutRung(at_mark=12.0, exit_factor=0.5)],
        )
        enrichment = {
            "US.NVDA260817C220000": {
                "trade_id": 5, "thesis_id": 6,
                "stop": 7.0, "target": 18.0,
                "exit_plan": plan.model_dump(),
                "mfe_so_far": None, "opened_at": None,
            }
        }
        state = _base_state(
            trigger="intraday_monitor",
            positions=[pos],
            regime={"label": "RANGE_LOW_VOL", "confidence": 0.7, "gate": {}},
        )
        with _patch_all_emits():
            with patch(
                "trading_agent.graph.nodes.intraday_nodes._load_journal_enrichment_by_symbol",
                return_value=enrichment,
            ):
                with patch(
                    "trading_agent.mcp_servers.moomoo.server.get_quote",
                    return_value={"rows": []},
                ):
                    from trading_agent.graph.nodes.intraday_nodes import detect_exit_triggers
                    result = detect_exit_triggers(state)
        dec = result["journal"]["exit_decisions"][0]
        assert dec["action"] == "EXIT_TARGET"
        assert dec["exit_qty_factor"] == 0.5

    def test_scale_rungs_taken_passed_through_skips_fired_rung(self):
        """Critical fix for the scale-rung repeat-fire bug.

        If a partial rung at mark=12 has already fired (scale_rungs_taken=1),
        a subsequent tick at mark=12.5 MUST NOT re-fire it. Without
        persistence + passthrough, the executor would close 50% every tick
        until qty=1 and the safety demotion kicks in.
        """
        from trading_agent.llm.schemas import ExitPlan, ScaleOutRung

        pos = {
            "symbol": "US.NVDA260817C220000",
            "asset_type": "OPT", "qty": 2,
            "entry_price": 10.0, "mark": 12.5,
        }
        plan = ExitPlan(
            hard_stop=7.0, hard_target=18.0,
            scale_out_ladder=[
                ScaleOutRung(at_mark=12.0, exit_factor=0.5),
                ScaleOutRung(at_mark=15.0, exit_factor=1.0),
            ],
        )
        enrichment = {
            "US.NVDA260817C220000": {
                "trade_id": 5, "thesis_id": 6,
                "stop": 7.0, "target": 18.0,
                "exit_plan": plan.model_dump(),
                "mfe_so_far": None, "opened_at": None,
                "scale_rungs_taken": 1,  # ← rung #1 already fired
            }
        }
        state = _base_state(
            trigger="intraday_monitor",
            positions=[pos],
            regime={"label": "RANGE_LOW_VOL", "confidence": 0.7, "gate": {}},
        )
        with _patch_all_emits():
            with patch(
                "trading_agent.graph.nodes.intraday_nodes._load_journal_enrichment_by_symbol",
                return_value=enrichment,
            ):
                with patch(
                    "trading_agent.mcp_servers.moomoo.server.get_quote",
                    return_value={"rows": []},
                ):
                    from trading_agent.graph.nodes.intraday_nodes import detect_exit_triggers
                    result = detect_exit_triggers(state)
        dec = result["journal"]["exit_decisions"][0]
        # Rung #1 already taken; mark=12.5 hasn't hit rung #2 at 15.0 → HOLD
        assert dec["action"] == "HOLD", (
            "regression: scale rung fired twice — scale_rungs_taken not "
            "being passed through to the executor"
        )


class TestRouteExitOrHold:
    @pytest.fixture(autouse=True)
    def _no_real_broker_order_scan(self):
        """route_exit_or_hold scans live broker orders for orphan adoption
        and consults the pending-close guard (Phase 0.3). Without OpenD the
        moomoo SDK blocks on connect retries, and without a journal DB the
        guard fails CLOSED (None → exits deferred) — stub both to the clean
        empty state for every test in this class; adoption and fail-closed
        behavior have their own dedicated tests."""
        with patch("trading_agent.mcp_servers.moomoo.server.get_orders",
                   return_value={"rows": []}), \
             patch("trading_agent.exits.fill_confirm.pending_close_trade_ids_sqlite",
                   return_value=set()), \
             patch("trading_agent.exits.fill_confirm.pending_close_trade_ids_pg",
                   return_value=set()):
            yield

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
            with patch("trading_agent.store.postgres.get_open_journal_trades",
                       return_value=[{"symbol": "US.AAPL", "trade_id": 7}]):
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
            with patch("trading_agent.store.postgres.get_open_journal_trades",
                       return_value=[{"symbol": opt_sym, "trade_id": 15}]):
                with patch("trading_agent.mcp_servers.moomoo.server.place_paper_option_order",
                           return_value=mock_order) as mock_place_opt:
                    with patch("trading_agent.exits.fill_confirm.write_pending_close",
                               return_value=True) as mock_pending:
                        with patch("trading_agent.exits.fill_confirm.pending_close_trade_ids_sqlite",
                                   return_value=set()):
                            with patch("trading_agent.exits.fill_confirm.pending_close_trade_ids_pg",
                                       return_value=set()):
                                with patch("trading_agent.mcp_servers.journal.server.close_trade") as mock_close:
                                    with patch("trading_agent.notify.send"):
                                        from trading_agent.graph.nodes.intraday_nodes import route_exit_or_hold
                                        route_exit_or_hold(state)
            mock_place_opt.assert_called_once()
            # Phase 0.3: SELL closes price toward the executable side —
            # no live bid in the position, so mark 6.5 × 0.98 = 6.37.
            assert mock_place_opt.call_args[1].get("price") == pytest.approx(6.37)
            # The journal must NOT close at placement — close happens in
            # sync_fill_status once the broker confirms the dealt price.
            mock_close.assert_not_called()
            mock_pending.assert_called_once()

    def test_exit_option_fallback_close_is_cost_honest(self):
        """When the pending-close state cannot be written, the legacy
        immediate close fires — at the executable-side limit, net of fees."""
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
        pg_cursor = self._mock_pg_cursor([])
        with _patch_all_emits():
            with patch("trading_agent.store.postgres.get_open_journal_trades",
                       return_value=[{"symbol": opt_sym, "trade_id": 15}]):
                with patch("trading_agent.store.postgres.cursor", return_value=pg_cursor):
                    with patch("trading_agent.mcp_servers.moomoo.server.place_paper_option_order",
                               return_value=mock_order):
                        with patch("trading_agent.exits.fill_confirm.write_pending_close",
                                   return_value=False):
                            with patch("trading_agent.exits.fill_confirm.pending_close_trade_ids_pg",
                                       return_value=set()):
                                with patch("trading_agent.notify.send"):
                                    from trading_agent.graph.nodes.intraday_nodes import route_exit_or_hold
                                    route_exit_or_hold(state)
        # Postgres-only journal: the fallback close is an UPDATE journal_trades
        # at the executable-side limit with the fee-derived outcome.
        # gross (6.37-3.0)×2×100 = 674; round-trip fees 2×$1×2 = $4 → net 670 → WIN
        closes = [c for c in pg_cursor.execute.call_args_list
                  if c.args and "SET exit_price" in c.args[0]]
        assert len(closes) == 1, f"expected exactly one journal close, saw {closes!r}"
        params = closes[0].args[1]
        assert params[0] == pytest.approx(6.37)   # exit at bid-side limit
        assert params[1] == "WIN"                 # outcome from NET pnl
        assert params[3] == 15                    # trade_id

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
            with patch("trading_agent.store.postgres.get_open_journal_trades",
                       return_value=[{"symbol": "US.SPY", "trade_id": 99}]):
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
            with patch("trading_agent.store.postgres.get_open_journal_trades",
                       return_value=[]):  # empty journal
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
        # thesis_id=None exercises the "no parent thesis" branch.
        pg_cursor = self._mock_pg_cursor([])

        with _patch_all_emits():
            with patch("trading_agent.store.postgres.get_open_journal_trades",
                       return_value=[{"symbol": opt_sym, "trade_id": 42,
                                      "thesis_id": None}]):
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

        # SQLite close_trade is never part of the autonomous path anymore
        mock_sqlite_close.assert_not_called()

        # Phase 0.3: the Postgres write is now the PENDING-CLOSE state stamp
        # (UPDATE journal_trades ... broker_fill_json), not an outcome close —
        # the close lands in sync_fill_status at the broker's dealt price.
        executes = [c.args[0] for c in pg_cursor.execute.call_args_list if c.args]
        update_called = any("UPDATE journal_trades" in q for q in executes)
        assert update_called, f"expected UPDATE journal_trades in executes: {executes!r}"
        outcome_closed = any("SET exit_price" in q for q in executes)
        assert not outcome_closed, "journal must not close at placement time"


    def test_duplicate_open_rows_resolve_to_newest(self):
        """Phantom (placed-never-filled) + real fill can share a symbol while
        both OPEN. Rows come newest-first; the resolver must keep the FIRST
        (newest) so the phantom never receives the real trade's close."""
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
        mock_order = {"thesis_id": 7, "rows": [{"order_id": "ORD9", "order_status": "SUBMITTED"}]}
        with _patch_all_emits():
            with patch("trading_agent.store.postgres.get_open_journal_trades",
                       return_value=[
                           {"symbol": "US.AAPL", "trade_id": 99},  # newest = real
                           {"symbol": "US.AAPL", "trade_id": 7},   # older = phantom
                       ]):
                with patch("trading_agent.mcp_servers.moomoo.server.place_paper_order",
                           return_value=mock_order) as mock_place:
                    with patch("trading_agent.exits.fill_confirm.write_pending_close",
                               return_value=True) as mock_pending:
                        with patch("trading_agent.notify.send"):
                            from trading_agent.graph.nodes.intraday_nodes import route_exit_or_hold
                            route_exit_or_hold(state)
        mock_place.assert_called_once()
        assert mock_place.call_args[1].get("thesis_id") == 99  # newest row wins
        assert mock_pending.call_args[0][0] == 99              # pending on newest

    def test_journal_unreadable_defers_all_exits(self):
        """get_open_journal_trades() → None means UNKNOWN, not 'no trades':
        every exit defers one tick; nothing is placed or misclassified as a
        manual position."""
        state = _base_state(
            trigger="intraday_monitor",
            journal={"exit_decisions": [
                {"symbol": "US.AAPL", "action": "EXIT_STOP", "exit_qty_factor": 1.0, "reason": "stop"},
            ]},
            positions=[{
                "symbol": "US.AAPL", "asset_type": "STK",
                "qty": 10, "entry_price": 200.0, "mark": 180.0,
            }],
        )
        with _patch_all_emits():
            with patch("trading_agent.store.postgres.get_open_journal_trades",
                       return_value=None):
                with patch("trading_agent.mcp_servers.moomoo.server.place_paper_order") as mock_place:
                    with patch("trading_agent.notify.send"):
                        from trading_agent.graph.nodes.intraday_nodes import route_exit_or_hold
                        route_exit_or_hold(state)
        mock_place.assert_not_called()


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
                    with patch("trading_agent.store.postgres.get_open_journal_trades",
                               return_value=[{"symbol": "US.AAPL", "trade_id": 1}]):
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
                    with patch("trading_agent.store.postgres.get_open_journal_trades",
                               return_value=[]):
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


class TestScoutPromptDeBias:
    """The 6/2 NVDA-bias fix: scout prompt must present HOT TODAY names
    separately and steer ranking toward relative volume, not absolute."""

    def _md(self, **rows):
        return rows

    def test_hot_today_section_present_and_first(self):
        from trading_agent.graph.nodes.premarket_nodes import _build_scout_prompt
        watchlist = ["AAPL", "PLTR", "NVDA"]
        market_data = {
            "AAPL": {"last": 200.0, "change_pct": 0.0, "volume": 5e7, "avg_volume_3m": 5e7},
            "PLTR": {"last": 160.0, "change_pct": 0.0, "volume": 1.2e8, "avg_volume_3m": 4e7, "hot_today": True},
            "NVDA": {"last": 224.0, "change_pct": 0.0, "volume": 2.1e8, "avg_volume_3m": 2.0e8},
        }
        prompt = _build_scout_prompt(watchlist, market_data, {"label": "RANGE_LOW_VOL"})
        assert "HOT TODAY" in prompt
        # PLTR (hot) should appear in the HOT TODAY block, before "rest of watchlist"
        hot_idx = prompt.index("HOT TODAY")
        rest_idx = prompt.index("rest of watchlist")
        pltr_idx = prompt.index("PLTR")
        assert hot_idx < pltr_idx < rest_idx, "hot name must be in the HOT TODAY section"

    def test_rvol_column_present(self):
        from trading_agent.graph.nodes.premarket_nodes import _build_scout_prompt
        # PLTR: 120M vol / 40M avg = 3.0x rvol (genuinely hot)
        # NVDA: 210M vol / 200M avg = 1.05x rvol (normal despite huge absolute)
        market_data = {
            "PLTR": {"last": 160.0, "change_pct": 0.0, "volume": 1.2e8, "avg_volume_3m": 4e7, "hot_today": True},
            "NVDA": {"last": 224.0, "change_pct": 0.0, "volume": 2.1e8, "avg_volume_3m": 2.0e8},
        }
        prompt = _build_scout_prompt(["PLTR", "NVDA"], market_data, {"label": "RANGE_LOW_VOL"})
        assert "rvol=3.0x" in prompt  # PLTR genuinely hot
        assert "rvol=1.0x" in prompt or "rvol=1.1x" in prompt  # NVDA normal
        # The prompt must explicitly warn against ranking by absolute volume
        assert "absolute" in prompt.lower()
        assert "rvol" in prompt.lower()

    def test_mom50d_column_present(self):
        """Multi-day momentum (price vs 50d MA) surfaces — the narrative-trend
        signal the operator wanted beyond rvol (task 58)."""
        from trading_agent.graph.nodes.premarket_nodes import _build_scout_prompt
        market_data = {
            "ARM": {"last": 160.0, "change_pct": 0.0, "volume": 8e7,
                    "avg_volume_3m": 4e7, "momentum_50d": 0.32, "hot_today": True},
            "AAPL": {"last": 200.0, "change_pct": 0.0, "volume": 5e7,
                     "avg_volume_3m": 5e7, "momentum_50d": -0.03},
        }
        prompt = _build_scout_prompt(["ARM", "AAPL"], market_data, {"label": "RANGE_LOW_VOL"})
        assert "mom50d=+32%" in prompt   # ARM trending up
        assert "mom50d=-3%" in prompt    # AAPL flat/down
        assert "narrative" in prompt.lower()

    def test_no_hot_today_still_builds(self):
        from trading_agent.graph.nodes.premarket_nodes import _build_scout_prompt
        market_data = {"AAPL": {"last": 200.0, "change_pct": 0.0, "volume": 5e7}}
        prompt = _build_scout_prompt(["AAPL"], market_data, {"label": "RANGE_LOW_VOL"})
        # No hot names → no hot SECTION header (the "HOT TODAY —" data block).
        # The phrase still appears in RANKING GUIDANCE, so check the header form.
        assert "HOT TODAY — your watchlist names" not in prompt
        assert "rest of watchlist" in prompt
        assert "AAPL" in prompt


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
        # CRITICAL: patch notify.send. Without this, the fallback path's
        # scout_llm_degraded ntfy call hits real ntfy.sh during every CI
        # run (and the pre-deploy pytest gate). User got buzzed at 17:49
        # local on 6/2 from this exact test before the patch was added.
        with _patch_all_emits():
            with patch("trading_agent.notify.send") as mock_ntfy:
                with patch("trading_agent.llm.get_router") as mock_router:
                    mock_router.return_value.call.side_effect = RuntimeError("quota exceeded")
                    from trading_agent.graph.nodes.premarket_nodes import rank_candidates
                    result = rank_candidates(state)
        # Fallback: sorted by abs change_pct, AAPL (5%) beats NVDA (1%)
        assert len(result["candidates"]) > 0
        assert result["candidates"][0]["ticker"] == "AAPL"
        # The new loud-alert behavior should fire exactly once
        mock_ntfy.assert_called_once()
        # And the title should mention scout degradation
        call_kwargs = mock_ntfy.call_args.kwargs
        assert "Scout" in call_kwargs.get("title", "")


class TestNtfyScanDigest:
    def _patch_dispatch_no_op(self):
        """Disable the candidate_entry subprocess fork in tests that don't care about it."""
        return patch("trading_agent.graph.nodes.premarket_nodes._dispatch_candidate_entry_if_eligible",
                     return_value=None)

    def _patch_clean_guards(self):
        """Mock the position/cooldown guards to allow dispatch by default."""
        from contextlib import ExitStack
        class _CleanGuards:
            def __enter__(self):
                self._stack = ExitStack()
                self._stack.enter_context(patch(
                    "trading_agent.graph.nodes.premarket_nodes._existing_exposure",
                    return_value=None,
                ))
                self._stack.enter_context(patch(
                    "trading_agent.graph.nodes.premarket_nodes._in_dispatch_cooldown",
                    return_value=None,
                ))
                # 0 open positions → full R2 dispatch budget available
                self._stack.enter_context(patch(
                    "trading_agent.graph.nodes.premarket_nodes._open_position_count",
                    return_value=0,
                ))
                # Not recently dispatched → dedup guard allows dispatch
                self._stack.enter_context(patch(
                    "trading_agent.graph.nodes.premarket_nodes._recently_dispatched",
                    return_value=False,
                ))
                # Market open → regular-hours gate allows dispatch (else the
                # gate makes these tests flaky depending on wall-clock time).
                self._stack.enter_context(patch(
                    "trading_agent.market_calendar.is_us_market_open",
                    return_value=True,
                ))
                return self
            def __exit__(self, *a):
                return self._stack.__exit__(*a)
        return _CleanGuards()

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
        """High-score top candidate + clean state → systemctl start fires.

        Dispatch is now via independent systemd unit (not subprocess.Popen
        with start_new_session) so the child doesn't get SIGTERM'd when its
        parent unit's cgroup is reaped. Verify the systemctl invocation is
        correct: reset-failed first (clear any stuck failed state), then
        start --no-block (fire-and-forget).
        """
        state = _base_state(
            trigger="premarket_scan",
            candidates=[{"ticker": "NVDA", "score": 0.85, "reason": "momentum breakout"}],
            regime={"label": "BULL_TREND", "gate": {"allow_new_entries": True}},
        )
        with _patch_all_emits(), self._patch_clean_guards():
            with patch("trading_agent.notify.send"):
                with patch("pathlib.Path.exists", return_value=False):  # halt flag absent
                    with patch("trading_agent.learning.soak.is_new_entry_allowed", return_value=True):
                        with patch("subprocess.run") as mock_run:
                            # Both reset-failed and start return code 0
                            mock_run.return_value = MagicMock(returncode=0, stderr=b"")
                            from trading_agent.graph.nodes.premarket_nodes import ntfy_scan_digest
                            ntfy_scan_digest(state)
        # Should have been called twice: reset-failed then start
        assert mock_run.call_count == 2
        reset_call, start_call = mock_run.call_args_list
        assert reset_call[0][0] == [
            "sudo", "-n", "/bin/systemctl", "reset-failed",
            "trading-agent-candidate-entry@NVDA.service",
        ]
        assert start_call[0][0] == [
            "sudo", "-n", "/bin/systemctl", "start", "--no-block",
            "trading-agent-candidate-entry@NVDA.service",
        ]

    def test_dispatch_top_n_not_just_top_1(self):
        """Top-N change: multiple candidates above threshold dispatch (bounded
        by MAX_DISPATCH_PER_SCAN + R2 slots). This breaks the NVDA winner-
        take-all — rotation names now get a slot even when a mega-cap is #1."""
        state = _base_state(
            trigger="premarket_scan",
            candidates=[
                {"ticker": "NVDA", "score": 0.72, "reason": "hot mega"},
                {"ticker": "PLTR", "score": 0.61, "reason": "rotation leader"},
                {"ticker": "HOOD", "score": 0.55, "reason": "fintech momentum"},
                {"ticker": "INTC", "score": 0.40, "reason": "below threshold"},  # < 0.50
            ],
            regime={"label": "BULL_TREND", "gate": {"allow_new_entries": True}},
        )
        started: list[str] = []

        def _fake_run(cmd, **kw):
            # Record the ticker from the start (not reset-failed) calls
            if "start" in cmd:
                started.append(cmd[-1])
            return MagicMock(returncode=0, stderr=b"")

        with _patch_all_emits(), self._patch_clean_guards():
            with patch("trading_agent.notify.send"):
                with patch("pathlib.Path.exists", return_value=False):
                    with patch("trading_agent.learning.soak.is_new_entry_allowed", return_value=True):
                        with patch("subprocess.run", side_effect=_fake_run):
                            from trading_agent.graph.nodes.premarket_nodes import ntfy_scan_digest
                            ntfy_scan_digest(state)
        # 3 candidates clear 0.50 (NVDA, PLTR, HOOD); INTC at 0.40 excluded.
        # MAX_DISPATCH_PER_SCAN=3, R2 budget=6 → all 3 dispatch.
        assert started == [
            "trading-agent-candidate-entry@NVDA.service",
            "trading-agent-candidate-entry@PLTR.service",
            "trading-agent-candidate-entry@HOOD.service",
        ], f"expected top-3 dispatched, got {started}"

    def test_dispatch_respects_r2_remaining_slots(self):
        """If only 1 position slot remains (5 of 6 open), dispatch exactly 1
        even when 3 candidates clear threshold."""
        state = _base_state(
            trigger="premarket_scan",
            candidates=[
                {"ticker": "NVDA", "score": 0.72, "reason": "a"},
                {"ticker": "PLTR", "score": 0.61, "reason": "b"},
                {"ticker": "HOOD", "score": 0.55, "reason": "c"},
            ],
            regime={"label": "BULL_TREND", "gate": {"allow_new_entries": True}},
        )
        started: list[str] = []
        with _patch_all_emits():
            with patch("trading_agent.market_calendar.is_us_market_open", return_value=True):
              with patch("trading_agent.graph.nodes.premarket_nodes._existing_exposure", return_value=None):
                with patch("trading_agent.graph.nodes.premarket_nodes._in_dispatch_cooldown", return_value=None):
                    with patch("trading_agent.graph.nodes.premarket_nodes._open_position_count", return_value=5):
                        with patch("trading_agent.notify.send"):
                            with patch("pathlib.Path.exists", return_value=False):
                                with patch("trading_agent.learning.soak.is_new_entry_allowed", return_value=True):
                                    def _fake_run(cmd, **kw):
                                        if "start" in cmd:
                                            started.append(cmd[-1])
                                        return MagicMock(returncode=0, stderr=b"")
                                    with patch("subprocess.run", side_effect=_fake_run):
                                        from trading_agent.graph.nodes.premarket_nodes import ntfy_scan_digest
                                        ntfy_scan_digest(state)
        assert started == ["trading-agent-candidate-entry@NVDA.service"], (
            f"only 1 slot remained (6-5); expected 1 dispatch, got {started}"
        )

    def test_dispatch_dedup_skips_recently_dispatched(self):
        """Same ticker dispatched in the last ~20 min → skip (no double-fire).

        Regression for the 6/2 silent-die false positive: a manual run + the
        scheduled 12:30 premarket both surfaced IBM/CBOE/SNOW; the second
        dispatch no-op'd on systemctl start and the watchdog false-alarmed."""
        state = _base_state(
            trigger="premarket_scan",
            candidates=[{"ticker": "NVDA", "score": 0.72, "reason": "x"}],
            regime={"label": "BULL_TREND", "gate": {"allow_new_entries": True}},
        )
        started: list[str] = []
        skipped: list[dict] = []
        with patch("trading_agent.graph.nodes.premarket_nodes.emit",
                   side_effect=lambda **kw: skipped.append(kw) if kw.get("event_type") == "candidate_entry_skipped" else None):
          with patch("trading_agent.market_calendar.is_us_market_open", return_value=True):
            with patch("trading_agent.graph.nodes.premarket_nodes._existing_exposure", return_value=None):
                with patch("trading_agent.graph.nodes.premarket_nodes._in_dispatch_cooldown", return_value=None):
                    with patch("trading_agent.graph.nodes.premarket_nodes._open_position_count", return_value=0):
                        with patch("trading_agent.graph.nodes.premarket_nodes._recently_dispatched", return_value=True):
                            with patch("trading_agent.notify.send"):
                                with patch("pathlib.Path.exists", return_value=False):
                                    with patch("trading_agent.learning.soak.is_new_entry_allowed", return_value=True):
                                        def _fake_run(cmd, **kw):
                                            if "start" in cmd:
                                                started.append(cmd[-1])
                                            return MagicMock(returncode=0, stderr=b"")
                                        with patch("subprocess.run", side_effect=_fake_run):
                                            from trading_agent.graph.nodes.premarket_nodes import ntfy_scan_digest
                                            ntfy_scan_digest(state)
        assert started == [], "recently-dispatched ticker must not re-dispatch"
        assert any(s["payload"].get("reason") == "already_dispatched_recently" for s in skipped)

    def test_dispatch_skipped_when_market_closed(self):
        """6/2 fix: the 12:30 premarket scan ranks + sends digest but does NOT
        dispatch (market closed) — the 13:35 open scan is what trades."""
        state = _base_state(
            trigger="premarket_scan",
            candidates=[{"ticker": "NVDA", "score": 0.9, "reason": "x"}],
            regime={"label": "BULL_TREND", "gate": {"allow_new_entries": True}},
        )
        skipped: list[dict] = []
        with patch("trading_agent.graph.nodes.premarket_nodes.emit",
                   side_effect=lambda **kw: skipped.append(kw) if kw.get("event_type") == "candidate_entry_skipped" else None):
            with patch("trading_agent.market_calendar.is_us_market_open", return_value=False):
                with patch("trading_agent.notify.send"):
                    with patch("pathlib.Path.exists", return_value=False):
                        with patch("trading_agent.learning.soak.is_new_entry_allowed", return_value=True):
                            with patch("subprocess.run") as mock_run:
                                from trading_agent.graph.nodes.premarket_nodes import ntfy_scan_digest
                                ntfy_scan_digest(state)
        mock_run.assert_not_called()  # no dispatch when market closed
        assert any(s["payload"].get("reason") == "outside_regular_hours" for s in skipped)

    def test_dispatch_failed_emits_severity_2(self):
        """systemctl start non-zero exit → dispatch_failed event, severity=2."""
        state = _base_state(
            trigger="premarket_scan",
            candidates=[{"ticker": "NVDA", "score": 0.85, "reason": "momentum"}],
            regime={"label": "BULL_TREND", "gate": {"allow_new_entries": True}},
        )
        emitted: list[dict] = []
        with patch("trading_agent.graph.nodes.premarket_nodes.emit",
                   side_effect=lambda **kw: emitted.append(kw)), self._patch_clean_guards():
            with patch("trading_agent.notify.send"):
                with patch("pathlib.Path.exists", return_value=False):
                    with patch("trading_agent.learning.soak.is_new_entry_allowed", return_value=True):
                        with patch("subprocess.run") as mock_run:
                            # reset-failed OK, start fails
                            mock_run.side_effect = [
                                MagicMock(returncode=0, stderr=b""),
                                MagicMock(returncode=1, stderr=b"Unit not found"),
                            ]
                            from trading_agent.graph.nodes.premarket_nodes import ntfy_scan_digest
                            ntfy_scan_digest(state)
        failure_events = [e for e in emitted if e.get("event_type") == "candidate_entry_dispatch_failed"]
        assert len(failure_events) == 1
        assert failure_events[0]["severity"] == 2
        assert "Unit not found" in failure_events[0]["payload"]["error"]

    def test_dispatch_skipped_when_already_exposed(self):
        """Same underlying already held → no dispatch."""
        state = _base_state(
            trigger="premarket_scan",
            candidates=[{"ticker": "NVDA", "score": 0.9, "reason": "strong setup"}],
            regime={"label": "BULL_TREND", "gate": {"allow_new_entries": True}},
        )
        with _patch_all_emits():
            with patch("trading_agent.market_calendar.is_us_market_open", return_value=True):
              with patch("trading_agent.notify.send"):
                with patch("pathlib.Path.exists", return_value=False):
                    with patch("trading_agent.learning.soak.is_new_entry_allowed", return_value=True):
                        with patch("trading_agent.graph.nodes.premarket_nodes._existing_exposure",
                                   return_value={"reason": "already_holding_same_underlying",
                                                 "detail": "US.NVDA qty=1"}):
                            with patch("trading_agent.graph.nodes.premarket_nodes._in_dispatch_cooldown",
                                       return_value=None):
                                with patch("subprocess.run") as mock_popen:  # dispatch uses systemctl now
                                    from trading_agent.graph.nodes.premarket_nodes import ntfy_scan_digest
                                    ntfy_scan_digest(state)
        mock_popen.assert_not_called()  # systemctl shouldn't be invoked when skipped

    def test_dispatch_skipped_when_in_cooldown(self):
        """Same ticker dispatched within last 7 days → no dispatch."""
        state = _base_state(
            trigger="premarket_scan",
            candidates=[{"ticker": "NVDA", "score": 0.9, "reason": "strong setup"}],
            regime={"label": "BULL_TREND", "gate": {"allow_new_entries": True}},
        )
        with _patch_all_emits():
            with patch("trading_agent.market_calendar.is_us_market_open", return_value=True):
              with patch("trading_agent.notify.send"):
                with patch("pathlib.Path.exists", return_value=False):
                    with patch("trading_agent.learning.soak.is_new_entry_allowed", return_value=True):
                        with patch("trading_agent.graph.nodes.premarket_nodes._existing_exposure",
                                   return_value=None):
                            with patch("trading_agent.graph.nodes.premarket_nodes._in_dispatch_cooldown",
                                       return_value={"last_ts": "2026-05-10T12:30Z", "age_days": 1.2}):
                                with patch("subprocess.run") as mock_popen:  # dispatch uses systemctl now
                                    from trading_agent.graph.nodes.premarket_nodes import ntfy_scan_digest
                                    ntfy_scan_digest(state)
        mock_popen.assert_not_called()  # systemctl shouldn't be invoked when skipped

    def test_existing_exposure_same_underlying(self):
        """Direct unit test: _existing_exposure flags same-underlying holding."""
        with patch("trading_agent.mcp_servers.moomoo.server.get_positions",
                   return_value={"rows": [{"code": "US.NVDA260605C500000", "qty": 1.0}]}):
            with patch("trading_agent.sectors.lookup", return_value="Technology"):
                with patch("trading_agent.store.postgres.cursor") as mock_cur_ctx:
                    mock_cur = MagicMock()
                    mock_cur.__enter__ = lambda s: mock_cur
                    mock_cur.__exit__ = MagicMock(return_value=False)
                    mock_cur.fetchall.return_value = []
                    mock_cur_ctx.return_value = mock_cur
                    from trading_agent.graph.nodes.premarket_nodes import _existing_exposure
                    result = _existing_exposure("NVDA")
        assert result is not None
        assert result["reason"] == "already_holding_same_underlying"

    def test_existing_exposure_same_sector(self):
        """Direct unit test: _existing_exposure flags same-sector concentration."""
        def fake_lookup(t):
            return {"NVDA": "Technology", "AMD": "Technology"}.get(t.upper())
        with patch("trading_agent.mcp_servers.moomoo.server.get_positions",
                   return_value={"rows": [{"code": "US.AMD260605C150000", "qty": 1.0}]}):
            with patch("trading_agent.sectors.lookup", side_effect=fake_lookup):
                with patch("trading_agent.store.postgres.cursor") as mock_cur_ctx:
                    mock_cur = MagicMock()
                    mock_cur.__enter__ = lambda s: mock_cur
                    mock_cur.__exit__ = MagicMock(return_value=False)
                    mock_cur.fetchall.return_value = []
                    mock_cur_ctx.return_value = mock_cur
                    from trading_agent.graph.nodes.premarket_nodes import _existing_exposure
                    result = _existing_exposure("NVDA")
        assert result is not None
        assert result["reason"] == "already_exposed_to_sector"
        assert "Technology" in result["detail"]

    def test_existing_exposure_ignores_zombie_positions(self):
        """qty=0 broker rows (zombies) should NOT count as exposure."""
        with patch("trading_agent.mcp_servers.moomoo.server.get_positions",
                   return_value={"rows": [{"code": "US.NVDA260605C500000", "qty": 0.0}]}):
            with patch("trading_agent.sectors.lookup", return_value="Technology"):
                with patch("trading_agent.store.postgres.cursor") as mock_cur_ctx:
                    mock_cur = MagicMock()
                    mock_cur.__enter__ = lambda s: mock_cur
                    mock_cur.__exit__ = MagicMock(return_value=False)
                    mock_cur.fetchall.return_value = []
                    mock_cur_ctx.return_value = mock_cur
                    from trading_agent.graph.nodes.premarket_nodes import _existing_exposure
                    result = _existing_exposure("NVDA")
        assert result is None  # zombie ignored → dispatch allowed

    def test_dispatch_skipped_below_threshold(self):
        """Top candidate score < 0.6 must NOT trigger dispatch."""
        state = _base_state(
            trigger="premarket_scan",
            candidates=[{"ticker": "AAPL", "score": 0.45, "reason": "weak setup"}],
            regime={"label": "BULL_TREND", "gate": {"allow_new_entries": True}},
        )
        with _patch_all_emits():
            with patch("trading_agent.notify.send"):
                with patch("subprocess.run") as mock_popen:  # dispatch uses systemctl now
                    from trading_agent.graph.nodes.premarket_nodes import ntfy_scan_digest
                    ntfy_scan_digest(state)
        mock_popen.assert_not_called()  # systemctl shouldn't be invoked when skipped

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
                    with patch("subprocess.run") as mock_popen:  # dispatch uses systemctl now
                        from trading_agent.graph.nodes.premarket_nodes import ntfy_scan_digest
                        ntfy_scan_digest(state)
        mock_popen.assert_not_called()  # systemctl shouldn't be invoked when skipped

    def test_dispatch_skipped_when_regime_blocks(self):
        """Regime gate says no new entries → no dispatch."""
        state = _base_state(
            trigger="premarket_scan",
            candidates=[{"ticker": "NVDA", "score": 0.9, "reason": "strong setup"}],
            regime={"label": "CRISIS", "gate": {"allow_new_entries": False}},
        )
        with _patch_all_emits(), self._patch_clean_guards():
            with patch("trading_agent.notify.send"):
                with patch("pathlib.Path.exists", return_value=False):
                    with patch("trading_agent.learning.soak.is_new_entry_allowed", return_value=True):
                        with patch("subprocess.run") as mock_popen:  # dispatch uses systemctl now
                            from trading_agent.graph.nodes.premarket_nodes import ntfy_scan_digest
                            ntfy_scan_digest(state)
        mock_popen.assert_not_called()  # systemctl shouldn't be invoked when skipped

    def test_dispatch_skipped_when_soak_read_only(self):
        """Soak phase READ_ONLY → no dispatch."""
        state = _base_state(
            trigger="premarket_scan",
            candidates=[{"ticker": "NVDA", "score": 0.9, "reason": "strong setup"}],
            regime={"label": "BULL_TREND", "gate": {"allow_new_entries": True}},
        )
        with _patch_all_emits(), self._patch_clean_guards():
            with patch("trading_agent.notify.send"):
                with patch("pathlib.Path.exists", return_value=False):
                    with patch("trading_agent.learning.soak.is_new_entry_allowed", return_value=False):
                        with patch("subprocess.run") as mock_popen:  # dispatch uses systemctl now
                            from trading_agent.graph.nodes.premarket_nodes import ntfy_scan_digest
                            ntfy_scan_digest(state)
        mock_popen.assert_not_called()  # systemctl shouldn't be invoked when skipped


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

    def test_persist_veto_marks_thesis_rejected(self):
        """A VETO must flip the linked thesis to 'rejected' so it doesn't
        sit at 'open' forever. Regression for the 5/22 zombie pattern."""
        state = _base_state(
            trigger="candidate_entry",
            proposal={"ticker": "AAPL", "proposal_id": "prop-001", "thesis_id": 99},
            risk={"hard_violations": ["R1_exceed"], "reasons": ["too large"]},
        )
        captured_updates: list[tuple] = []
        fake_cur = MagicMock()
        fake_cur.execute = MagicMock(
            side_effect=lambda q, params=None: captured_updates.append((q, params))
        )
        fake_ctx = MagicMock()
        fake_ctx.__enter__.return_value = fake_cur
        fake_ctx.__exit__.return_value = False

        with patch("trading_agent.graph.nodes.trade_nodes.emit", return_value=1):
            with patch("trading_agent.store.postgres.cursor", return_value=fake_ctx):
                from trading_agent.graph.nodes.trade_nodes import persist_veto
                persist_veto(state)
        # Expect at least one UPDATE journal_theses set status='rejected'
        assert any(
            "UPDATE journal_theses" in q and "rejected" in q
            for q, _ in captured_updates
        ), f"expected rejection UPDATE in {captured_updates!r}"

    def test_persist_defer_marks_thesis_rejected(self):
        state = _base_state(
            trigger="candidate_entry",
            proposal={"ticker": "META", "proposal_id": "prop-003", "thesis_id": 77},
            risk={"reasons": ["burn_in_trade_5_of_15"]},
        )
        captured_updates: list[tuple] = []
        fake_cur = MagicMock()
        fake_cur.execute = MagicMock(
            side_effect=lambda q, params=None: captured_updates.append((q, params))
        )
        fake_ctx = MagicMock()
        fake_ctx.__enter__.return_value = fake_cur
        fake_ctx.__exit__.return_value = False

        with patch("trading_agent.graph.nodes.trade_nodes.emit", return_value=1):
            with patch("trading_agent.store.postgres.cursor", return_value=fake_ctx):
                from trading_agent.graph.nodes.trade_nodes import persist_defer
                persist_defer(state)
        assert any(
            "UPDATE journal_theses" in q and "rejected" in q
            for q, _ in captured_updates
        ), f"expected rejection UPDATE in {captured_updates!r}"

    def test_persist_veto_without_thesis_id_does_not_crash(self):
        """Old proposals without thesis_id shouldn't break the VETO path."""
        state = _base_state(
            trigger="candidate_entry",
            proposal={"ticker": "AAPL", "proposal_id": "prop-001"},  # no thesis_id
            risk={"hard_violations": ["R1_exceed"], "reasons": ["too large"]},
        )
        with patch("trading_agent.graph.nodes.trade_nodes.emit", return_value=1):
            from trading_agent.graph.nodes.trade_nodes import persist_veto
            result = persist_veto(state)
        assert result == {}  # graceful pass-through

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

    def test_regime_gate_rounds_half_up_for_options(self):
        """1 contract × 0.5 multiplier must round to 1, not 0 (the 5/12 bug)."""
        state = _base_state(
            trigger="candidate_entry",
            proposal={"ticker": "NVDA", "qty": 1, "asset_type": "OPT"},
            regime={"label": "BULL_TREND",
                    "gate": {"allow_new_entries": True, "size_multiplier": 0.5}},
        )
        with _patch_all_emits():
            with patch("trading_agent.learning.soak.is_new_entry_allowed", return_value=True):
                from trading_agent.graph.nodes.trade_nodes import regime_execution_gate
                result = regime_execution_gate(state)
        # 1 * 0.5 = 0.5 → round half-up → 1 (was 0 with int truncation)
        assert result["proposal"]["qty"] == 1.0

    def test_regime_gate_rounds_down_for_aggressive_downsize(self):
        """When size_multiplier is small enough that scaled qty < 0.5, must round to 0."""
        state = _base_state(
            trigger="candidate_entry",
            proposal={"ticker": "NVDA", "qty": 1, "asset_type": "OPT"},
            regime={"label": "VOLATILE_TRANSITION",
                    "gate": {"allow_new_entries": True, "size_multiplier": 0.3}},
        )
        with _patch_all_emits():
            with patch("trading_agent.learning.soak.is_new_entry_allowed", return_value=True):
                from trading_agent.graph.nodes.trade_nodes import regime_execution_gate
                result = regime_execution_gate(state)
        # 1 * 0.3 = 0.3 → round half-up → 0 (regime strongly downsizes)
        assert result["proposal"]["qty"] == 0.0

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
