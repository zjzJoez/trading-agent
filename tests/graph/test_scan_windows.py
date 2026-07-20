"""Week-1 revival steps 4+5: time-normalized rvol, decline cooldown,
and the zero-eligible-scan liveness tripwire.

All external I/O (Postgres, ntfy, systemctl, market calendar where the
assertion depends on a fixed clock) is mocked.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> dict:
    s = {
        "run_id": "test-run-scan-windows",
        "trigger": "premarket_scan",
        "watchlist": [],
        "market_data": {},
        "candidates": [],
        "errors": [],
    }
    s.update(overrides)
    return s


def _fake_cursor_ctx(fetchone=None, fetchall=None):
    cur = MagicMock()
    cur.fetchone.return_value = fetchone
    cur.fetchall.return_value = fetchall or []
    ctx = MagicMock()
    ctx.__enter__.return_value = cur
    ctx.__exit__.return_value = False
    return ctx


# ---------------------------------------------------------------------------
# Step 4 — intraday volume fraction f(t)
# ---------------------------------------------------------------------------

class TestIntradayVolumeFraction:
    def test_endpoints(self):
        from trading_agent.graph.nodes.premarket_nodes import (
            RVOL_INTRADAY_FRACTION_FLOOR,
            intraday_volume_fraction,
        )
        # t=0 interpolates to 0.00 but the floor kicks in
        assert intraday_volume_fraction(0.0) == RVOL_INTRADAY_FRACTION_FLOOR
        assert intraday_volume_fraction(390.0) == 1.0

    def test_anchor_values(self):
        from trading_agent.graph.nodes.premarket_nodes import intraday_volume_fraction
        assert abs(intraday_volume_fraction(30.0) - 0.12) < 1e-9
        assert abs(intraday_volume_fraction(60.0) - 0.20) < 1e-9
        assert abs(intraday_volume_fraction(240.0) - 0.46) < 1e-9

    def test_linear_interpolation_between_anchors(self):
        from trading_agent.graph.nodes.premarket_nodes import intraday_volume_fraction
        # midway between (30, 0.12) and (60, 0.20)
        assert abs(intraday_volume_fraction(45.0) - 0.16) < 1e-9

    def test_monotone_nondecreasing(self):
        from trading_agent.graph.nodes.premarket_nodes import intraday_volume_fraction
        vals = [intraday_volume_fraction(float(t)) for t in range(0, 391, 5)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))

    def test_floor_prevents_divide_by_near_zero(self):
        from trading_agent.graph.nodes.premarket_nodes import (
            RVOL_INTRADAY_FRACTION_FLOOR,
            intraday_volume_fraction,
        )
        # First minutes after the open: raw interpolation would be ~0
        for t in (0.0, 1.0, 5.0, 10.0):
            assert intraday_volume_fraction(t) >= RVOL_INTRADAY_FRACTION_FLOOR

    def test_clamps_out_of_session_inputs(self):
        from trading_agent.graph.nodes.premarket_nodes import (
            RVOL_INTRADAY_FRACTION_FLOOR,
            intraday_volume_fraction,
        )
        assert intraday_volume_fraction(-30.0) == RVOL_INTRADAY_FRACTION_FLOOR
        assert intraday_volume_fraction(500.0) == 1.0

    def test_half_day_rescales_to_full_profile(self):
        from trading_agent.graph.nodes.premarket_nodes import intraday_volume_fraction
        # A 210-min half-day at its close has all the day's volume in.
        assert intraday_volume_fraction(210.0, session_minutes=210.0) == 1.0
        # Midway through a half-day == midway through a full day.
        assert abs(
            intraday_volume_fraction(105.0, session_minutes=210.0)
            - intraday_volume_fraction(195.0)
        ) < 1e-9


# ---------------------------------------------------------------------------
# Step 4 — rvol semantics in the scout prompt
# ---------------------------------------------------------------------------

def _patch_rth(minutes: float, session: float = 390.0):
    from contextlib import ExitStack

    class _Rth:
        def __enter__(self):
            self._stack = ExitStack()
            self._stack.enter_context(patch(
                "trading_agent.market_calendar.is_us_market_open", return_value=True))
            self._stack.enter_context(patch(
                "trading_agent.market_calendar.minutes_since_open",
                return_value=(minutes, session)))
            return self

        def __exit__(self, *a):
            return self._stack.__exit__(*a)

    return _Rth()


class TestRvolTimeNormalization:
    NOW = datetime(2026, 7, 20, 14, 15, tzinfo=timezone.utc)  # Mon 10:15 ET

    def test_rvol_at_1015_et_lands_near_1x_for_normal_name(self):
        """45 min after the open, a name that has done 16% of its average
        daily volume (the typical cumulative fraction by then) reads 1.0x —
        not the mechanical 0.16x the old full-day division produced."""
        from trading_agent.graph.nodes.premarket_nodes import _build_scout_prompt
        market_data = {
            "AAPL": {"last": 200.0, "change_pct": 0.01,
                     "volume": 8e6, "avg_volume_3m": 5e7},
        }
        with _patch_rth(45.0):
            prompt = _build_scout_prompt(["AAPL"], market_data, {}, now_utc=self.NOW)
        assert "rvol=1.0x" in prompt

    def test_premarket_path_unchanged_prior_day_semantics(self):
        """Outside RTH the volume field carries the PRIOR day's total, so the
        plain full-day ratio still applies (f=1.0)."""
        from trading_agent.graph.nodes.premarket_nodes import _build_scout_prompt
        market_data = {
            "PLTR": {"last": 160.0, "change_pct": 0.0,
                     "volume": 1.2e8, "avg_volume_3m": 4e7},
        }
        with patch("trading_agent.market_calendar.is_us_market_open", return_value=False):
            prompt = _build_scout_prompt(["PLTR"], market_data, {}, now_utc=self.NOW)
        assert "rvol=3.0x" in prompt
        assert "pre-open digest" in prompt

    def test_935_style_input_no_longer_reads_mechanically_dead(self):
        """Regression: 5 min after the open a normally-trading name has done
        ~4% of its daily volume. Old math: 0.0x (looked dead). New math with
        the f-floor: 0.04 / 0.06 = 0.7x — a normal reading."""
        from trading_agent.graph.nodes.premarket_nodes import _build_scout_prompt
        market_data = {
            "MSFT": {"last": 500.0, "change_pct": 0.002,
                     "volume": 2e6, "avg_volume_3m": 5e7},
        }
        with _patch_rth(5.0):
            prompt = _build_scout_prompt(["MSFT"], market_data, {}, now_utc=self.NOW)
        assert "rvol=0.7x" in prompt
        assert "rvol=0.0x" not in prompt

    def test_prompt_states_time_and_minutes_since_open(self):
        from trading_agent.graph.nodes.premarket_nodes import _build_scout_prompt
        market_data = {"AAPL": {"last": 200.0, "change_pct": 0.0, "volume": 1e7}}
        with _patch_rth(45.0):
            prompt = _build_scout_prompt(["AAPL"], market_data, {}, now_utc=self.NOW)
        assert "time: 14:15 UTC" in prompt
        assert "45 min since the open" in prompt

    def test_prompt_guidance_says_rvol_comparable_any_time(self):
        from trading_agent.graph.nodes.premarket_nodes import _build_scout_prompt
        market_data = {"AAPL": {"last": 200.0, "change_pct": 0.0, "volume": 1e7}}
        with _patch_rth(45.0):
            prompt = _build_scout_prompt(["AAPL"], market_data, {}, now_utc=self.NOW)
        assert "TIME-NORMALIZED" in prompt
        assert "ANY time of day" in prompt


# ---------------------------------------------------------------------------
# Step 5 — decline/veto trading-day cooldown
# ---------------------------------------------------------------------------

class TestDeclineCooldown:
    # Wed 2026-06-10 — a normal trading day
    NOW = datetime(2026, 6, 10, 18, 0, tzinfo=timezone.utc)

    def _check(self, declined_ts, event_type="declined"):
        from trading_agent.graph.nodes.premarket_nodes import _in_decline_cooldown
        ctx = _fake_cursor_ctx(fetchone=(declined_ts, event_type))
        with patch("trading_agent.store.postgres.cursor", return_value=ctx):
            return _in_decline_cooldown("NVDA", now_utc=self.NOW)

    def _rows_cursor(self, rows):
        """Fake cursor that applies the query's reason-exclusion plus
        ORDER BY ts DESC LIMIT 1 to `rows` of (ts, event_type, reason) —
        so the tests exercise the SQL filter's semantics, and fail if the
        no_parsed_output exclusion ever drops out of the query text."""
        cur = MagicMock()

        def _execute(sql, params=None):
            assert "no_parsed_output" in sql, (
                "decline-cooldown query must exclude LLM parse failures"
            )
            kept = sorted(
                (r for r in rows if (r[2] or "") != "no_parsed_output"),
                key=lambda r: r[0],
                reverse=True,
            )
            cur.fetchone.return_value = (kept[0][0], kept[0][1]) if kept else None

        cur.execute.side_effect = _execute
        ctx = MagicMock()
        ctx.__enter__.return_value = cur
        ctx.__exit__.return_value = False
        return ctx

    def test_no_parsed_output_decline_is_not_a_considered_decline(self):
        """A declined/reason=no_parsed_output event 1 trading day ago is an
        LLM parse FAILURE (trade_nodes.build_trade_proposal), not the
        pipeline saying no — it must NOT put the ticker in cooldown."""
        from trading_agent.graph.nodes.premarket_nodes import _in_decline_cooldown
        ctx = self._rows_cursor([
            (datetime(2026, 6, 9, 15, 0, tzinfo=timezone.utc),
             "declined", "no_parsed_output"),
        ])
        with patch("trading_agent.store.postgres.cursor", return_value=ctx):
            assert _in_decline_cooldown("NVDA", now_utc=self.NOW) is None

    def test_genuine_decline_still_cooldowns_under_reason_filter(self):
        from trading_agent.graph.nodes.premarket_nodes import _in_decline_cooldown
        ctx = self._rows_cursor([
            (datetime(2026, 6, 9, 15, 0, tzinfo=timezone.utc),
             "declined", "no edge vs regime"),
        ])
        with patch("trading_agent.store.postgres.cursor", return_value=ctx):
            res = _in_decline_cooldown("NVDA", now_utc=self.NOW)
        assert res is not None
        assert res["trading_days_ago"] == 1

    def test_declined_one_trading_day_ago_in_cooldown(self):
        # Tue 6/9 → Wed 6/10 = 1 trading day ≤ 3 → blocked
        res = self._check(datetime(2026, 6, 9, 15, 0, tzinfo=timezone.utc))
        assert res is not None
        assert res["trading_days_ago"] == 1
        assert res["declined_at"].startswith("2026-06-09")

    def test_declined_four_trading_days_ago_eligible(self):
        # Thu 6/4 → Wed 6/10 spans Fri 6/5, Mon 6/8, Tue 6/9, Wed 6/10 = 4 > 3
        res = self._check(datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc))
        assert res is None

    def test_veto_event_also_triggers_cooldown(self):
        res = self._check(
            datetime(2026, 6, 9, 15, 0, tzinfo=timezone.utc),
            event_type="veto_persisted",
        )
        assert res is not None
        assert res["event_type"] == "veto_persisted"

    def test_no_decline_history_eligible(self):
        from trading_agent.graph.nodes.premarket_nodes import _in_decline_cooldown
        ctx = _fake_cursor_ctx(fetchone=None)
        with patch("trading_agent.store.postgres.cursor", return_value=ctx):
            assert _in_decline_cooldown("NVDA", now_utc=self.NOW) is None

    def test_db_error_is_best_effort_none(self):
        from trading_agent.graph.nodes.premarket_nodes import _in_decline_cooldown
        with patch("trading_agent.store.postgres.cursor",
                   side_effect=Exception("conn refused")):
            assert _in_decline_cooldown("NVDA", now_utc=self.NOW) is None

    def test_dispatch_falls_through_to_next_ranked_candidate(self):
        """#1 in decline cooldown → emit candidate_skipped_cooldown for it and
        dispatch #2 (never dispatch nothing because #1 is cooling down)."""
        state = _base_state(
            candidates=[
                {"ticker": "NVDA", "score": 0.80, "reason": "declined yesterday"},
                {"ticker": "PLTR", "score": 0.70, "reason": "rotation leader"},
            ],
            regime={"label": "BULL_TREND", "gate": {"allow_new_entries": True}},
        )
        started: list[str] = []
        emitted: list[dict] = []

        def _fake_run(cmd, **kw):
            if "start" in cmd:
                started.append(cmd[-1])
            return MagicMock(returncode=0, stderr=b"")

        def _decline(ticker, *a, **kw):
            if ticker == "NVDA":
                return {"declined_at": "2026-06-09T15:00:00+00:00",
                        "event_type": "declined", "trading_days_ago": 1}
            return None

        from trading_agent.graph.nodes.premarket_nodes import (
            _dispatch_candidate_entry_if_eligible,
        )
        with patch("trading_agent.graph.nodes.premarket_nodes.emit",
                   side_effect=lambda **kw: emitted.append(kw)):
            with patch("trading_agent.market_calendar.is_us_market_open", return_value=True):
                with patch("pathlib.Path.exists", return_value=False):
                    with patch("trading_agent.learning.soak.is_new_entry_allowed",
                               return_value=True):
                        with patch("trading_agent.graph.nodes.premarket_nodes._open_position_count",
                                   return_value=0):
                            with patch("trading_agent.graph.nodes.premarket_nodes._existing_exposure",
                                       return_value=None):
                                with patch("trading_agent.graph.nodes.premarket_nodes._in_decline_cooldown",
                                           side_effect=_decline):
                                    with patch("trading_agent.graph.nodes.premarket_nodes._in_dispatch_cooldown",
                                               return_value=None):
                                        with patch("trading_agent.graph.nodes.premarket_nodes._recently_dispatched",
                                                   return_value=False):
                                            with patch("subprocess.run", side_effect=_fake_run):
                                                _dispatch_candidate_entry_if_eligible(
                                                    state, state["candidates"],
                                                    state["regime"],
                                                )
        assert started == ["trading-agent-candidate-entry@PLTR.service"], (
            f"expected fallthrough to PLTR, got {started}"
        )
        skips = [e for e in emitted if e.get("event_type") == "candidate_skipped_cooldown"]
        assert len(skips) == 1
        assert skips[0]["payload"]["ticker"] == "NVDA"
        assert skips[0]["payload"]["declined_at"] == "2026-06-09T15:00:00+00:00"


# ---------------------------------------------------------------------------
# Dispatch guard ordering + empty-scan decision events
# ---------------------------------------------------------------------------

class TestDispatchGuardOrdering:
    """The regular-hours gate must run BEFORE halt/soak/regime, and an empty
    in-session scan must still emit a dispatch decision — both keep the
    scan_dispatch_liveness outside_regular_hours exclusion sound: every
    non-outside_regular_hours skip provably comes from an in-session scan."""

    def _run(self, candidates, market_open, halt=False):
        state = _base_state(
            candidates=candidates,
            regime={"label": "BULL_TREND", "gate": {"allow_new_entries": True}},
        )
        emitted: list[dict] = []
        from trading_agent.graph.nodes.premarket_nodes import (
            _dispatch_candidate_entry_if_eligible,
        )
        with patch("trading_agent.graph.nodes.premarket_nodes.emit",
                   side_effect=lambda **kw: emitted.append(kw)):
            with patch("trading_agent.market_calendar.is_us_market_open",
                       return_value=market_open):
                with patch("pathlib.Path.exists", return_value=halt):
                    _dispatch_candidate_entry_if_eligible(
                        state, state["candidates"], state["regime"])
        return emitted

    def test_market_closed_with_halt_flag_reads_outside_regular_hours(self):
        """Market closed + halt flag set → the 08:30 digest scan must emit
        outside_regular_hours (which liveness excludes), NOT halt_flag_set
        (which would count as an executing-scan decision and mask two dead
        executing scans during a halted stretch)."""
        emitted = self._run(
            candidates=[{"ticker": "NVDA", "score": 0.9, "reason": "x"}],
            market_open=False, halt=True,
        )
        skips = [e for e in emitted if e["event_type"] == "candidate_entry_skipped"]
        assert len(skips) == 1
        assert skips[0]["payload"]["reason"] == "outside_regular_hours"

    def test_market_open_with_halt_flag_still_reads_halt(self):
        """Dispatch outcome unchanged by the re-order: in-session + halted
        still skips with halt_flag_set."""
        emitted = self._run(
            candidates=[{"ticker": "NVDA", "score": 0.9, "reason": "x"}],
            market_open=True, halt=True,
        )
        skips = [e for e in emitted if e["event_type"] == "candidate_entry_skipped"]
        assert len(skips) == 1
        assert skips[0]["payload"]["reason"] == "halt_flag_set"

    def test_empty_in_session_scan_emits_no_candidates(self):
        """Zero candidates during regular hours → emit a no_candidates skip
        (a decision event) instead of returning silently, so two completed-
        but-empty executing scans don't trip a false 17:00 ET liveness
        alert on a merely quiet day."""
        emitted = self._run(candidates=[], market_open=True)
        skips = [e for e in emitted if e["event_type"] == "candidate_entry_skipped"]
        assert len(skips) == 1
        assert skips[0]["payload"]["reason"] == "no_candidates"

    def test_empty_closed_scan_reads_outside_regular_hours(self):
        """Zero candidates while the market is closed → the (excluded)
        outside_regular_hours reason, not no_candidates — the 08:30 digest
        must never count toward executing-scan liveness."""
        emitted = self._run(candidates=[], market_open=False)
        skips = [e for e in emitted if e["event_type"] == "candidate_entry_skipped"]
        assert len(skips) == 1
        assert skips[0]["payload"]["reason"] == "outside_regular_hours"


# ---------------------------------------------------------------------------
# Step 5 — scan_dispatch_liveness tripwire
# ---------------------------------------------------------------------------

class TestScanDispatchLiveness:
    # Tue 2026-06-02 22:30 UTC = 18:30 ET — trading day, past the 17:00 cutoff
    LATE_TRADING_DAY = datetime(2026, 6, 2, 22, 30, tzinfo=timezone.utc)
    # Sat 2026-06-06 22:30 UTC = 18:30 ET Saturday
    WEEKEND = datetime(2026, 6, 6, 22, 30, tzinfo=timezone.utc)
    # Tue 2026-06-02 18:00 UTC = 14:00 ET — trading day, before the cutoff
    EARLY_TRADING_DAY = datetime(2026, 6, 2, 18, 0, tzinfo=timezone.utc)

    def _run(self, now, n_decisions=None, alerted_recently=False):
        from trading_agent.graph.nodes.health_nodes import scan_dispatch_liveness
        emitted: list[dict] = []
        alerts: list[dict] = []
        ctx = _fake_cursor_ctx(
            fetchone=(n_decisions,) if n_decisions is not None else None)
        with patch("trading_agent.store.postgres.cursor", return_value=ctx):
            with patch("trading_agent.graph.nodes.health_nodes.emit",
                       side_effect=lambda **kw: emitted.append(kw)):
                with patch("trading_agent.graph.nodes.health_nodes._alert_ops",
                           side_effect=lambda **kw: alerts.append(kw)):
                    with patch("trading_agent.graph.nodes.health_nodes."
                               "_was_scan_silence_alerted_recently",
                               return_value=alerted_recently):
                        scan_dispatch_liveness(
                            {"run_id": "t", "trigger": "healthcheck"}, now_utc=now)
        return emitted, alerts

    def test_zero_decisions_after_cutoff_alerts(self):
        emitted, alerts = self._run(self.LATE_TRADING_DAY, n_decisions=0)
        silent = [e for e in emitted if e["event_type"] == "scan_dispatch_silent"]
        assert len(silent) == 1
        assert silent[0]["severity"] == 2
        assert len(alerts) == 1
        assert "NO executing scan" in alerts[0]["title"]

    def test_weekend_no_alert(self):
        emitted, alerts = self._run(self.WEEKEND, n_decisions=0)
        assert emitted == []
        assert alerts == []

    def test_before_cutoff_no_alert(self):
        emitted, alerts = self._run(self.EARLY_TRADING_DAY, n_decisions=0)
        assert emitted == []
        assert alerts == []

    def test_decisions_present_emits_alive(self):
        emitted, alerts = self._run(self.LATE_TRADING_DAY, n_decisions=3)
        alive = [e for e in emitted if e["event_type"] == "scan_dispatch_alive"]
        assert len(alive) == 1
        assert alive[0]["payload"]["n_decisions_today"] == 3
        assert alerts == []

    def test_cooldown_suppresses_repeat_ntfy_but_still_emits(self):
        emitted, alerts = self._run(
            self.LATE_TRADING_DAY, n_decisions=0, alerted_recently=True)
        silent = [e for e in emitted if e["event_type"] == "scan_dispatch_silent"]
        assert len(silent) == 1
        assert silent[0]["payload"]["ntfy_suppressed"] is True
        assert alerts == []

    def test_liveness_node_wired_into_healthcheck_graph(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch("trading_agent.graph.builder.get_saver", MemorySaver):
            from trading_agent.graph.builder import build_healthcheck_graph
            g = build_healthcheck_graph()
        assert "scan_dispatch_liveness" in g.get_graph().nodes
