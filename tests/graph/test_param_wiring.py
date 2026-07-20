"""ITEM 11 — live consumption of learning-resolver values in trade_nodes.

Pins the three consumers that turned the canary loop from an A/A test into
a real treatment:

  * deterministic_sizing — sizing_aggression soft caps tighten qty
  * regime_execution_gate — regime_size_multipliers tighten the multiplier
  * persist_trade_event — stop_distances supply the fallback stop for the
    default exit plan

and the two invariants every consumer shares: defaults preserved when no
resolver values are in state, and exactly ONE param_consumed audit event
per node invocation when a value actually alters the outcome.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_with_resolver(values: dict | None, **overrides) -> dict:
    learning = {}
    if values is not None:
        learning = {
            "params": {
                "version_id": 42, "status": "CANARY", "scope": "global",
                "scope_key": None, "values": values,
            },
            "params_version_id": 42,
        }
    s = {
        "run_id": "test-run-wiring",
        "trigger": "candidate_entry",
        "ts": "2026-06-12T00:00:00Z",
        "research": {},
        "candidates": [],
        "sizing": {},
        "order": {},
        "fill": {},
        "learning": learning,
        "errors": [],
    }
    s.update(overrides)
    return s


def _opt_proposal(qty: float = 10.0) -> dict:
    """Passes R1-R7 at qty=10 with $100k equity so only soft caps can bind:
    notional 10×$100=$1000 ≤ R5 $1500; max_loss $1000 ≤ R1 $2500; R:R 2.0."""
    return {
        "ticker": "SPY", "symbol": "US.SPY260619C00600000",
        "asset_type": "OPT", "direction": "LONG_CALL", "side": "BUY",
        "qty": qty, "entry_price": 1.0, "stop": 0.5, "target": 2.0,
        "strategy_label": "momentum_long_call",
        "option_delta": 0.45, "option_dte": 30,
    }


def _param_consumed_calls(mock_emit) -> list[dict]:
    return [
        c.kwargs for c in mock_emit.call_args_list
        if c.kwargs.get("event_type") == "param_consumed"
    ]


def _run_sizing(state):
    # spec_trading_block is patched off: since the 2026-07-20 convexity
    # retirement EVERY single-leg OPT open is R_spec_status-refused (the
    # structure fallback covers _opt_proposal's unmapped fixture label
    # too). These tests exercise the resolver soft-cap machinery UNDER the
    # status gate, which must keep working for a revived spec.
    with patch("trading_agent.graph.nodes.trade_nodes.emit",
               return_value=1) as mock_emit, \
         patch("trading_agent.learning.soak.tiny_paper_qty_cap",
               return_value=None), \
         patch("trading_agent.strategy_specs.spec_trading_block",
               return_value=None), \
         patch("trading_agent.sectors.known_count", return_value=10), \
         patch("trading_agent.sectors.lookup", return_value="Technology"):
        from trading_agent.graph.nodes.trade_nodes import deterministic_sizing
        result = deterministic_sizing(state)
    return result, mock_emit


# ---------------------------------------------------------------------------
# deterministic_sizing — sizing_aggression soft caps
# ---------------------------------------------------------------------------


class TestSizingSoftCaps:
    def test_r5_soft_notional_cap_tightens_qty(self):
        """resolver r5=0.3% → $300 notional budget → 3 of 10 contracts."""
        state = _state_with_resolver(
            {"r5_soft_notional_pct": 0.003},
            proposal=_opt_proposal(qty=10),
            account={"equity": 100_000.0}, positions=[],
        )
        result, mock_emit = _run_sizing(state)
        assert result["sizing"]["approved_qty"] == 3.0
        assert result["sizing"]["infeasible"] is False
        calls = _param_consumed_calls(mock_emit)
        assert len(calls) == 1  # one emit per node invocation
        effects = calls[0]["payload"]["effects"]
        assert effects[0]["key"] == "r5_soft_notional_pct"
        assert calls[0]["payload"]["params_version_id"] == 42

    def test_r1_soft_cap_tightens_qty(self):
        """resolver r1=0.5% → $500 risk budget / $100-per-contract debit → 5."""
        state = _state_with_resolver(
            {"r1_soft_cap_pct": 0.005},
            proposal=_opt_proposal(qty=10),
            account={"equity": 100_000.0}, positions=[],
        )
        result, mock_emit = _run_sizing(state)
        assert result["sizing"]["approved_qty"] == 5.0
        assert len(_param_consumed_calls(mock_emit)) == 1

    def test_r3_soft_cap_counts_existing_ticker_exposure(self):
        """resolver r3=5% → $5000 ticker budget; $4500 already open on SPY
        leaves $500 → 5 contracts of the new $100 notional."""
        state = _state_with_resolver(
            {"r3_ticker_exposure_pct": 0.05},
            proposal=_opt_proposal(qty=10),
            account={"equity": 100_000.0},
            positions=[{
                "symbol": "US.SPY", "asset_type": "STK", "qty": 30.0,
                "entry_price": 150.0, "stop": 140.0, "sector": "Technology",
            }],
        )
        result, _ = _run_sizing(state)
        assert result["sizing"]["approved_qty"] == 5.0

    def test_multiple_binding_keys_dedupe_to_one_emit(self):
        """r1 binds 10→5, then r5 binds 5→3: TWO effects, ONE emit."""
        state = _state_with_resolver(
            {"r1_soft_cap_pct": 0.005, "r5_soft_notional_pct": 0.003},
            proposal=_opt_proposal(qty=10),
            account={"equity": 100_000.0}, positions=[],
        )
        result, mock_emit = _run_sizing(state)
        assert result["sizing"]["approved_qty"] == 3.0
        calls = _param_consumed_calls(mock_emit)
        assert len(calls) == 1
        assert [e["key"] for e in calls[0]["payload"]["effects"]] == [
            "r1_soft_cap_pct", "r5_soft_notional_pct",
        ]

    def test_no_resolver_values_leaves_defaults_unchanged(self):
        """Empty learning state = pre-ITEM-11 behavior, byte for byte."""
        state = _state_with_resolver(
            None,
            proposal=_opt_proposal(qty=10),
            account={"equity": 100_000.0}, positions=[],
        )
        result, mock_emit = _run_sizing(state)
        assert result["sizing"]["approved_qty"] == 10.0
        assert _param_consumed_calls(mock_emit) == []

    def test_resolver_cannot_loosen_hard_caps(self):
        """min(hard, resolver): a corrupt value above the R5 hard cap must
        not raise the budget. qty=20 busts hard R5 ($2000 > $1500) and the
        candidate loop must still downsize, resolver or not — and since the
        resolver wasn't the binding constraint, NO param_consumed row."""
        state = _state_with_resolver(
            {"r5_soft_notional_pct": 0.10},  # 10× the hard cap
            proposal=_opt_proposal(qty=20),
            account={"equity": 100_000.0}, positions=[],
        )
        result, mock_emit = _run_sizing(state)
        # hard R5 allows 15; the coarse loop lands on 75% of 20 = 15
        assert result["sizing"]["approved_qty"] == 15.0
        assert _param_consumed_calls(mock_emit) == []

    def test_soft_cap_to_zero_is_infeasible_not_resurrected(self):
        """When even 1 contract busts the soft budget the node must return
        infeasible — the max(1, ...) candidate ladder previously would have
        resurrected qty=1 and silently undone the cap."""
        proposal = _opt_proposal(qty=10)
        proposal["entry_price"] = 5.0  # $500/contract > $300 soft budget
        proposal["stop"] = 2.5
        proposal["target"] = 10.0
        state = _state_with_resolver(
            {"r5_soft_notional_pct": 0.003},
            proposal=proposal,
            account={"equity": 100_000.0}, positions=[],
        )
        result, mock_emit = _run_sizing(state)
        assert result["sizing"]["infeasible"] is True
        assert result["sizing"]["approved_qty"] == 0.0
        assert result["proposal"]["qty"] == 0.0
        assert len(_param_consumed_calls(mock_emit)) == 1


# ---------------------------------------------------------------------------
# deterministic_sizing — min_dte entry filter (area E)
# ---------------------------------------------------------------------------


class TestMinDteEntryFilter:
    def test_short_dated_option_is_filtered(self):
        """resolver min_dte=18, option DTE 10 → entry filtered (infeasible),
        one param_consumed row keyed on min_dte."""
        p = _opt_proposal(qty=10)
        p["option_dte"] = 10
        state = _state_with_resolver(
            {"min_dte": 18}, proposal=p,
            account={"equity": 100_000.0}, positions=[],
        )
        result, mock_emit = _run_sizing(state)
        assert result["sizing"]["infeasible"] is True
        assert result["sizing"]["approved_qty"] == 0.0
        assert result["proposal"]["qty"] == 0.0
        assert result["sizing"]["r1_r6_violations"][0]["rule"] == "entry_min_dte"
        calls = _param_consumed_calls(mock_emit)
        assert len(calls) == 1
        assert calls[0]["payload"]["effects"][0]["key"] == "min_dte"

    def test_long_dated_option_passes(self):
        """resolver min_dte=18, option DTE 30 → not filtered; normal sizing
        approves all 10 and emits no min_dte param_consumed."""
        p = _opt_proposal(qty=10)
        p["option_dte"] = 30
        state = _state_with_resolver(
            {"min_dte": 18}, proposal=p,
            account={"equity": 100_000.0}, positions=[],
        )
        result, mock_emit = _run_sizing(state)
        assert result["sizing"]["infeasible"] is False
        assert result["sizing"]["approved_qty"] == 10.0
        assert _param_consumed_calls(mock_emit) == []

    def test_no_resolver_value_is_noop_even_for_short_dte(self):
        """Without a min_dte override the filter must not fire (byte-identical
        to the pre-learning path); the R5 hard floor still governs downstream."""
        p = _opt_proposal(qty=10)
        p["option_dte"] = 10
        state = _state_with_resolver(
            None, proposal=p,
            account={"equity": 100_000.0}, positions=[],
        )
        result, _ = _run_sizing(state)
        # not filtered by the learner — the entry_min_dte rule never fires
        viols = result["sizing"].get("r1_r6_violations", [])
        assert all(v.get("rule") != "entry_min_dte" for v in viols)

    def test_stock_proposal_is_exempt(self):
        """Stock proposals have no DTE → min_dte filter must not touch them."""
        p = _opt_proposal(qty=10)
        p["asset_type"] = "STK"
        p.pop("option_dte", None)
        state = _state_with_resolver(
            {"min_dte": 18}, proposal=p,
            account={"equity": 100_000.0}, positions=[],
        )
        result, _ = _run_sizing(state)
        viols = result["sizing"].get("r1_r6_violations", [])
        assert all(v.get("rule") != "entry_min_dte" for v in viols)


# ---------------------------------------------------------------------------
# regime_execution_gate — regime_size_multipliers
# ---------------------------------------------------------------------------


class TestRegimeSizeMultiplierWiring:
    def _run_gate(self, state):
        with patch("trading_agent.graph.nodes.trade_nodes.emit",
                   return_value=1) as mock_emit, \
             patch("trading_agent.learning.soak.is_new_entry_allowed",
                   return_value=True):
            from trading_agent.graph.nodes.trade_nodes import (
                regime_execution_gate,
            )
            result = regime_execution_gate(state)
        return result, mock_emit

    def test_resolver_tightens_multiplier(self):
        state = _state_with_resolver(
            {"size_mult_bull_trend": 0.5},
            proposal={"ticker": "SPY", "qty": 4, "asset_type": "OPT"},
            regime={"label": "BULL_TREND",
                    "gate": {"allow_new_entries": True, "size_multiplier": 1.0}},
        )
        result, mock_emit = self._run_gate(state)
        assert result["proposal"]["qty"] == 2.0
        calls = _param_consumed_calls(mock_emit)
        assert len(calls) == 1
        assert calls[0]["payload"]["effects"][0]["key"] == "size_mult_bull_trend"

    def test_resolver_cannot_loosen_gate_multiplier(self):
        """gate already at 0.5 (e.g. confidence blend); resolver 1.0 must NOT
        loosen it back up — min() and no param_consumed (nothing altered)."""
        state = _state_with_resolver(
            {"size_mult_bull_trend": 1.0},
            proposal={"ticker": "SPY", "qty": 4, "asset_type": "OPT"},
            regime={"label": "BULL_TREND",
                    "gate": {"allow_new_entries": True, "size_multiplier": 0.5}},
        )
        result, mock_emit = self._run_gate(state)
        assert result["proposal"]["qty"] == 2.0
        assert _param_consumed_calls(mock_emit) == []

    def test_no_resolver_keeps_gate_multiplier(self):
        state = _state_with_resolver(
            None,
            proposal={"ticker": "SPY", "qty": 4, "asset_type": "OPT"},
            regime={"label": "BULL_TREND",
                    "gate": {"allow_new_entries": True, "size_multiplier": 1.0}},
        )
        result, mock_emit = self._run_gate(state)
        assert result["proposal"]["qty"] == 4.0
        assert _param_consumed_calls(mock_emit) == []


# ---------------------------------------------------------------------------
# persist_trade_event — stop_distances fallback
# ---------------------------------------------------------------------------


def _fake_pg():
    cur = MagicMock()
    cur.fetchone.return_value = (123,)
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=cur)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx, cur


class TestStopDistanceWiring:
    def _run_persist(self, state):
        ctx, cur = _fake_pg()
        with patch("trading_agent.graph.nodes.trade_nodes.emit",
                   return_value=1) as mock_emit, \
             patch("trading_agent.store.postgres.cursor", return_value=ctx):
            from trading_agent.graph.nodes.trade_nodes import (
                persist_trade_event,
            )
            result = persist_trade_event(state)
        return result, mock_emit, cur

    @staticmethod
    def _exit_plan_param(cur) -> str | None:
        # exit_plan is the LAST parameter of the journal_trades INSERT
        return cur.execute.call_args[0][1][-1]

    def _stopless_state(self, values, *, asset_type="OPT"):
        proposal = {
            "ticker": "SPY", "symbol": "US.SPY260619C00600000",
            "asset_type": asset_type, "thesis_id": 7,
            "proposal_id": "prop_x", "stop": None, "target": 4.0,
        }
        return _state_with_resolver(
            values,
            proposal=proposal,
            order={"placed": True, "order_id": "o1",
                   "symbol": proposal["symbol"], "qty": 1, "limit_price": 2.0},
            fill={"status": "FILLED_ALL", "fill_qty": 1, "avg_fill_price": 2.0},
            risk={"risk_decision_id": "rd1"},
            regime={"state_id": 5, "label": "BULL_TREND"},
        )

    def test_option_premium_stop_pct_supplies_fallback_stop(self):
        """Stopless option proposal + resolver 0.40 → plan stop at 40% of
        the $2.00 entry premium = $0.80; differs from the 0.50 default
        ($1.00) → param_consumed."""
        state = self._stopless_state({"option_premium_stop_pct": 0.40})
        _, mock_emit, cur = self._run_persist(state)
        plan = json.loads(self._exit_plan_param(cur))
        assert plan["hard_stop"] == 0.8
        assert plan["hard_target"] == 4.0
        calls = _param_consumed_calls(mock_emit)
        assert len(calls) == 1
        assert calls[0]["payload"]["effects"][0]["key"] == "option_premium_stop_pct"

    def test_implicit_stop_frac_supplies_stock_fallback_stop(self):
        """Stopless stock proposal + resolver 0.08 → stop 8% below entry."""
        state = self._stopless_state({"implicit_stop_frac": 0.08},
                                     asset_type="STK")
        _, mock_emit, cur = self._run_persist(state)
        plan = json.loads(self._exit_plan_param(cur))
        assert abs(plan["hard_stop"] - 2.0 * 0.92) < 1e-9
        assert len(_param_consumed_calls(mock_emit)) == 1

    def test_default_valued_resolver_builds_plan_without_audit_noise(self):
        """Seeded-baseline resolver (value == code default): plan exists but
        nothing was ALTERED → no param_consumed row."""
        state = self._stopless_state({"option_premium_stop_pct": 0.50})
        _, mock_emit, cur = self._run_persist(state)
        plan = json.loads(self._exit_plan_param(cur))
        assert plan["hard_stop"] == 1.0
        assert _param_consumed_calls(mock_emit) == []

    def test_no_resolver_preserves_null_exit_plan(self):
        """Pre-ITEM-11 behavior: stopless proposal + empty learning state →
        exit_plan stays NULL (hard executor keeps its legacy fallback)."""
        state = self._stopless_state(None)
        _, mock_emit, cur = self._run_persist(state)
        assert self._exit_plan_param(cur) is None
        assert _param_consumed_calls(mock_emit) == []

    def test_explicit_trader_stop_is_never_overridden(self):
        """The resolver only supplies FALLBACK stops — an explicit stop on
        the proposal wins and emits nothing."""
        state = self._stopless_state({"option_premium_stop_pct": 0.40})
        state["proposal"]["stop"] = 1.5
        _, mock_emit, cur = self._run_persist(state)
        plan = json.loads(self._exit_plan_param(cur))
        assert plan["hard_stop"] == 1.5
        assert _param_consumed_calls(mock_emit) == []
