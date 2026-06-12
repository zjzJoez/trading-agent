"""ITEM 11 — paired counterfactual evaluation for qty-only canaries.

The traffic-split gates in promote.py can't evaluate qty-only families
efficiently (assignment noise, half the sample discarded); these tests pin
the replacement: exact paired replay over ALL closed trades, decision on
the 95% CI of the paired differences d_i = r_shadow_i − r_actual_i.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from trading_agent.learning.counterfactual import (
    N_PAIRED_MIN,
    CounterfactualDecision,
    _mean_ci_95,
    evaluate_qty_only,
    is_qty_only_params,
    maybe_evaluate_qty_only,
    paired_diffs,
)
from trading_agent.learning.params import ParamResolver, defaults_resolver
from trading_agent.learning.replay import ClosedTradeRow

_T0 = datetime(2026, 6, 1, tzinfo=UTC)


def _trade(i: int, *, realized_r: float | None, qty: int = 2) -> ClosedTradeRow:
    """Synthetic closed BULL_TREND option trade.

    Geometry chosen so the soft caps never bind at defaults (entry=1.0,
    stop=0.5 → max_loss/contract $50, notional/contract $100 vs default
    budgets $2000/$1000) — only the regime size multiplier moves qty."""
    return ClosedTradeRow(
        trade_id=i,
        ticker="US.SPY260619C00600000",
        strategy_label="momentum_long_call",
        entry_price=1.0,
        exit_price=1.5,
        stop=0.5,
        qty=qty,
        contract_multiplier=100,
        opened_at=_T0 + timedelta(hours=i),
        closed_at=_T0 + timedelta(hours=i + 4),
        params_version_id=1,
        entry_regime_label="BULL_TREND",
        entry_feature_snapshot=None,
        realized_r=realized_r,
        equity_at_entry=100_000.0,
    )


_HALVER = {"size_mult_bull_trend": 0.5}  # candidate halves size vs default 1.0


# -- family classification -----------------------------------------------

def test_is_qty_only_params():
    assert is_qty_only_params({"r1_soft_cap_pct": 0.01})
    assert is_qty_only_params({"size_mult_bull_trend": 0.8})
    # stop_distances is NOT qty-only — its effect needs the price path
    assert not is_qty_only_params({"implicit_stop_frac": 0.04})
    # mixed → not qty-only (the non-qty key needs the traffic split)
    assert not is_qty_only_params(
        {"r1_soft_cap_pct": 0.01, "implicit_stop_frac": 0.04}
    )
    # r3 is family-qty-only but replay can't model per-ticker exposure
    # netting (needs the rest of the portfolio at each historical entry) —
    # canaries touching it must take the traffic-split path.
    assert not is_qty_only_params({"r3_ticker_exposure_pct": 0.08})
    assert not is_qty_only_params(
        {"r5_soft_notional_pct": 0.005, "r3_ticker_exposure_pct": 0.08}
    )
    # empty / unrecognised-only → False (fall back, never auto-promote)
    assert not is_qty_only_params({})
    assert not is_qty_only_params({"phantom": 1.0})


# -- CI math ---------------------------------------------------------------

def test_mean_ci_95_empty_and_singleton_are_unbounded():
    m, lb, ub = _mean_ci_95([])
    assert lb == float("-inf") and ub == float("inf")
    m, lb, ub = _mean_ci_95([0.3])
    assert m == 0.3 and lb == float("-inf") and ub == float("inf")


def test_mean_ci_95_zero_variance_collapses_to_mean():
    """Identical candidate ⇒ d ≡ 0 ⇒ CI = [0,0] ⇒ RETAIN, never PROMOTE."""
    m, lb, ub = _mean_ci_95([0.0] * 30)
    assert m == lb == ub == 0.0
    m, lb, ub = _mean_ci_95([0.5] * 30)
    assert m == lb == ub == 0.5


def test_mean_ci_95_widens_with_variance():
    m, lb, ub = _mean_ci_95([1.0, -1.0] * 15)
    assert m == 0.0 and lb < 0 < ub


# -- paired replay ---------------------------------------------------------

def test_paired_diffs_halver_on_losers_is_positive():
    """Halving size on a losing trade halves the loss → d = +0.5R."""
    trades = [_trade(i, realized_r=-1.0) for i in range(5)]
    candidate = ParamResolver(
        version_id=9, status="CANARY", scope="global", scope_key=None,
        values=dict(_HALVER),
    )
    ds, n_skipped = paired_diffs(trades, candidate, defaults_resolver())
    assert n_skipped == 0
    assert ds == [0.5] * 5


def test_paired_diffs_excludes_missing_outcomes():
    """realized_r IS NULL means enrichment missing, NOT a 0R trade —
    zero-filling those pairs would bias the gate to RETAIN forever."""
    trades = [_trade(0, realized_r=-1.0), _trade(1, realized_r=None)]
    candidate = ParamResolver(
        version_id=9, status="CANARY", scope="global", scope_key=None,
        values=dict(_HALVER),
    )
    ds, n_skipped = paired_diffs(trades, candidate, defaults_resolver())
    assert len(ds) == 1 and n_skipped == 1


# -- decision rule -----------------------------------------------------------

def test_promote_when_halver_meets_losers_only_sample():
    """Candidate that halves size over a losers-only history: every paired
    diff is +0.5R → LB95 > 0 → PROMOTE."""
    trades = [_trade(i, realized_r=-1.0) for i in range(25)]
    d = evaluate_qty_only(
        canary_version_id=9, candidate_params=dict(_HALVER),
        baseline_version_id=None, since=None,
        trades=trades, baseline=defaults_resolver(),
    )
    assert d.decision == "PROMOTE"
    assert d.n_pairs == 25
    assert d.lb95 > 0
    assert any("paired counterfactual" in r for r in d.rationale)


def test_reject_when_halver_meets_winners_only_sample():
    """Reversed sample: halving size on winners halves the gains → every
    paired diff is −0.5R → UB95 < 0 → REJECT."""
    trades = [_trade(i, realized_r=1.0) for i in range(25)]
    d = evaluate_qty_only(
        canary_version_id=9, candidate_params=dict(_HALVER),
        baseline_version_id=None, since=None,
        trades=trades, baseline=defaults_resolver(),
    )
    assert d.decision == "REJECT"
    assert d.ub95 < 0


def test_retain_below_min_pairs_even_if_diffs_positive():
    trades = [_trade(i, realized_r=-1.0) for i in range(N_PAIRED_MIN - 1)]
    d = evaluate_qty_only(
        canary_version_id=9, candidate_params=dict(_HALVER),
        baseline_version_id=None, since=None,
        trades=trades, baseline=defaults_resolver(),
    )
    assert d.decision == "RETAIN"
    assert f"{N_PAIRED_MIN - 1}/{N_PAIRED_MIN}" in d.rationale[0]


def test_retain_when_ci_straddles_zero():
    trades = [
        _trade(i, realized_r=(1.0 if i % 2 == 0 else -1.0)) for i in range(24)
    ]
    d = evaluate_qty_only(
        canary_version_id=9, candidate_params=dict(_HALVER),
        baseline_version_id=None, since=None,
        trades=trades, baseline=defaults_resolver(),
    )
    assert d.decision == "RETAIN"
    assert d.lb95 < 0 < d.ub95


def test_identical_candidate_retains():
    """A/A by construction: candidate == defaults ⇒ d ≡ 0 ⇒ RETAIN."""
    trades = [_trade(i, realized_r=-1.0) for i in range(25)]
    d = evaluate_qty_only(
        canary_version_id=9,
        candidate_params={"size_mult_bull_trend": 1.0},
        baseline_version_id=None, since=None,
        trades=trades, baseline=defaults_resolver(),
    )
    assert d.decision == "RETAIN"
    assert d.mean_diff == 0.0


# -- dispatch ----------------------------------------------------------------

class _FakeCfDB:
    """Scripted cursor for maybe_evaluate_qty_only's two row fetches."""

    def __init__(self, dispatch_row, baseline_row=None):
        self.dispatch_row = dispatch_row
        self.baseline_row = baseline_row
        self._sql = ""

    def execute(self, sql, params=()):
        self._sql = sql

    def fetchone(self):
        if "params, created_at" in self._sql:
            return self.dispatch_row
        if "SELECT status, scope, scope_key, params" in self._sql:
            return self.baseline_row
        return None


def _patched_cf_db(db: _FakeCfDB):
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=db)
    ctx.__exit__ = MagicMock(return_value=False)
    return patch(
        "trading_agent.learning.counterfactual.cursor", return_value=ctx
    )


def test_maybe_evaluate_qty_only_full_path_promotes():
    db = _FakeCfDB(
        dispatch_row=("CANARY", 1, dict(_HALVER), _T0),
        baseline_row=("ACTIVE", "global", None, {}),
    )
    trades = [_trade(i, realized_r=-1.0) for i in range(25)]
    with _patched_cf_db(db), patch(
        "trading_agent.learning.counterfactual.fetch_closed_trades",
        return_value=trades,
    ) as mock_fetch:
        d = maybe_evaluate_qty_only(42)
    assert d is not None and d.decision == "PROMOTE"
    # must replay everything since the canary's created_at
    assert mock_fetch.call_args.kwargs["since"] == _T0


def test_maybe_evaluate_returns_none_for_non_qty_family():
    db = _FakeCfDB(
        dispatch_row=("CANARY", 1, {"implicit_stop_frac": 0.04}, _T0),
    )
    with _patched_cf_db(db):
        assert maybe_evaluate_qty_only(42) is None


def test_maybe_evaluate_returns_none_for_non_canary_status():
    db = _FakeCfDB(dispatch_row=("ACTIVE", None, dict(_HALVER), _T0))
    with _patched_cf_db(db):
        assert maybe_evaluate_qty_only(42) is None


def test_maybe_evaluate_fails_open_on_db_error():
    """Any read failure → None → promote.py's traffic-split path (which has
    its own fail-safe RETAIN gating) — never an exception up the EOD graph."""
    with patch(
        "trading_agent.learning.counterfactual.cursor",
        side_effect=RuntimeError("pg down"),
    ):
        assert maybe_evaluate_qty_only(42) is None


def _patched_promote_preamble(status: str = "CANARY"):
    """Patch promote.cursor (status read) + the hard-rollback triggers so a
    dispatch test reaches the counterfactual hand-off. The dispatch sits
    AFTER the sanity check and rollback triggers — they are routed-arm
    safety brakes that must run for every canary, qty-only included."""
    cur = MagicMock()
    cur.fetchone.return_value = (status, 1)
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=cur)
    ctx.__exit__ = MagicMock(return_value=False)
    return (
        patch("trading_agent.learning.promote.cursor", return_value=ctx),
        patch("trading_agent.learning.promote.first_n_trades_breach",
              return_value=False),
        patch("trading_agent.learning.promote.consecutive_losing_days",
              return_value=0),
    )


def test_promote_evaluate_canary_dispatches_qty_only():
    """evaluate_canary must return the counterfactual decision verbatim for
    qty-only canaries — no traffic-split stats."""
    from trading_agent.learning.promote import evaluate_canary

    cf = CounterfactualDecision(
        canary_version_id=42, decision="PROMOTE",
        rationale=["paired counterfactual: n=25 ..."], n_pairs=25,
        mean_diff=0.5, lb95=0.5, ub95=0.5,
    )
    p1, p2, p3 = _patched_promote_preamble()
    with p1, p2, p3, patch(
        "trading_agent.learning.promote.maybe_evaluate_qty_only",
        return_value=cf,
    ):
        d = evaluate_canary(42)
    assert d.decision == "PROMOTE"
    assert d.rationale == cf.rationale


def test_promote_evaluate_canary_reject_carries_rollback_reason():
    from trading_agent.learning.promote import evaluate_canary

    cf = CounterfactualDecision(
        canary_version_id=42, decision="REJECT",
        rationale=["..."], n_pairs=25, mean_diff=-0.5, lb95=-0.5, ub95=-0.5,
    )
    p1, p2, p3 = _patched_promote_preamble()
    with p1, p2, p3, patch(
        "trading_agent.learning.promote.maybe_evaluate_qty_only",
        return_value=cf,
    ):
        d = evaluate_canary(42)
    assert d.decision == "REJECT"
    assert d.rollback_reason == "counterfactual_ub95_negative"


def test_promote_hard_rollback_precedes_counterfactual_dispatch():
    """A qty-only canary with a first-5 hard-risk breach must be REJECTED by
    the rollback trigger BEFORE the counterfactual can say PROMOTE — the
    canary routes live traffic, so the safety brakes apply regardless of
    which statistic decides promotion."""
    from trading_agent.learning.promote import evaluate_canary

    cur = MagicMock()
    cur.fetchone.return_value = ("CANARY", 1)
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=cur)
    ctx.__exit__ = MagicMock(return_value=False)
    with patch("trading_agent.learning.promote.cursor", return_value=ctx), \
         patch("trading_agent.learning.promote.first_n_trades_breach",
               return_value=True), \
         patch("trading_agent.learning.promote.maybe_evaluate_qty_only") as cf_mock:
        d = evaluate_canary(42)
    assert d.decision == "REJECT"
    assert d.rollback_reason == "first5_breach"
    cf_mock.assert_not_called()
