"""Item 9 — strategy spec registry: integrity, breakeven arithmetic, label
mapping, the per-spec R7 floor in sizing, and the journal's declared-vs-
realized spec_comparison block."""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from unittest.mock import patch

import pytest

from trading_agent.execution_costs import fees_per_side, half_spread_cost
from trading_agent.strategy_specs import (
    REGISTRY,
    breakeven_wr_gross,
    breakeven_wr_net,
    round_trip_friction_r,
    spec_band_violations,
    spec_for_label,
    spec_trading_block,
)

# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------


def test_registry_has_exactly_the_three_declared_specs():
    assert set(REGISTRY) == {
        "convexity_long_premium",
        "credit_put_spread_30_45",
        "credit_vertical_index_30_45",
    }
    # Interim convexity retirement (operator-approved 2026-07-20,
    # docs/REVIVAL_PLAN_2026-07-20.md sleeve 3): shadow-only, zero capital.
    assert REGISTRY["convexity_long_premium"].status == "shadow_only"
    assert not REGISTRY["convexity_long_premium"].is_tradeable
    # area A: unblocked once the atomic multi-leg combo path (place_paper_option_
    # combo + R5e) landed; defined-risk verticals are now tradeable.
    assert REGISTRY["credit_put_spread_30_45"].status == "active"
    assert REGISTRY["credit_put_spread_30_45"].is_tradeable
    # Sleeve 1 (M1-1): declared but blocked until every M1-0 prereq is green.
    assert REGISTRY["credit_vertical_index_30_45"].status == "pending_prereqs"
    assert not REGISTRY["credit_vertical_index_30_45"].is_tradeable


def test_every_spec_breakeven_net_above_gross():
    """Friction only ever raises the bar — a spec whose net breakeven is not
    strictly above its gross breakeven means the cost model was bypassed.
    credit_vertical_index_30_45 deliberately has NO profile yet — its
    expectancy must come from the M1-0.4 managed-payoff replay, not from
    the expiry-binary formula the plan rejected."""
    for spec in REGISTRY.values():
        p = spec.expectancy_profile
        if spec.name == "credit_vertical_index_30_45":
            assert p is None
            continue
        assert p is not None, spec.name
        assert p.breakeven_wr_net > p.breakeven_wr_gross, spec.name


def test_every_spec_declares_falsification_and_eval_n():
    for spec in REGISTRY.values():
        assert spec.falsification, spec.name
        assert spec.min_trades_for_eval >= 20, spec.name
        if spec.expectancy_profile is not None:
            lo, hi = spec.expectancy_profile.expected_wr_range
            assert 0.0 < lo < hi < 1.0, spec.name


def test_credit_vertical_falsification_quotes_three_tier_contract():
    """The M1-3 revised contract must be in the spec text, with every number
    marked placeholder pending the M1-0.4 replay."""
    f = REGISTRY["credit_vertical_index_30_45"].falsification
    assert "n=30" in f and "n=60" in f
    assert "LB95(mean R) < -0.10R" in f
    assert "block-bootstrap" in f
    assert "97.5%" in f
    assert "PLACEHOLDER" in f


def test_credit_vertical_m1_1_gates():
    g = REGISTRY["credit_vertical_index_30_45"].entry_gates
    assert g["underlying_whitelist"] == ("SPY", "QQQ")
    assert g["dte_range"] == (30, 45)
    assert g["abs_delta_range"] == (0.20, 0.35)
    assert g["min_credit_frac_of_width"] == 0.25
    assert g["max_spread_pct_mid"] == 0.05
    assert g["news_veto_required"] is True
    # The plan deletes min_risk_reward as redundant with the credit gate.
    assert "min_risk_reward" not in g
    assert REGISTRY["credit_vertical_index_30_45"].allowed_regimes == (
        "BULL_TREND", "RANGE_LOW_VOL")


def test_convexity_min_rr_is_respecced_above_global():
    """The convexity track demands 2:1, NOT the global 1.3 — at a declared
    30-45% WR, 1.3:1 winners cannot pay for the losers."""
    from trading_agent.llm.schemas import MIN_RISK_REWARD
    spec = REGISTRY["convexity_long_premium"]
    assert spec.min_risk_reward == 2.0
    assert spec.min_risk_reward > MIN_RISK_REWARD


def test_convexity_declared_range_contains_profitable_territory():
    """The declared WR envelope must reach above the net breakeven —
    otherwise the spec declares a strategy that cannot make money."""
    p = REGISTRY["convexity_long_premium"].expectancy_profile
    lo, hi = p.expected_wr_range
    assert hi > p.breakeven_wr_net


# ---------------------------------------------------------------------------
# Breakeven arithmetic — functions, not constants
# ---------------------------------------------------------------------------


def test_breakeven_gross_matches_cited_anchor():
    # The R7 1.3 floor's 43.5% gross breakeven, cited in execution_costs.
    assert breakeven_wr_gross(1.3) == pytest.approx(1.0 / 2.3)
    assert breakeven_wr_gross(2.0) == pytest.approx(1.0 / 3.0)
    with pytest.raises(ValueError):
        breakeven_wr_gross(0.0)


def test_breakeven_net_formula_and_monotonicity():
    # p = (1 + f) / (1 + rr); zero friction degenerates to gross.
    assert breakeven_wr_net(1.3, 0.18) == pytest.approx(1.18 / 2.3)
    assert breakeven_wr_net(2.0, 0.0) == pytest.approx(breakeven_wr_gross(2.0))
    assert (breakeven_wr_net(2.0, 0.2) > breakeven_wr_net(2.0, 0.1)
            > breakeven_wr_gross(2.0))


def test_vertical_spec_friction_comes_from_the_vertical_cost_function():
    """A vertical's friction must be priced from (width, credit) through
    execution_costs.friction_r — which resolves BOTH leg marks — not from
    round_trip_friction_r with a hand-typed per-leg premium.

    Regression for the 2026-07-25 friction-truth fix: the old derivation
    guessed $1.50 for both legs (leg-mid-sum $3.00 against a $1.43 credit,
    ratio 2.1) where real vertical quotes measure 6.43 on a $5-wide.
    """
    from trading_agent.execution_costs import friction_r as ec_friction_r
    spec = REGISTRY["credit_put_spread_30_45"]
    min_rr = spec.min_risk_reward
    width = 5.0
    credit = width * min_rr / (1.0 + min_rr)
    p = spec.expectancy_profile
    assert p.breakeven_wr_net == pytest.approx(
        breakeven_wr_net(min_rr, ec_friction_r(None, width, credit)))
    # The old (understated) derivation must not survive.
    old = round_trip_friction_r(typical_premium=1.50,
                                risk_per_unit=(width - credit) * 100.0,
                                n_option_legs=2)
    assert p.breakeven_wr_net > breakeven_wr_net(min_rr, old)


def test_vertical_spec_breakeven_now_exceeds_its_declared_wr_envelope():
    """HONEST CONSEQUENCE, pinned deliberately (2026-07-25).

    With the friction bill computed off real per-leg marks, a $5-wide
    single-name vertical at the global uncalibrated 4%-of-mark half-spread
    pays ~0.22R round trip, which puts breakeven_wr_net ABOVE the spec's own
    declared 70-80% envelope: as specced and as costed, this structure cannot
    make money. That is the spec system doing its job, not a number to tune
    away — resolving it (wider widths, tighter names, a measured single-name
    wing ratio, or retirement) is the operator's call. If the spec or the
    calibration is deliberately changed, this test SHOULD fail and be
    updated with the reasoning.
    """
    p = REGISTRY["credit_put_spread_30_45"].expectancy_profile
    lo, hi = p.expected_wr_range
    assert p.breakeven_wr_net > hi
    assert p.breakeven_wr_net == pytest.approx(0.8694, abs=1e-3)


def test_index_vertical_credit_floor_is_never_auto_tuned_from_measurements(
        tmp_path, monkeypatch):
    """A3: the 0.25 credit floor is an UNRESOLVED PLACEHOLDER.

    Real IWM quotes measure credit/width medians of 0.191 ($5-wide) and
    0.176 ($10-wide) — below the floor. Nothing may read a measured
    credit_frac_median out of the calibration and quietly relax the gate to
    make trades appear (the plan's "second better-documented zero-order
    funnel"). The floor stays where it is and the spec stays untradeable
    until the operator resolves it on SPY/QQQ data.
    """
    import dataclasses
    import json

    import trading_agent.strategy_specs as ss
    from trading_agent import config as config_mod
    from trading_agent import execution_costs as ec
    (tmp_path / "execution_costs.json").write_text(json.dumps({
        "per_underlying": {"IWM": {"median_spread_pct_mid": 0.046,
                                   "credit_frac_median": 0.191,
                                   "wing_ratio": 0.731}}}))
    monkeypatch.setattr(config_mod, "CONFIG",
                        dataclasses.replace(config_mod.CONFIG, data_dir=tmp_path))
    ec.reset_calibration_cache()
    try:
        # rebuild the spec WITH the measured calibration loaded
        spec = ss._credit_vertical_index_spec()
        assert spec.entry_gates["min_credit_frac_of_width"] == 0.25
        assert spec.status == "pending_prereqs" and not spec.is_tradeable
        assert spec.expectancy_profile is None
        assert spec == ss.REGISTRY["credit_vertical_index_30_45"]
    finally:
        ec.reset_calibration_cache()


def test_friction_r_is_derived_from_cost_model_not_hardcoded():
    """round_trip_friction_r must agree with the execution-cost model's own
    primitives, whatever calibration is loaded — fees both sides per leg
    plus half-spread both sides per leg, over the 1R denominator."""
    expected = (2.0 * fees_per_side(1, "OPT")
                + 2.0 * half_spread_cost(2.0, 1, "OPT")) / 100.0
    got = round_trip_friction_r(typical_premium=2.0, risk_per_unit=100.0)
    assert got == pytest.approx(expected)
    # two-leg vertical doubles both components
    got2 = round_trip_friction_r(
        typical_premium=2.0, risk_per_unit=100.0, n_option_legs=2)
    assert got2 == pytest.approx(2.0 * expected)
    with pytest.raises(ValueError):
        round_trip_friction_r(typical_premium=2.0, risk_per_unit=0.0)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", [
    "directional_long_call",
    "directional_long_put",
    "earnings_iv_drop",
    "earnings_iv_drop_post_print",
    "pullback_reversal",
    "breakout_squeeze",
    "Directional_Long_Call",   # case-insensitive
])
def test_long_premium_families_map_to_convexity(label):
    spec = spec_for_label(label)
    assert spec is not None and spec.name == "convexity_long_premium"


def test_exact_spec_name_resolves_directly():
    spec = spec_for_label("credit_put_spread_30_45")
    assert spec is not None and spec.name == "credit_put_spread_30_45"


@pytest.mark.parametrize("label", [
    None, "", "mean_reversion_pairs", "trend",
    # earnings_ alone is NOT enough — only the iv_drop family is long
    # premium; an earnings_straddle would be a different (undeclared) spec.
    "earnings_straddle",
])
def test_unknown_labels_are_legacy(label):
    assert spec_for_label(label) is None


@pytest.mark.parametrize("label", [
    "credit_vertical_index_30_45",
    "credit_vertical_spy_put",
    "Credit_Vertical_QQQ",
])
def test_credit_vertical_labels_map_to_index_spec(label):
    spec = spec_for_label(label)
    assert spec is not None and spec.name == "credit_vertical_index_30_45"


# ---------------------------------------------------------------------------
# Status enforcement — spec_trading_block
# ---------------------------------------------------------------------------


def test_trading_block_on_shadow_only_convexity_labels():
    block = spec_trading_block("directional_long_call")
    assert block is not None
    assert block["spec"] == "convexity_long_premium"
    assert block["status"] == "shadow_only"
    assert "shadow book" in block["message"]


def test_trading_block_on_pending_prereqs_vertical_labels():
    block = spec_trading_block("credit_vertical_spy_put")
    assert block is not None
    assert block["spec"] == "credit_vertical_index_30_45"
    assert block["status"] == "pending_prereqs"


def test_no_trading_block_for_active_or_unmapped_labels():
    # Active spec — tradeable.
    assert spec_trading_block("credit_put_spread_30_45") is None
    # Unmapped/legacy labels are governed by R5/R7, not spec status.
    assert spec_trading_block("mean_reversion_pairs") is None
    assert spec_trading_block(None) is None


# ---------------------------------------------------------------------------
# spec_band_violations — schema-level band enforcement (Week-1 Step 6b)
# ---------------------------------------------------------------------------


def test_band_mapped_label_tightens_to_spec_dte():
    """DTE 50 is inside the global R5 band (14–60) but outside convexity's
    declared 21–45 — the spec's inner band must bind for mapped labels."""
    v = spec_band_violations(
        strategy_label="directional_long_call", asset_type="OPT",
        option_dte=50, option_delta=0.45,
    )
    assert [x["band"] for x in v] == ["dte_range"]
    assert v[0]["spec"] == "convexity_long_premium"
    assert v[0]["bounds"] == [21.0, 45.0]


def test_band_mapped_label_tightens_to_spec_delta():
    """|delta| 0.60 passes R5's 0.25–0.65 but not convexity's 0.30–0.55."""
    v = spec_band_violations(
        strategy_label="pullback_reversal", asset_type="OPT",
        option_dte=30, option_delta=0.60,
    )
    assert [x["band"] for x in v] == ["abs_delta_range"]
    assert v[0]["spec"] == "convexity_long_premium"
    assert v[0]["bounds"] == [0.30, 0.55]


def test_band_mapped_label_inside_spec_passes():
    assert spec_band_violations(
        strategy_label="directional_long_call", asset_type="OPT",
        option_dte=30, option_delta=-0.45,   # abs() — puts pass too
    ) == []


def test_band_unmapped_label_falls_back_to_global_r5():
    """CRNX 2026-07-08 regression shape: an UNMAPPED legacy label is not a
    free pass — delta 0.815 dies against the global R5 band (0.25–0.65)."""
    v = spec_band_violations(
        strategy_label="momentum-continuation-ITM-call", asset_type="OPT",
        option_dte=44, option_delta=0.815,
    )
    assert [x["band"] for x in v] == ["abs_delta_range"]
    assert v[0]["spec"] == "global_r5"
    assert v[0]["bounds"] == [0.25, 0.65]
    assert v[0]["value"] == pytest.approx(0.815)


def test_band_unmapped_label_inside_global_band_passes():
    assert spec_band_violations(
        strategy_label="momentum-continuation-ITM-call", asset_type="OPT",
        option_dte=44, option_delta=0.60,
    ) == []


def test_band_spec_never_relaxes_below_global():
    """credit_put_spread declares 0.20–0.30, but R5's 0.25 floor wins: the
    effective band is the intersection [0.25, 0.30] — specs only tighten."""
    v = spec_band_violations(
        strategy_label="credit_put_spread_30_45", asset_type="OPT",
        option_dte=35, option_delta=0.22,
    )
    assert [x["band"] for x in v] == ["abs_delta_range"]
    assert v[0]["bounds"] == [0.25, 0.30]


def test_band_stock_and_missing_greeks_skip():
    # STK proposals have no option bands
    assert spec_band_violations(
        strategy_label="trend", asset_type="STK",
        option_dte=None, option_delta=None,
    ) == []
    # Missing dte/delta skip only that band (matches sizing R5 semantics)
    assert spec_band_violations(
        strategy_label="momentum-continuation-ITM-call", asset_type="OPT",
        option_dte=None, option_delta=None,
    ) == []
    v = spec_band_violations(
        strategy_label="momentum-continuation-ITM-call", asset_type="OPT",
        option_dte=None, option_delta=0.815,
    )
    assert [x["band"] for x in v] == ["abs_delta_range"]


# ---------------------------------------------------------------------------
# R7 per-spec floor in sizing
# ---------------------------------------------------------------------------

from trading_agent.sizing import (  # noqa: E402
    ProposedTrade,
    R7,
    SizingContext,
    check,
)


def _ctx() -> SizingContext:
    return SizingContext(equity=100_000.0, opens=(), sector_lookup_available=True)


def _trade(label: str, target: float = 13.0) -> ProposedTrade:
    """entry=10, stop=8 → risk $2/sh; target 13 → R:R 1.5."""
    return ProposedTrade(
        ticker="AAPL", asset_type="STK", side="BUY",
        qty=10, entry_price=10.0, stop=8.0, target=target,
        strategy_label=label,
    )


def test_r7_spec_floor_blocks_mapped_label_at_15():
    """R:R 1.5 clears the global 1.3 but NOT the convexity spec's 2.0 —
    a mapped label is held to its declared floor."""
    vs = [v for v in check(_ctx(), _trade("directional_long_call"))
          if v.rule == R7]
    assert len(vs) == 1 and vs[0].severity == "block"
    assert "convexity_long_premium" in vs[0].message  # names the spec
    assert "2.0" in vs[0].message


def test_r7_global_floor_still_governs_unmapped_label_at_15():
    """Same geometry under a legacy label passes at the global 1.3 floor."""
    vs = check(_ctx(), _trade("trend"))
    assert not any(v.rule == R7 for v in vs)


def test_r7_spec_floor_accepts_exactly_20():
    # entry=10, stop=8, target=14 → R:R exactly 2.0
    vs = check(_ctx(), _trade("directional_long_call", target=14.0))
    assert not any(v.rule == R7 for v in vs)


def test_r7_registry_failure_degrades_to_global_floor():
    """sizing runs inside the pretool hook + moomoo guard: a registry bug
    must degrade to the global floor, never crash or block validation."""
    with patch("trading_agent.strategy_specs.spec_for_label",
               side_effect=RuntimeError("registry exploded")):
        vs = check(_ctx(), _trade("directional_long_call"))
    # 1.5 >= global 1.3 → no R7 violation; and no exception escaped
    assert not any(v.rule == R7 for v in vs)


def test_r7_spec_floor_never_relaxes_below_global():
    """The blocked credit spread spec declares min_rr 0.40 — if that label
    somehow reaches a LONG open, the global 1.3 must still govern."""
    t = _trade("credit_put_spread_30_45", target=12.4)  # R:R 1.2 < 1.3
    vs = [v for v in check(_ctx(), t) if v.rule == R7]
    assert len(vs) == 1 and vs[0].severity == "block"
    assert "global floor" in vs[0].message


# ---------------------------------------------------------------------------
# Spec status gate in sizing — opens refused, closes exempt
# ---------------------------------------------------------------------------

from trading_agent.sizing import R_SPEC_STATUS  # noqa: E402


def _opt_trade(label: str | None, intent: str = "open",
               side: str = "BUY") -> ProposedTrade:
    """In-band long-premium option shape (dte 30, |delta| 0.45)."""
    return ProposedTrade(
        ticker="AAPL", asset_type="OPT", side=side,  # type: ignore[arg-type]
        qty=1, entry_price=1.0, stop=0.5, target=3.0,
        strategy_label=label, delta=0.45, dte=30,
        intent=intent,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("label", [
    "directional_long_call", "pullback_reversal", "earnings_iv_drop",
    "convexity_long_premium",
])
def test_spec_status_blocks_convexity_opens(label):
    """Interim retirement (2026-07-20): no NEW real long-premium fills —
    whichever path built the order, the guard-of-record refuses it."""
    vs = [v for v in check(_ctx(), _opt_trade(label)) if v.rule == R_SPEC_STATUS]
    assert len(vs) == 1 and vs[0].severity == "block"
    assert "shadow_only" in vs[0].message


@pytest.mark.parametrize("label", [
    "directional_long_call", "credit_vertical_spy_put",
    "mean_reversion_pairs",  # unmapped — the structure fallback must not
    None,                    # gate closes either (exit engine passes no label)
])
def test_spec_status_never_blocks_closes(label):
    """Retirement must never strand an open position: SELL-to-close of a
    legacy convexity position (or any non-active/unmapped/absent label)
    stays allowed."""
    vs = check(_ctx(), _opt_trade(label, intent="close", side="SELL"))
    assert not any(v.rule == R_SPEC_STATUS for v in vs)
    assert not any(v.severity == "block" for v in vs)


@pytest.mark.parametrize("label", [
    "mean_reversion_pairs", "momentum_long_call", "swing_call",
    "momentum-continuation-ITM-call",  # the 2026-07-08 CRNX label
    None,                              # bare tool call with no label at all
])
def test_spec_status_blocks_unmapped_opt_opens(label):
    """Fail closed on the STRUCTURE, not the label: single-leg SELL-to-open
    is hard-blocked, so an unmapped-label (or label-less) single-leg OPT
    open is exactly the retired long-premium structure — a free-text label
    the prefix table doesn't know must not bypass the retirement."""
    vs = [v for v in check(_ctx(), _opt_trade(label)) if v.rule == R_SPEC_STATUS]
    assert len(vs) == 1 and vs[0].severity == "block"
    assert "unmapped" in vs[0].message
    assert "shadow_only" in vs[0].message


def test_spec_status_allows_unmapped_stock_opens():
    """The structure fallback is OPT-only: unmapped-label STOCK opens stay
    governed by the global R1-R8 gates, not by any spec status."""
    assert not any(
        v.rule == R_SPEC_STATUS
        for v in check(_ctx(), _trade("mean_reversion_pairs")))


def test_single_leg_open_cannot_borrow_a_vertical_spec_label():
    """A SINGLE-LEG option open labeled as an active vertical spec is a
    structure mismatch, not an authorization: relabeling a long single-leg
    as a credit spec would reopen exactly the structure convexity's
    retirement closed (and the retry loop makes label-shopping a live
    path). Verticals open via place_paper_option_combo only."""
    for label in ("credit_put_spread_30_45", "credit_vertical_index_30_45"):
        vs = [v for v in check(_ctx(), _opt_trade(label))
              if v.rule == R_SPEC_STATUS]
        assert vs, f"{label} single-leg open must be blocked"
        assert "structure" in vs[0].message


def test_spec_status_allows_active_single_leg_mapped_opens():
    """A label mapping to an ACTIVE spec passes the status gate when the
    spec's structure actually IS single-leg (band / R5 / R7 gates still
    apply downstream). No such spec exists while convexity is retired, so
    pin the contract with a temporary registry entry."""
    from trading_agent import strategy_specs as ss
    active_single = dataclasses.replace(
        ss.REGISTRY["convexity_long_premium"], status="active")
    with patch.object(ss, "spec_for_label", return_value=active_single):
        assert not any(
            v.rule == R_SPEC_STATUS
            for v in check(_ctx(), _opt_trade("directional_long_call")))


def test_spec_status_registry_failure_degrades_open():
    """Same degradation contract as R7: a registry bug must not crash the
    pretool hook / moomoo guard — the R5/R7 gates still apply."""
    with patch("trading_agent.strategy_specs.spec_for_label",
               side_effect=RuntimeError("registry exploded")):
        vs = check(_ctx(), _opt_trade("directional_long_call"))
    assert not any(v.rule == R_SPEC_STATUS for v in vs)


def test_spec_status_blocks_pending_prereqs_combo():
    """A credit_vertical_* labeled combo passes the R5e structural proof but
    must die on spec status until M1-0 is green; the area-A
    credit_put_spread label stays unaffected."""
    from trading_agent.sizing import ComboLeg, ProposedCombo, check_combo
    legs = (
        ComboLeg(option_symbol="US.SPY260918P00600000", side="SELL",
                 contracts=1, price=1.80, right="P", strike=600.0,
                 dte=38, delta=-0.25),
        ComboLeg(option_symbol="US.SPY260918P00595000", side="BUY",
                 contracts=1, price=1.10, right="P", strike=595.0,
                 dte=38, delta=-0.18),
    )
    blocked = check_combo(
        _ctx(), ProposedCombo(ticker="SPY", legs=legs,
                              strategy_label="credit_vertical_spy_put"))
    hits = [v for v in blocked if v.rule == R_SPEC_STATUS]
    assert len(hits) == 1 and hits[0].severity == "block"
    assert "pending_prereqs" in hits[0].message

    allowed = check_combo(
        _ctx(), ProposedCombo(ticker="SPY", legs=legs,
                              strategy_label="credit_put_spread_30_45"))
    assert not any(v.rule == R_SPEC_STATUS for v in allowed)


# ---------------------------------------------------------------------------
# Journal spec_comparison — declared vs realized
# ---------------------------------------------------------------------------


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


def _seed_closed_trades(label: str, pnls: list[float], *,
                        provenance: str = "agent") -> None:
    """Closed OPT trades: entry 2.0, stop 1.0, qty 1 → risk $100 → R=pnl/100."""
    from trading_agent.db import connection
    with connection() as conn:
        for pnl in pnls:
            conn.execute(
                "INSERT INTO trades (symbol, asset_type, strategy_label, side,"
                " qty, entry_price, exit_price, stop, target, pnl, outcome,"
                " opened_at, closed_at, provenance, is_paper)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                ("US.AAPL260717C00200000", "OPT", label, "BUY",
                 1, 2.0, 2.0 + pnl / 100.0, 1.0, 4.0, pnl,
                 "WIN" if pnl > 0 else "LOSS",
                 "2026-06-08T14:30:00+00:00", "2026-06-10T19:00:00+00:00",
                 provenance),
            )


def test_spec_comparison_insufficient_n_and_realized_stats(tmp_journal_db):
    _seed_closed_trades("directional_long_call", [250.0, -100.0, -100.0])
    from trading_agent.mcp_servers.journal.server import (
        generate_post_mortem_prompt,
    )
    out = generate_post_mortem_prompt("2026-06-01")
    sc = out["spec_comparison"]["directional_long_call"]
    assert sc["spec"] == "convexity_long_premium"
    assert sc["verdict_hint"] == "INSUFFICIENT_N"  # 3 < min_trades_for_eval 30
    r = sc["realized"]
    assert r["n"] == 3 and r["n_with_r"] == 3
    assert r["win_rate"] == pytest.approx(1 / 3, abs=1e-3)
    # R = pnl/100: (2.5 - 1.0 - 1.0)/3
    assert r["mean_r"] == pytest.approx(0.1667, abs=1e-3)
    assert r["avg_win"] == pytest.approx(250.0)
    assert r["avg_loss"] == pytest.approx(100.0)
    d = sc["declared"]
    assert d["expected_wr_range"] == [0.30, 0.45]
    assert d["min_risk_reward"] == 2.0
    assert 0 < d["breakeven_wr_net"] < 1


def test_spec_comparison_within_and_outside_declared(tmp_journal_db):
    # 30 trades at WR .40 → inside the declared 30-45% envelope
    _seed_closed_trades("pullback_reversal",
                        [250.0] * 12 + [-100.0] * 18)
    # 30 trades at WR .667 → ABOVE the envelope: also flagged (the structure
    # is not doing what was declared, even though it's winning)
    _seed_closed_trades("breakout_squeeze",
                        [250.0] * 20 + [-100.0] * 10)
    from trading_agent.mcp_servers.journal.server import (
        generate_post_mortem_prompt,
    )
    out = generate_post_mortem_prompt("2026-06-01")
    assert out["spec_comparison"]["pullback_reversal"]["verdict_hint"] == \
        "WITHIN_DECLARED"
    assert out["spec_comparison"]["breakout_squeeze"]["verdict_hint"] == \
        "OUTSIDE_DECLARED"


def test_spec_comparison_skips_legacy_labels_and_shadow_rows(tmp_journal_db):
    _seed_closed_trades("old_style_momo", [50.0, -50.0])       # no spec
    _seed_closed_trades("directional_long_call", [100.0],
                        provenance="virtual_backfill")          # shadow bucket
    from trading_agent.mcp_servers.journal.server import (
        generate_post_mortem_prompt,
    )
    out = generate_post_mortem_prompt("2026-06-01")
    assert out["spec_comparison"] == {}


def test_moratorium_instruction_rekeyed_to_expectancy(tmp_journal_db):
    from trading_agent.mcp_servers.journal.server import (
        generate_post_mortem_prompt,
    )
    instr = generate_post_mortem_prompt("2026-06-01")["instructions"]
    assert "win rate alone is NEVER a moratorium reason" in instr
    assert "LB95(mean R) < 0" in instr
    assert "<40% win rate" not in instr  # the old WR moratorium is gone
