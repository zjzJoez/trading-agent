"""Branch coverage for the deterministic exit executor.

One test per priority-level branch in hard_exit_decision, plus parsing
helpers (DTE + intrinsic). When we add a new rule to the executor,
add a parallel test here first.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading_agent.exits import (
    ExitDecision,
    compute_dte_from_symbol,
    compute_intrinsic,
    hard_exit_decision,
)
from trading_agent.exits.hard_executor import load_exit_plan
from trading_agent.llm.schemas import (
    DteRulesConfig,
    ExitPlan,
    RegimeRulesConfig,
    ScaleOutRung,
    TrailStopConfig,
    default_exit_plan,
)


# ---------------------------------------------------------------------------
# Symbol parsing
# ---------------------------------------------------------------------------


class TestComputeDte:
    def test_moomoo_format(self):
        now = datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)
        # SPY 5/29/2026 call → 7 days from 5/22
        dte = compute_dte_from_symbol("US.SPY260529C00742000", now)
        assert dte == 7

    def test_short_strike(self):
        now = datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)
        dte = compute_dte_from_symbol("US.SPY260529C742000", now)
        assert dte == 7

    def test_stock_returns_none(self):
        now = datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)
        assert compute_dte_from_symbol("US.SPY", now) is None
        assert compute_dte_from_symbol("AAPL", now) is None

    def test_unparseable_returns_none(self):
        now = datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)
        assert compute_dte_from_symbol("bogus", now) is None
        assert compute_dte_from_symbol("", now) is None

    def test_expired_returns_zero(self):
        # Expiry yesterday → DTE 0 (floors at 0)
        now = datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc)
        dte = compute_dte_from_symbol("US.SPY260529C742000", now)
        assert dte == 0


class TestComputeIntrinsic:
    def test_itm_call(self):
        # SPY 742 call, spot at 745 → intrinsic 3.0
        assert compute_intrinsic("US.SPY260529C00742000", 745.0) == 3.0

    def test_otm_call_zero(self):
        # SPY 742 call, spot at 740 → intrinsic 0
        assert compute_intrinsic("US.SPY260529C00742000", 740.0) == 0.0

    def test_itm_put(self):
        # SPY 742 put, spot at 740 → intrinsic 2.0
        assert compute_intrinsic("US.SPY260529P00742000", 740.0) == 2.0

    def test_otm_put_zero(self):
        assert compute_intrinsic("US.SPY260529P00742000", 745.0) == 0.0

    def test_strike_decoding_short_form(self):
        # 742000 → $742, not $7420
        assert compute_intrinsic("US.SPY260529C742000", 750.0) == 8.0

    def test_no_underlying_returns_none(self):
        assert compute_intrinsic("US.SPY260529C742000", None) is None
        assert compute_intrinsic("US.SPY260529C742000", 0) is None


# ---------------------------------------------------------------------------
# Plan loading + fallbacks
# ---------------------------------------------------------------------------


class TestLoadExitPlan:
    def test_dict_wins(self):
        raw = {
            "hard_stop": 10.0, "hard_target": 20.0,
            "version": 1, "scale_out_ladder": [],
            "trail_stop": None,
            "dte_rules": {"force_exit_at_dte": 2,
                          "force_exit_at_dte_5_if_delta_below": 0.40,
                          "switch_to_intrinsic_floor_at_dte": 5,
                          "intrinsic_floor_buffer": 0.10},
            "regime_rules": {"exit_on_labels": ["CRISIS"],
                             "downsize_50_on_labels": []},
            "time_in_trade_max_days": 30,
        }
        plan = load_exit_plan(raw, fallback_stop=5.0, fallback_target=15.0)
        assert plan is not None
        assert plan.hard_stop == 10.0  # dict wins, not fallback

    def test_legacy_fallback(self):
        plan = load_exit_plan(None, fallback_stop=5.0, fallback_target=15.0)
        assert plan is not None
        assert plan.hard_stop == 5.0
        assert plan.hard_target == 15.0
        # Defaults applied
        assert plan.dte_rules.force_exit_at_dte == 2

    def test_no_data_returns_none(self):
        assert load_exit_plan(None, None, None) is None
        assert load_exit_plan({}, None, None) is None


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


_NOW = datetime(2026, 5, 22, 12, 0, tzinfo=timezone.utc)


def _option_pos(symbol="US.SPY260605C00742000",
                entry=10.0, mark=10.0, qty=1, delta=0.50):
    """Default: SPY 742C, 14 DTE from _NOW, entry=10.0."""
    return {
        "symbol": symbol,
        "asset_type": "OPT",
        "entry_price": entry,
        "mark": mark,
        "qty": qty,
        "delta": delta,
        "opened_at": _NOW,
    }


def _stock_pos(symbol="US.SPY", entry=400.0, mark=400.0, qty=10):
    return {
        "symbol": symbol,
        "asset_type": "STK",
        "entry_price": entry,
        "mark": mark,
        "qty": qty,
        "delta": None,
        "opened_at": _NOW,
    }


def _plan(hard_stop=5.0, hard_target=15.0, scale=None, trail=None, max_days=30):
    return ExitPlan(
        hard_stop=hard_stop,
        hard_target=hard_target,
        scale_out_ladder=scale or [],
        trail_stop=trail,
        time_in_trade_max_days=max_days,
    )


# ---------------------------------------------------------------------------
# P0 — Regime kill switch
# ---------------------------------------------------------------------------


class TestP0Regime:
    def test_crisis_force_exits(self):
        pos = _option_pos(mark=10.0)  # nowhere near stop or target
        d = hard_exit_decision(
            pos=pos, plan=_plan(), regime_label="CRISIS",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is not None
        assert d.action == "EXIT_REGIME"
        assert d.exit_qty_factor == 1.0

    def test_non_crisis_no_exit_from_regime(self):
        pos = _option_pos(mark=10.0)
        d = hard_exit_decision(
            pos=pos, plan=_plan(), regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is None

    def test_custom_exit_labels(self):
        pos = _option_pos(mark=10.0)
        plan = ExitPlan(
            hard_stop=5.0, hard_target=15.0,
            regime_rules=RegimeRulesConfig(exit_on_labels=["CRISIS", "BEAR_TREND"]),
        )
        d = hard_exit_decision(
            pos=pos, plan=plan, regime_label="BEAR_TREND",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is not None and d.action == "EXIT_REGIME"


# ---------------------------------------------------------------------------
# P1 — Hard stop
# ---------------------------------------------------------------------------


class TestP1HardStop:
    def test_at_stop_triggers(self):
        pos = _option_pos(mark=5.0)
        d = hard_exit_decision(
            pos=pos, plan=_plan(hard_stop=5.0), regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is not None and d.action == "EXIT_STOP"

    def test_below_stop_triggers(self):
        pos = _option_pos(mark=4.5)
        d = hard_exit_decision(
            pos=pos, plan=_plan(hard_stop=5.0), regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is not None and d.action == "EXIT_STOP"

    def test_above_stop_holds(self):
        pos = _option_pos(mark=5.01)
        d = hard_exit_decision(
            pos=pos, plan=_plan(hard_stop=5.0), regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is None


# ---------------------------------------------------------------------------
# P2 — DTE rules
# ---------------------------------------------------------------------------


class TestP2DteRules:
    def test_dte_2_force_exit(self):
        # SPY option expiring 5/24, now is 5/22 → DTE=2
        pos = _option_pos(symbol="US.SPY260524C00742000", mark=10.0, delta=0.60)
        d = hard_exit_decision(
            pos=pos, plan=_plan(), regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is not None and d.action == "EXIT_DTE_HARD"

    def test_dte_5_otm_force_exit(self):
        # SPY option expiring 5/27, DTE=5, delta=0.20 (OTM)
        pos = _option_pos(symbol="US.SPY260527C00742000", mark=10.0, delta=0.20)
        d = hard_exit_decision(
            pos=pos, plan=_plan(), regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is not None and d.action == "EXIT_DTE_OTM"
        assert "0.20" in d.reason

    def test_dte_5_itm_no_force_exit(self):
        # DTE=5, delta=0.55 (ITM) → no auto-exit; falls through to intrinsic floor
        pos = _option_pos(symbol="US.SPY260527C00742000", mark=10.0, delta=0.55)
        d = hard_exit_decision(
            pos=pos, plan=_plan(hard_stop=5.0), regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
            underlying_mark=750.0,  # ITM call: 750-742=8 intrinsic
        )
        # mark=10 > floor=8-0.1=7.9 → no exit
        assert d is None

    def test_dte_5_itm_below_intrinsic_floor_exits(self):
        # mark below floor of intrinsic-buffer → EXIT_INTRINSIC_FLOOR
        pos = _option_pos(symbol="US.SPY260527C00742000", mark=7.0, delta=0.55)
        d = hard_exit_decision(
            pos=pos, plan=_plan(hard_stop=5.0), regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
            underlying_mark=750.0,  # intrinsic=8, floor=7.9
        )
        assert d is not None and d.action == "EXIT_INTRINSIC_FLOOR"

    def test_dte_8_otm_holds(self):
        # DTE>5 → no DTE force exit even if OTM
        pos = _option_pos(symbol="US.SPY260530C00742000", mark=10.0, delta=0.20)
        d = hard_exit_decision(
            pos=pos, plan=_plan(), regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is None


# ---------------------------------------------------------------------------
# P3 — Hard target + scale-out ladder
# ---------------------------------------------------------------------------


class TestP3Target:
    def test_at_target_full_exit(self):
        pos = _option_pos(mark=15.0)
        d = hard_exit_decision(
            pos=pos, plan=_plan(hard_target=15.0), regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is not None and d.action == "EXIT_TARGET"
        assert d.exit_qty_factor == 1.0

    def test_scale_out_rung_partial(self):
        plan = _plan(scale=[ScaleOutRung(at_mark=12.0, exit_factor=0.5)])
        pos = _option_pos(mark=12.0)
        d = hard_exit_decision(
            pos=pos, plan=plan, regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is not None and d.action == "EXIT_TARGET"
        assert d.exit_qty_factor == 0.5
        assert "rung#1" in d.reason

    def test_scale_out_rung_skipped_when_taken(self):
        plan = _plan(scale=[
            ScaleOutRung(at_mark=12.0, exit_factor=0.5),
            ScaleOutRung(at_mark=14.0, exit_factor=1.0),
        ])
        pos = _option_pos(mark=12.5)
        d = hard_exit_decision(
            pos=pos, plan=plan, regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
            scale_rungs_taken=1,
        )
        # First rung skipped; mark=12.5 hasn't hit second rung (14.0)
        assert d is None


# ---------------------------------------------------------------------------
# P4 — Trailing stop
# ---------------------------------------------------------------------------


class TestP4Trailing:
    def test_trail_not_engaged_holds(self):
        # entry=10, stop=5 → R_unit=5. mfe=0.4 → high_water=12.0.
        # trail engage_at=15 → not yet engaged.
        plan = _plan(
            hard_stop=5.0, hard_target=20.0,
            trail=TrailStopConfig(engage_at_mark=15.0, distance_pct=0.15,
                                  never_below="break_even"),
        )
        pos = _option_pos(mark=11.0)
        d = hard_exit_decision(
            pos=pos, plan=plan, regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=0.4,
        )
        assert d is None

    def test_trail_engaged_and_breached(self):
        # entry=10, stop=5 → R_unit=5. mfe=2.0 → high_water=20.0.
        # engaged. trail distance=15% → trailed_stop = 20*(1-0.15)=17.0
        # mark=16.0 → exits.
        plan = _plan(
            hard_stop=5.0, hard_target=30.0,
            trail=TrailStopConfig(engage_at_mark=15.0, distance_pct=0.15,
                                  never_below="break_even"),
        )
        pos = _option_pos(mark=16.0)
        d = hard_exit_decision(
            pos=pos, plan=plan, regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=2.0,
        )
        assert d is not None and d.action == "EXIT_TRAIL"

    def test_trail_never_below_break_even(self):
        # high_water=20, trail at 0.15 → would trail to 17.0, never_below
        # break_even=entry=10. Floor wins only if trail < floor; here 17>10.
        # Different scenario: high_water=11, trail to 9.35 — but never_below
        # =10 → trailed_stop=10. mark=9.5 → exits (mark < 10).
        plan = _plan(
            hard_stop=5.0, hard_target=30.0,
            trail=TrailStopConfig(engage_at_mark=11.0, distance_pct=0.15,
                                  never_below="break_even"),
        )
        pos = _option_pos(entry=10.0, mark=9.5)
        d = hard_exit_decision(
            pos=pos, plan=plan, regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=0.2,  # high_water = 10 + 0.2*5 = 11.0
        )
        assert d is not None and d.action == "EXIT_TRAIL"


# ---------------------------------------------------------------------------
# P5 — Max age
# ---------------------------------------------------------------------------


class TestP5Age:
    def test_old_position_exits(self):
        # Opened 31 days ago
        opened = _NOW.replace(month=4, day=21)
        pos = _option_pos(mark=10.0)
        pos["opened_at"] = opened
        d = hard_exit_decision(
            pos=pos, plan=_plan(max_days=30), regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is not None and d.action == "EXIT_AGE"

    def test_young_position_holds(self):
        pos = _option_pos(mark=10.0)
        # opened_at = _NOW (set by fixture)
        d = hard_exit_decision(
            pos=pos, plan=_plan(max_days=30), regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is None


# ---------------------------------------------------------------------------
# Default-plan generator
# ---------------------------------------------------------------------------


class TestDefaultExitPlan:
    def test_option_plan_has_scale_and_trail(self):
        plan = default_exit_plan(entry=10.0, stop=5.0, target=20.0, asset_type="OPT")
        assert plan.hard_stop == 5.0
        assert plan.hard_target == 20.0
        assert len(plan.scale_out_ladder) == 1
        assert plan.scale_out_ladder[0].at_mark == pytest.approx(17.0)  # 10 + 0.7*10
        assert plan.scale_out_ladder[0].exit_factor == 0.5
        assert plan.trail_stop is not None
        assert plan.time_in_trade_max_days == 30

    def test_stock_plan_no_scale_out(self):
        plan = default_exit_plan(entry=100.0, stop=95.0, target=120.0, asset_type="STK")
        assert plan.scale_out_ladder == []
        assert plan.trail_stop is not None
        assert plan.time_in_trade_max_days == 60


# ---------------------------------------------------------------------------
# Priority ordering — regime beats stop beats DTE beats target
# ---------------------------------------------------------------------------


class TestPriorityOrder:
    def test_regime_beats_target(self):
        # mark above target AND regime is CRISIS → regime wins
        pos = _option_pos(mark=20.0)
        plan = _plan(hard_target=15.0)
        d = hard_exit_decision(
            pos=pos, plan=plan, regime_label="CRISIS",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is not None and d.action == "EXIT_REGIME"

    def test_stop_beats_target(self):
        # Both true (pathological): mark=5 satisfies stop=5; also above some target
        pos = _option_pos(mark=5.0)
        plan = _plan(hard_stop=5.0, hard_target=4.0)  # target below stop = pathological
        d = hard_exit_decision(
            pos=pos, plan=plan, regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is not None and d.action == "EXIT_STOP"

    def test_stop_beats_dte(self):
        # mark=5 hits stop AND DTE=2 → stop wins (mark stop is fastest signal)
        pos = _option_pos(symbol="US.SPY260524C00742000", mark=5.0, delta=0.6)
        d = hard_exit_decision(
            pos=pos, plan=_plan(hard_stop=5.0), regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is not None and d.action == "EXIT_STOP"


# ---------------------------------------------------------------------------
# SHORT direction — inverted comparisons (price up = bad, price down = good)
# ---------------------------------------------------------------------------


def _short_plan(hard_stop=6.0, hard_target=1.0, scale=None, max_days=30):
    """Short option: collected $3 premium, stop at $6 (200% loss), target at $1
    (33% premium retained = 67% captured). Geometry: stop > entry > target."""
    return ExitPlan(
        direction="SHORT",
        hard_stop=hard_stop,
        hard_target=hard_target,
        scale_out_ladder=scale or [],
        time_in_trade_max_days=max_days,
    )


def _short_pos(symbol="US.SPY260817P00500000",
               entry=3.0, mark=3.0, qty=1, delta=-0.35):
    """SHORT put: entered at $3.00 premium. ~3 months to expiry."""
    return {
        "symbol": symbol,
        "asset_type": "OPT",
        "entry_price": entry,
        "mark": mark,
        "qty": qty,
        "delta": delta,
        "opened_at": _NOW,
    }


class TestShortDirection:
    def test_short_stop_fires_when_mark_rises_above(self):
        # entry=3, stop=6. mark=6.1 means option price rose against us.
        pos = _short_pos(mark=6.1)
        d = hard_exit_decision(
            pos=pos, plan=_short_plan(), regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is not None and d.action == "EXIT_STOP"
        assert ">=" in d.reason  # short stop uses >=

    def test_short_stop_holds_when_mark_below(self):
        pos = _short_pos(mark=5.99)
        d = hard_exit_decision(
            pos=pos, plan=_short_plan(), regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is None  # not breached yet

    def test_short_target_fires_when_mark_falls_below(self):
        # entry=3, target=1. mark=0.9 = premium has decayed past target.
        pos = _short_pos(mark=0.9)
        d = hard_exit_decision(
            pos=pos, plan=_short_plan(), regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is not None and d.action == "EXIT_TARGET"
        assert "<=" in d.reason

    def test_short_target_holds_when_mark_above(self):
        pos = _short_pos(mark=1.5)
        d = hard_exit_decision(
            pos=pos, plan=_short_plan(), regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is None

    def test_short_scale_out_fires_when_mark_falls_to_rung(self):
        # Rung at $1.50 (premium decayed to 50% of entry $3.00). qty=2 → factor 0.5 ok.
        plan = _short_plan(scale=[ScaleOutRung(at_mark=1.5, exit_factor=0.5)])
        pos = _short_pos(mark=1.5, qty=2)
        d = hard_exit_decision(
            pos=pos, plan=plan, regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is not None and d.action == "EXIT_TARGET"
        assert d.exit_qty_factor == 0.5

    def test_short_no_otm_theta_exit(self):
        """OTM near expiry is FAVORABLE for a short — should NOT force exit."""
        # SHORT put 500 strike, DTE=4, delta=-0.20 (far OTM = good for us)
        pos = _short_pos(symbol="US.SPY260526P00500000", mark=2.0, delta=-0.20)
        d = hard_exit_decision(
            pos=pos, plan=_short_plan(), regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        # DTE=4, not at force_exit_at_dte=2 yet → HOLD
        assert d is None

    def test_short_force_exit_at_dte_2(self):
        # DTE=2 → assignment risk regardless of direction
        pos = _short_pos(symbol="US.SPY260524P00500000", mark=2.0, delta=-0.30)
        d = hard_exit_decision(
            pos=pos, plan=_short_plan(), regime_label="RANGE_LOW_VOL",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is not None and d.action == "EXIT_DTE_HARD"
        assert "assignment risk" in d.reason

    def test_short_regime_crisis_still_exits(self):
        pos = _short_pos(mark=3.0)  # nowhere near stop or target
        d = hard_exit_decision(
            pos=pos, plan=_short_plan(), regime_label="CRISIS",
            now_utc=_NOW, mfe_so_far=None,
        )
        assert d is not None and d.action == "EXIT_REGIME"


class TestDefaultExitPlanShort:
    def test_short_plan_geometry(self):
        # Short: stop > entry > target
        plan = default_exit_plan(
            entry=3.0, stop=6.0, target=1.0,
            asset_type="OPT", direction="SHORT",
        )
        assert plan.direction == "SHORT"
        assert plan.hard_stop == 6.0
        assert plan.hard_target == 1.0
        # Scale rung at entry - 0.5×(entry-target) = 3.0 - 1.0 = 2.0
        assert len(plan.scale_out_ladder) == 1
        assert plan.scale_out_ladder[0].at_mark == pytest.approx(2.0)
        # Default LONG-direction stays unchanged
        long_plan = default_exit_plan(
            entry=10.0, stop=5.0, target=20.0,
            asset_type="OPT", direction="LONG",
        )
        assert long_plan.direction == "LONG"
