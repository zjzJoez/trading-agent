"""Phase 2.6 — shadow tracker counterfactual math."""
from __future__ import annotations

from datetime import datetime, timezone

from trading_agent.learning.params import ParamResolver, defaults_resolver
from trading_agent.learning.shadow import (
    replay_crisis_overlay,
    replay_sizing_caps,
)
from trading_agent.regime.features import FeatureSnapshot


def _make_resolver(**overrides) -> ParamResolver:
    return ParamResolver(
        version_id=1, status="SHADOW", scope="global", scope_key=None,
        values=overrides,
    )


def test_replay_sizing_identity_when_resolvers_match():
    """Same control + shadow on qty that already fits hard caps → unchanged."""
    control = defaults_resolver()
    shadow = defaults_resolver()
    # entry=4 → notional per contract = 400; qty=2 → 800 notional, well
    # below R5 cap (1% × 100k = 1000) and R1 cap (2% × 100k = 2000).
    cf = replay_sizing_caps(
        equity=100_000.0,
        entry_price=4.0,
        contract_multiplier=100,
        qty_actual=2,
        max_loss_per_contract=200.0,
        regime_label="BULL_TREND",
        control=control,
        shadow=shadow,
    )
    assert cf.qty_shadow == cf.qty_actual == 2
    assert cf.multiplier_actual == cf.multiplier_shadow == 1.0


def test_replay_sizing_shadow_tighter_r1_blocks_qty():
    """Lower soft R1 → smaller qty_shadow."""
    control = defaults_resolver()
    shadow = _make_resolver(r1_soft_cap_pct=0.005)  # half of default
    cf = replay_sizing_caps(
        equity=100_000.0,
        entry_price=20.0,
        contract_multiplier=100,
        qty_actual=10,
        max_loss_per_contract=400.0,
        regime_label="BULL_TREND",
        control=control,
        shadow=shadow,
    )
    assert cf.qty_shadow < cf.qty_actual
    assert cf.blocking_rule in {"R1", "R5"}


def test_replay_sizing_smaller_regime_multiplier_shrinks_qty():
    control = defaults_resolver()
    shadow = _make_resolver(size_mult_bull_trend=0.50)
    cf = replay_sizing_caps(
        equity=100_000.0,
        entry_price=10.0,
        contract_multiplier=100,
        qty_actual=4,
        max_loss_per_contract=100.0,
        regime_label="BULL_TREND",
        control=control,
        shadow=shadow,
    )
    assert cf.qty_shadow <= cf.qty_actual


def test_replay_crisis_overlay_default_features_no_crisis():
    snap = FeatureSnapshot(
        as_of=datetime.now(timezone.utc),
        features={
            "vix_level": 15.0,
            "vix_chg_5": 1.0,
            "spy_ret_5": 0.01,
            "avg_pairwise_corr_20": 0.3,
            "spy_rv_20": 0.10,
            "hyg_mom_20": 0.0,
        },
    )
    control = defaults_resolver()
    shadow = defaults_resolver()
    cf = replay_crisis_overlay(snap, control, shadow)
    assert cf.is_crisis_actual is False
    assert cf.is_crisis_shadow is False
    assert cf.flips is False


def test_replay_crisis_overlay_flips_when_threshold_lowered():
    """VIX=32 below default 35 → no crisis on control; lower threshold → crisis on shadow."""
    snap = FeatureSnapshot(
        as_of=datetime.now(timezone.utc),
        features={
            "vix_level": 32.0,
            "vix_chg_5": 12.0,  # already trips vix_5d_change_high
            "spy_ret_5": 0.0,
            "avg_pairwise_corr_20": 0.3,
            "spy_rv_20": 0.10,
            "hyg_mom_20": 0.0,
        },
    )
    control = defaults_resolver()
    # Drop vix_level threshold to 30 → second flag now trips on the SAME
    # snapshot, pushing it past min_flags_to_crisis=2.
    shadow = _make_resolver(crisis_vix_level=30.0)
    cf = replay_crisis_overlay(snap, control, shadow)
    assert cf.is_crisis_actual is False
    assert cf.is_crisis_shadow is True
    assert cf.flips is True
