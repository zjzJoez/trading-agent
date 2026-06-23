"""Area E — regime_thresholds are LIVE-consumed by the classifier.

Two layers:
  1. classify() honours crisis_params / confidence kwargs (the override surface).
  2. classify_regime (the premarket node) threads the ACTIVE resolver values
     into classify() with the correct key mapping (incl. the percent→fraction
     conversion for the SPY 5d drop), so an ACTIVE param swap actually moves
     the live classifier.

These keys stay canary_eligible=False (global/premarket scope), but the
consumer is real — this pins it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from trading_agent.regime.classifier import RegimePrediction, classify
from trading_agent.regime.features import FeatureSnapshot


def _snap(features: dict) -> FeatureSnapshot:
    s = FeatureSnapshot(as_of=datetime.now(timezone.utc))
    s.features = features
    s.data_quality = {"missing": [], "stale": [], "outlier": [], "degradation_level": 0}
    return s


# -- 1. classify() honours the crisis_params override ------------------------


def test_classify_crisis_params_override_changes_outcome():
    """VIX 32 + SPY -8%: one flag (spy) under defaults → NOT crisis. Tighten
    vix_abs_high to 30 and the VIX flag also trips → 2 flags → CRISIS."""
    snap = _snap({
        "vix_level": 32.0, "vix_chg_5": 2.0, "spy_ret_5": -0.08,
        "avg_pairwise_corr_20": 0.4, "spy_rv_20": 0.18, "hyg_mom_20": 0.0,
    })
    # Default thresholds: vix 32 < 35 → only the spy flag → not crisis.
    assert classify(snap, model=None).label != "CRISIS"
    # Tightened crisis_vix_level (35 → 30): vix 32 ≥ 30 now flags too → crisis.
    pred = classify(snap, model=None, crisis_params={"vix_abs_high": 30.0})
    assert pred.label == "CRISIS"


# -- 2. classify_regime node threads resolver values into classify() ---------


def test_classify_regime_threads_resolver_into_classify():
    import trading_agent.graph.nodes.regime_nodes as rn

    captured: dict = {}

    def _fake_classify(snap, model=None, **kwargs):
        captured.update(kwargs)
        return RegimePrediction(label="BULL_TREND", confidence=0.9,
                                probabilities={"BULL_TREND": 0.9})

    state = {
        "run_id": "t", "trigger": "premarket_scan",
        "regime": {"_features_snapshot": {
            "as_of": datetime(2026, 6, 12, tzinfo=timezone.utc).isoformat(),
            "_obj_features": {"vix_level": 18.0},
            "_obj_data_quality": {"degradation_level": 0},
        }},
        "learning": {"params": {"values": {
            "crisis_vix_level": 40.0,
            "crisis_vix_delta_5d": 12.0,
            "crisis_spy_5d_drop_pct": -8.0,   # percent → -0.08 fraction
            "regime_confidence_transition": 0.50,
            "regime_confidence_llm_review": 0.60,
        }}},
    }

    with patch.object(rn, "classify", _fake_classify), \
         patch.object(rn, "get_active_model", return_value=None), \
         patch.object(rn, "get_latest_regime_state", return_value=None), \
         patch.object(rn, "emit", return_value=1):
        rn.classify_regime(state)

    assert captured["crisis_params"] == {
        "vix_abs_high": 40.0,
        "vix_5d_change_high": 12.0,
        "spy_5d_drop": -0.08,   # converted from -8.0 percent
    }
    assert captured["confidence_transition"] == 0.50
    assert captured["confidence_llm_review"] == 0.60


def test_classify_regime_no_overrides_passes_defaults():
    """Empty resolver → crisis_params=None and no confidence kwargs, so the
    classifier behaves byte-identically to the pre-area-E path."""
    import trading_agent.graph.nodes.regime_nodes as rn

    captured: dict = {}

    def _fake_classify(snap, model=None, **kwargs):
        captured.update(kwargs)
        return RegimePrediction(label="BULL_TREND", confidence=0.9,
                                probabilities={"BULL_TREND": 0.9})

    state = {
        "run_id": "t", "trigger": "premarket_scan",
        "regime": {"_features_snapshot": {
            "as_of": datetime(2026, 6, 12, tzinfo=timezone.utc).isoformat(),
            "_obj_features": {"vix_level": 18.0},
            "_obj_data_quality": {"degradation_level": 0},
        }},
        "learning": {},
    }

    with patch.object(rn, "classify", _fake_classify), \
         patch.object(rn, "get_active_model", return_value=None), \
         patch.object(rn, "get_latest_regime_state", return_value=None), \
         patch.object(rn, "emit", return_value=1):
        rn.classify_regime(state)

    assert captured["crisis_params"] is None
    assert "confidence_transition" not in captured
    assert "confidence_llm_review" not in captured
