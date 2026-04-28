"""Tests for the regime classifier — Layers 0 and 1.

These tests don't need Postgres — they exercise pure Python logic on
synthetic features.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from trading_agent.regime.classifier import (
    ALL_REGIMES,
    classify,
    crisis_overlay,
)
from trading_agent.regime.features import FeatureSnapshot


def _snap(features: dict[str, float], **dq) -> FeatureSnapshot:
    s = FeatureSnapshot(as_of=datetime.now(timezone.utc))
    s.features = features
    s.data_quality = {
        "missing": dq.get("missing", []),
        "stale": dq.get("stale", []),
        "outlier": dq.get("outlier", []),
        "degradation_level": dq.get("degradation_level", 0),
    }
    return s


# ---------------------------------------------------------------------------
# Layer 0: Crisis overlay
# ---------------------------------------------------------------------------


def test_crisis_overlay_two_flags_triggers():
    """VIX 40 + SPY -8% in 5d → crisis (2 flags)."""
    snap = _snap({"vix_level": 40.0, "spy_ret_5": -0.08, "vix_chg_5": 0,
                  "avg_pairwise_corr_20": 0.4, "spy_rv_20": 0.18, "hyg_mom_20": 0.0})
    res = crisis_overlay(snap)
    assert res.is_crisis is True
    assert "vix_abs_high" in res.flags
    assert "spy_5d_drop" in res.flags


def test_crisis_overlay_one_flag_does_not_trigger():
    """High VIX alone is not enough (need 2+ flags)."""
    snap = _snap({"vix_level": 40.0, "spy_ret_5": -0.01, "vix_chg_5": 0,
                  "avg_pairwise_corr_20": 0.4, "spy_rv_20": 0.18, "hyg_mom_20": 0.0})
    res = crisis_overlay(snap)
    assert res.is_crisis is False
    assert res.flags == ["vix_abs_high"]


def test_crisis_overlay_correlation_combo():
    """Avg corr 0.85 + RV 35% trips the corr_rv_combo flag."""
    snap = _snap({"vix_level": 25.0, "spy_ret_5": -0.02, "vix_chg_5": 5,
                  "avg_pairwise_corr_20": 0.85, "spy_rv_20": 0.35, "hyg_mom_20": -0.03})
    res = crisis_overlay(snap)
    assert "corr_rv_combo" in res.flags


def test_crisis_overlay_quiet_market_no_flags():
    """Calm tape — no flags, no crisis."""
    snap = _snap({"vix_level": 14.0, "spy_ret_5": 0.005, "vix_chg_5": -1.0,
                  "avg_pairwise_corr_20": 0.4, "spy_rv_20": 0.10, "hyg_mom_20": 0.01})
    res = crisis_overlay(snap)
    assert res.is_crisis is False
    assert res.flags == []


# ---------------------------------------------------------------------------
# Top-level classify (no HMM yet → rule-based fallback)
# ---------------------------------------------------------------------------


def test_classify_safe_mode_when_critical_data_missing():
    """degradation_level=2 → VOLATILE_TRANSITION + confidence 0."""
    snap = _snap({}, degradation_level=2, missing=["US.SPY", "US.VIX"])
    pred = classify(snap, model=None)
    assert pred.label == "VOLATILE_TRANSITION"
    assert pred.confidence == 0.0
    assert pred.degradation_level == 2


def test_classify_crisis_short_circuits_hmm():
    """When crisis overlay fires, HMM is never consulted (we pass model=None
    and would fall to rule-based, but crisis overlay wins first)."""
    snap = _snap({
        "vix_level": 38.0, "vix_chg_5": 12.0,
        "spy_ret_5": -0.09, "avg_pairwise_corr_20": 0.5,
        "spy_rv_20": 0.20, "hyg_mom_20": 0.0,
    })
    pred = classify(snap, model=None)
    assert pred.label == "CRISIS"
    assert pred.confidence == 1.0
    assert "vix_abs_high" in pred.crisis_flags


def test_classify_rule_fallback_bull_trend():
    """Trend up + low vol → BULL_TREND under fallback."""
    snap = _snap({
        "vix_level": 14.0, "vix_chg_5": 0, "spy_ret_5": 0.02, "spy_ret_20": 0.04,
        "spy_above_50dma": 1.0, "spy_above_200dma": 1.0, "spy_rv_20": 0.10,
        "avg_pairwise_corr_20": 0.4, "hyg_mom_20": 0.005,
    })
    pred = classify(snap, model=None)
    assert pred.label == "BULL_TREND"
    assert pred.needs_llm_review  # rule fallback always invites review


def test_classify_rule_fallback_bear_trend():
    """Trend down + recent loss → BEAR_TREND."""
    snap = _snap({
        "vix_level": 24.0, "vix_chg_5": 4.0, "spy_ret_5": -0.03, "spy_ret_20": -0.05,
        "spy_above_50dma": 0.0, "spy_above_200dma": 0.0, "spy_rv_20": 0.22,
        "avg_pairwise_corr_20": 0.5, "hyg_mom_20": -0.02,
    })
    pred = classify(snap, model=None)
    assert pred.label == "BEAR_TREND"


def test_classify_rule_fallback_range_low_vol():
    """Low vol + flat returns → RANGE_LOW_VOL."""
    snap = _snap({
        "vix_level": 12.0, "vix_chg_5": -0.5, "spy_ret_5": 0.001, "spy_ret_20": 0.005,
        "spy_above_50dma": 1.0, "spy_above_200dma": 1.0, "spy_rv_20": 0.08,
        "avg_pairwise_corr_20": 0.3, "hyg_mom_20": 0.001,
    })
    pred = classify(snap, model=None)
    assert pred.label == "RANGE_LOW_VOL"
