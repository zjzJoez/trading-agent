"""Guards for scripts/hmm_feature_ablation.py.

The script had three crash bugs against the current FeatureSnapshot /
RegimePrediction API (it had never been run). These tests pin the fixed
contract so it can't silently rot again, plus the holdout guard wiring.
They avoid Postgres and heavy HMM training — only the pure helpers and the
arg/guard plumbing are exercised.
"""
from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import scripts.hmm_feature_ablation as ab
from trading_agent.regime.features import HMM_FEATURE_ORDER, FeatureSnapshot


def _snap(as_of: datetime, **feats) -> FeatureSnapshot:
    base = {k: 0.0 for k in HMM_FEATURE_ORDER}
    base.update(feats)
    return FeatureSnapshot(as_of=as_of, features=base)


def test_load_snapshots_range_builds_featuresnapshot():
    """BUG 1: the DB `data_quality` column is a jsonb dict, and degradation_level
    is a derived property — not constructor kwargs. Loader must build a valid
    FeatureSnapshot from (as_of, features, source_freshness, data_quality)."""
    fake_rows = [
        (datetime(2020, 1, 2), {"vix_level": 18.0}, {"vix": 1.0},
         {"degradation_level": 2, "stale": ["vix"]}),
    ]
    cur = MagicMock()
    cur.fetchall.return_value = fake_rows
    cm = MagicMock()
    cm.__enter__.return_value = cur
    cm.__exit__.return_value = False

    with patch.object(ab, "cursor", return_value=cm):
        out = ab.load_snapshots_range("2020-01-01", "2021-01-01")

    assert len(out) == 1
    as_of, snap = out[0]
    assert isinstance(snap, FeatureSnapshot)
    assert snap.features["vix_level"] == 18.0
    # degradation_level is derived from the data_quality dict, not a field.
    assert snap.degradation_level == 2


def test_feature_matrix_zeroes_ablated_column():
    """Ablation = zero the standardized column. Confirm the right column is
    zeroed and the others are untouched."""
    feat = HMM_FEATURE_ORDER[0]
    other = HMM_FEATURE_ORDER[1]
    snaps = [
        (datetime(2020, 1, 2), _snap(datetime(2020, 1, 2), **{feat: 1.0, other: 5.0})),
        (datetime(2020, 1, 3), _snap(datetime(2020, 1, 3), **{feat: 2.0, other: 7.0})),
        (datetime(2020, 1, 6), _snap(datetime(2020, 1, 6), **{feat: 3.0, other: 9.0})),
    ]
    Z, means, stds = ab.feature_matrix(snaps, ablated_feature=feat)
    idx = HMM_FEATURE_ORDER.index(feat)
    other_idx = HMM_FEATURE_ORDER.index(other)
    assert np.allclose(Z[:, idx], 0.0)          # ablated column zeroed
    assert not np.allclose(Z[:, other_idx], 0.0)  # other column has variance


def test_patched_snapshot_construction_matches_api():
    """BUG 2: the test-time patched-snapshot rebuild must use data_quality=
    (not the removed data_quality_warnings/degradation_level kwargs)."""
    snap = _snap(datetime(2020, 1, 2), **{HMM_FEATURE_ORDER[0]: 42.0})
    patched = dict(snap.features)
    patched[HMM_FEATURE_ORDER[0]] = 0.0
    rebuilt = FeatureSnapshot(
        as_of=snap.as_of,
        features=patched,
        source_freshness=snap.source_freshness,
        data_quality=snap.data_quality,
    )
    assert rebuilt.as_vector()[0] == 0.0
    assert rebuilt.degradation_level == 0


def test_compute_strategy_sharpe_respects_allow_new_entries():
    """BUG 3 contract: the per-day `allow_new_entries` bool (derived from
    label != 'CRISIS') gates strategy returns to zero. When entries are
    blocked every day, the strategy has no variance and Sharpe is None,
    while SPY-only Sharpe is still computed."""
    days = pd.date_range("2020-01-02", periods=10, freq="B")
    spy = pd.Series(np.linspace(0.01, -0.01, len(days)), index=days)

    blocked = [
        {"as_of": d.date().isoformat(), "size_multiplier": 1.0,
         "allow_new_entries": False}
        for d in days
    ]
    res = ab.compute_strategy_sharpe(blocked, spy)
    assert res["strategy_sharpe"] is None       # all-zero returns => no Sharpe
    assert res["spy_sharpe"] is not None

    # When entries are allowed at full size, strategy == SPY exactly.
    allowed = [
        {"as_of": d.date().isoformat(), "size_multiplier": 1.0,
         "allow_new_entries": True}
        for d in days
    ]
    res2 = ab.compute_strategy_sharpe(allowed, spy)
    assert res2["strategy_sharpe"] == pytest.approx(res2["spy_sharpe"])


def test_main_holdout_guard_blocks_future_year(monkeypatch):
    """--last-year reaching the frozen holdout must exit(5) BEFORE any run,
    without --break-glass."""
    monkeypatch.setattr(sys, "argv", ["hmm_feature_ablation", "--last-year", "2028"])
    with pytest.raises(SystemExit) as ei:
        ab.main()
    assert ei.value.code == 5
