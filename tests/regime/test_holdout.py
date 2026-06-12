"""Tests for the v6+ holdout freeze + the NW-variance evaluation helper.

The holdout guard is the only thing standing between "we have one clean
test window left" and an eighth implicit-selection pass on 2017-2026
(see regime/holdout.py EVALUATIONS_BURNED) — so the guard itself gets
tested on both sides of the boundary, all accepted input types, and the
break-glass escape hatch.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pytest

from trading_agent.regime.evaluation import (
    active_return_stats,
    newey_west_variance,
)
from trading_agent.regime.holdout import (
    EVALUATIONS_BURNED,
    HOLDOUT_START,
    HoldoutViolation,
    require_holdout_intact,
)


# ---------------------------------------------------------------------------
# Holdout constants — the freeze documentation must stay machine-checkable
# ---------------------------------------------------------------------------

def test_holdout_start_is_a_date():
    # date, not datetime/str — comparisons in the guard rely on date semantics
    assert type(HOLDOUT_START) is date
    assert HOLDOUT_START == date(2026, 6, 13)


def test_evaluations_burned_documents_history():
    """Non-empty and one line per burned run — the count is the argument."""
    assert len(EVALUATIONS_BURNED) >= 5  # v1-v5 at minimum
    assert all(isinstance(line, str) and line.strip() for line in EVALUATIONS_BURNED)
    # Every line should cite where the artifacts live, so the claim is auditable
    assert all("reports/" in line for line in EVALUATIONS_BURNED)


# ---------------------------------------------------------------------------
# Guard behavior
# ---------------------------------------------------------------------------

def test_guard_passes_before_holdout():
    # Last burned day (2026-06-12) is fine without any flag
    require_holdout_intact(HOLDOUT_START - timedelta(days=1))
    require_holdout_intact(date(2017, 1, 1))


def test_guard_raises_on_holdout_start_itself():
    """Boundary is inclusive: HOLDOUT_START is holdout data."""
    with pytest.raises(HoldoutViolation):
        require_holdout_intact(HOLDOUT_START)


def test_guard_raises_post_holdout_without_break_glass():
    with pytest.raises(HoldoutViolation) as exc:
        require_holdout_intact(date(2026, 12, 31))
    # Message must tell the operator both the why and the escape hatch
    msg = str(exc.value)
    assert "2026-06-13" in msg
    assert "--break-glass" in msg


def test_guard_accepts_string_and_datetime_inputs():
    """Scripts pass --test-end as ISO strings; computed ends are datetimes."""
    with pytest.raises(HoldoutViolation):
        require_holdout_intact("2026-12-31")
    with pytest.raises(HoldoutViolation):
        require_holdout_intact(datetime(2026, 12, 31, 15, 30))
    require_holdout_intact("2026-06-12")  # pre-holdout string ok


def test_guard_rejects_garbage_input():
    # Fail closed: an unparseable date must error, never silently pass
    with pytest.raises((TypeError, ValueError)):
        require_holdout_intact(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        require_holdout_intact("not-a-date")


def test_break_glass_passes_with_loud_warning(capsys):
    require_holdout_intact(date(2027, 6, 1), break_glass=True)
    captured = capsys.readouterr()
    # Warning goes to stderr so it survives stdout JSON/CSV redirection
    assert "BURNED" in captured.err
    assert "2026-06-13" in captured.err


# ---------------------------------------------------------------------------
# Newey-West variance helper (used by validate_production_hmm's IR SE)
# ---------------------------------------------------------------------------

def test_nw_variance_close_to_naive_on_iid_data():
    """On iid data all autocovariances are ~0, so NW ≈ naive variance.

    The 2x band is generous — with n=4000 and lag=20 the ratio lands
    within a few percent of 1; the band guards against sign/weight bugs
    (e.g. forgetting the Bartlett taper doubles the noise contribution).
    """
    rng = np.random.default_rng(42)
    x = rng.normal(loc=0.0003, scale=0.01, size=4000)  # plausible daily returns
    nw = newey_west_variance(x, lag=20)
    naive = float(np.var(x))
    assert nw > 0
    assert 0.5 < nw / naive < 2.0


def test_nw_variance_inflates_on_autocorrelated_data():
    """AR(1) with phi=0.9 has long-run variance ~19x the marginal variance.

    NW(lag=20) underestimates the full factor but must land well above
    naive — this is the entire reason the helper exists (regime labels
    persist for weeks, so iid SEs overstate the effective sample).
    """
    rng = np.random.default_rng(7)
    n, phi = 6000, 0.9
    eps = rng.normal(size=n)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    nw = newey_west_variance(x, lag=20)
    naive = float(np.var(x))
    assert nw > 2.0 * naive


def test_nw_variance_handles_nans_and_tiny_series():
    x = np.array([0.01, np.nan, -0.02, 0.005, np.nan, 0.0])
    assert newey_west_variance(x, lag=20) == newey_west_variance(x, lag=20)  # not NaN
    assert np.isnan(newey_west_variance(np.array([0.01]), lag=20))


def test_active_return_stats_zero_edge_has_small_t():
    """Identical strategy and benchmark → zero active return, NaN/0 t-stat
    must not blow up; a pure-noise edge must not look significant."""
    rng = np.random.default_rng(123)
    spy = rng.normal(0.0004, 0.012, size=2500)
    noise_overlay = spy + rng.normal(0.0, 0.001, size=2500)  # no real edge
    stats = active_return_stats(noise_overlay, spy, nw_lag=20)
    assert stats["n_days"] == 2500
    assert abs(stats["t_stat_nw"]) < 3.0  # pure noise shouldn't show an edge
    assert stats["nw_se_mean_daily"] > 0
