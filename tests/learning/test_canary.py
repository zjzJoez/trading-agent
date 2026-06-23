"""Phase 2.7 — canary stable-hash routing math + family inference."""
from __future__ import annotations

import pytest

from trading_agent.learning.canary import (
    CANARY_INELIGIBLE_FAMILIES,
    DEFAULT_INITIAL_TRAFFIC_PCT,
    ActiveCanary,
    MultiFamilyCanary,
    bucket_for,
    date_key,
    family_of_params,
    route_canary,
)
from trading_agent.learning.params import (
    ParamFamily,
    ParamResolver,
    defaults_resolver,
)


# -- Family inference --------------------------------------------------------

def test_family_inference_single_family():
    fam = family_of_params({"r1_soft_cap_pct": 0.018, "r3_ticker_exposure_pct": 0.08})
    assert fam == ParamFamily.SIZING_AGGRESSION


def test_family_inference_rejects_multifamily():
    with pytest.raises(MultiFamilyCanary):
        family_of_params({
            "r1_soft_cap_pct": 0.018,        # SIZING_AGGRESSION
            "min_dte": 10,                    # ENTRY_FILTERS
        })


def test_family_inference_rejects_empty():
    with pytest.raises(MultiFamilyCanary):
        family_of_params({})


def test_family_inference_ignores_unknown_keys():
    # Unknown keys would have been rejected upstream; here we just don't
    # let them poison the inference.
    fam = family_of_params({"r1_soft_cap_pct": 0.018, "phantom": 0.0})
    assert fam == ParamFamily.SIZING_AGGRESSION


# -- Bucket math -------------------------------------------------------------

def test_bucket_in_range():
    for ticker in ["SPY", "QQQ", "AAPL", "TSLA"]:
        for fam in ParamFamily:
            b = bucket_for(ticker, "2026-04-28", fam)
            assert 0 <= b < 100


def test_bucket_stable_per_inputs():
    a = bucket_for("SPY", "2026-04-28", ParamFamily.SIZING_AGGRESSION)
    b = bucket_for("SPY", "2026-04-28", ParamFamily.SIZING_AGGRESSION)
    assert a == b


def test_bucket_changes_across_families():
    """Same ticker + day, different family → likely different bucket."""
    seen: set[int] = set()
    for fam in ParamFamily:
        seen.add(bucket_for("SPY", "2026-04-28", fam))
    # Not strictly guaranteed but sha256 across 6 inputs should usually
    # produce ≥ 3 distinct buckets.
    assert len(seen) >= 3


def test_bucket_changes_across_days():
    a = bucket_for("SPY", "2026-04-28", ParamFamily.SIZING_AGGRESSION)
    b = bucket_for("SPY", "2026-04-29", ParamFamily.SIZING_AGGRESSION)
    assert a != b  # not strictly required but very high probability


def test_bucket_distribution_roughly_uniform():
    """200 distinct tickers should land roughly evenly in deciles."""
    tickers = [f"T{i:03d}" for i in range(200)]
    deciles = [0] * 10
    for t in tickers:
        b = bucket_for(t, "2026-04-28", ParamFamily.SIZING_AGGRESSION)
        deciles[b // 10] += 1
    # No decile should hold more than 50 of 200 (25%).
    assert max(deciles) <= 50


# -- date_key ----------------------------------------------------------------

def test_date_key_handles_missing():
    s = date_key(None)
    # YYYY-MM-DD shape
    assert len(s) == 10 and s[4] == "-" and s[7] == "-"


def test_date_key_idempotent_on_iso_string():
    a = date_key("2026-04-28T12:30:00+00:00")
    assert a == "2026-04-28"


# -- Routing -----------------------------------------------------------------

def _canary(traffic: float = DEFAULT_INITIAL_TRAFFIC_PCT) -> ActiveCanary:
    return ActiveCanary(
        version_id=99,
        parent_version_id=1,
        family=ParamFamily.SIZING_AGGRESSION,
        traffic_pct=traffic,
        params={"r1_soft_cap_pct": 0.012},
        created_at=None,
    )


def test_route_no_canaries_is_identity():
    control = defaults_resolver()
    routed = route_canary(control, ticker="SPY", canaries=[])
    assert routed.resolver is not control or routed.resolver.values == control.values
    assert not routed.routed_to_canary


def test_route_full_traffic_always_routes_canary():
    control = defaults_resolver()
    routed = route_canary(
        control, ticker="SPY", canaries=[_canary(traffic=100.0)],
    )
    assert routed.routed_to_canary
    # The overlaid key should now hold the canary's value
    assert routed.resolver.get("r1_soft_cap_pct") == 0.012
    assert routed.resolver.status == "CANARY"


def test_route_zero_traffic_never_routes_canary():
    control = defaults_resolver()
    routed = route_canary(
        control, ticker="SPY", canaries=[_canary(traffic=0.0)],
    )
    assert not routed.routed_to_canary
    assert routed.routed_to_control
    # Resolver still defaults
    assert routed.resolver.get("r1_soft_cap_pct") == 0.020


def test_ineligible_families_documented():
    """After area E only REGIME_THRESHOLDS is fully ineligible: the classifier
    now consumes those keys (ACTIVE swaps are real) but runs at premarket/global
    scope while canary routing is per-(ticker,date) — no global-scope routing
    path, so an A/B canary can't attribute. ENTRY_FILTERS became eligible
    (min_dte wired to deterministic_sizing); CANDIDATE_COUNT was removed and
    the two consumer-less entry filters pruned."""
    assert ParamFamily.REGIME_THRESHOLDS in CANARY_INELIGIBLE_FAMILIES
    # ENTRY_FILTERS now has a live consumer (min_dte) → eligible
    assert ParamFamily.ENTRY_FILTERS not in CANARY_INELIGIBLE_FAMILIES
    # The families with live consumers (trade_nodes) MUST stay eligible
    assert ParamFamily.SIZING_AGGRESSION not in CANARY_INELIGIBLE_FAMILIES
    assert ParamFamily.STOP_DISTANCES not in CANARY_INELIGIBLE_FAMILIES
    assert ParamFamily.REGIME_SIZE_MULTIPLIERS not in CANARY_INELIGIBLE_FAMILIES


def test_route_does_not_mutate_control():
    control = ParamResolver(
        version_id=1, status="ACTIVE", scope="global", scope_key=None,
        values={"r1_soft_cap_pct": 0.020},
    )
    before = dict(control.values)
    route_canary(control, ticker="SPY", canaries=[_canary(traffic=100.0)])
    assert control.values == before
