"""ITEM 11 — CI guard pinning the A/A-test bug class permanently.

Background: every Phase-2.7 canary before ITEM 11 was an A/A test — all 13
mutable PARAM_BOUNDS keys were read only inside learning/ (shadow/replay),
never by live decision code, so canary-vs-control compared two identical
treatments.  Two invariants are pinned here:

  1. Every key flagged ``canary_eligible=True`` MUST appear (as a string)
     somewhere in the src tree OUTSIDE learning/ — i.e. it has a live
     consumer.  Flipping a key eligible without wiring it goes red.
  2. ``route_canary`` / ``insert_canary`` must never spend traffic on a
     canary touching an ineligible key.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from trading_agent.learning.canary import (
    CANARY_INELIGIBLE_FAMILIES,
    ActiveCanary,
    CanaryFamilyIneligible,
    ineligible_keys,
    insert_canary,
    route_canary,
)
from trading_agent.learning.params import (
    PARAM_BOUNDS,
    ParamFamily,
    defaults_resolver,
)

# tests/learning/test_param_consumers.py → repo root is parents[2]
_SRC_PKG = Path(__file__).resolve().parents[2] / "src" / "trading_agent"

_ELIGIBLE_KEYS = sorted(
    k for k, b in PARAM_BOUNDS.items() if b.canary_eligible
)
_INELIGIBLE_KEYS = sorted(
    k for k, b in PARAM_BOUNDS.items() if not b.canary_eligible
)


def _non_learning_sources() -> list[Path]:
    """All package .py files OUTSIDE the learning/ subpackage."""
    assert _SRC_PKG.is_dir(), f"src tree not found at {_SRC_PKG}"
    return [
        p for p in _SRC_PKG.rglob("*.py")
        if "learning" not in p.relative_to(_SRC_PKG).parts
    ]


# -- 1. every eligible key has a consumer outside learning/ -------------------

@pytest.mark.parametrize("key", _ELIGIBLE_KEYS)
def test_canary_eligible_key_has_live_consumer(key: str):
    """grep-level pin: the key string must occur outside learning/.

    If this fails for a key you just flipped eligible, you forgot to wire
    the consumer (or you wired it via an indirection that hides the key
    literal — keep the literal at the consumer, that's the contract)."""
    hits = [
        str(p.relative_to(_SRC_PKG))
        for p in _non_learning_sources()
        if key in p.read_text()
    ]
    assert hits, (
        f"PARAM_BOUNDS[{key!r}] is canary_eligible=True but no file outside "
        f"learning/ mentions it — that canary would be an A/A test. Wire a "
        f"consumer or flag the key canary_eligible=False."
    )


def test_expected_wiring_snapshot():
    """The keys ITEM 11 wired must be eligible; the rest must not be.

    This is a deliberate snapshot — when someone wires a new consumer they
    flip the flag in params.py AND update this list, keeping the diff
    self-documenting."""
    assert _ELIGIBLE_KEYS == [
        "implicit_stop_frac",
        "option_premium_stop_pct",
        "r1_soft_cap_pct",
        "r3_ticker_exposure_pct",
        "r5_soft_notional_pct",
        "size_mult_bear_trend",
        "size_mult_bull_trend",
        "size_mult_range_low_vol",
        "size_mult_volatile_transition",
    ]


def test_fully_ineligible_families_derived_from_key_flags():
    """CANARY_INELIGIBLE_FAMILIES is derived, not hand-listed — one source
    of truth (the per-key flags in params.py)."""
    assert CANARY_INELIGIBLE_FAMILIES == frozenset({
        ParamFamily.ENTRY_FILTERS,
        ParamFamily.CANDIDATE_COUNT,
        ParamFamily.REGIME_THRESHOLDS,
    })


def test_ineligible_keys_helper():
    assert ineligible_keys({"r1_soft_cap_pct": 0.01}) == []
    assert ineligible_keys(
        {"implicit_stop_frac": 0.04, "default_stop_atr_mult": 2.0}
    ) == ["default_stop_atr_mult"]
    # Unknown keys are ignored (rejected upstream at insert time)
    assert ineligible_keys({"phantom": 1}) == []


# -- 2. routing / insertion refuse ineligible canaries ------------------------

def _thresholds_canary(traffic: float = 100.0) -> ActiveCanary:
    return ActiveCanary(
        version_id=77,
        parent_version_id=1,
        family=ParamFamily.REGIME_THRESHOLDS,
        traffic_pct=traffic,
        params={"crisis_vix_level": 40.0},
        created_at=None,
    )


def test_route_canary_skips_ineligible_even_at_full_traffic():
    """100% traffic on a label-only canary must still route ZERO traffic."""
    control = defaults_resolver()
    routed = route_canary(
        control, ticker="SPY", canaries=[_thresholds_canary(traffic=100.0)],
    )
    assert routed.routed_to_canary == []
    assert routed.routed_to_control == []  # no assignment row either
    assert routed.resolver.status == control.status
    assert routed.resolver.get("crisis_vix_level") == 35.0  # default intact
    assert routed.skipped_ineligible == [{
        "canary_version_id": 77,
        "family": "regime_thresholds",
        "ineligible_keys": ["crisis_vix_level"],
    }]
    # ...and the skip is auditable in the emitted payload
    assert routed.to_audit_dict()["skipped_ineligible"]


def test_route_canary_still_routes_eligible_families():
    """The skip must not over-reach: a sizing canary keeps routing."""
    control = defaults_resolver()
    c = ActiveCanary(
        version_id=88, parent_version_id=1,
        family=ParamFamily.SIZING_AGGRESSION, traffic_pct=100.0,
        params={"r1_soft_cap_pct": 0.012}, created_at=None,
    )
    routed = route_canary(control, ticker="SPY", canaries=[c])
    assert routed.routed_to_canary
    assert routed.resolver.get("r1_soft_cap_pct") == 0.012
    assert routed.skipped_ineligible == []


def test_insert_canary_rejects_ineligible_family():
    with pytest.raises(CanaryFamilyIneligible):
        insert_canary(
            parent_version_id=1,
            params={"crisis_vix_level": 40.0},
            rationale="x", created_by="test",
        )


def test_insert_canary_rejects_ineligible_key_in_mixed_family():
    """stop_distances is family-eligible, but default_stop_atr_mult has no
    ATR consumer — the per-key check must refuse it at creation time."""
    with pytest.raises(CanaryFamilyIneligible):
        insert_canary(
            parent_version_id=1,
            params={"default_stop_atr_mult": 2.0},
            rationale="x", created_by="test",
        )
