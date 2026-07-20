"""Phase 2.6 — params allow-list, bounds, and resolver."""
from __future__ import annotations

import pytest

from trading_agent.learning.params import (
    FROZEN_HARD_CAPS,
    PARAM_BOUNDS,
    FrozenParamMutationAttempt,
    ParamFamily,
    ParamResolver,
    UnknownParamKey,
    _validate_payload,
    defaults_resolver,
)


def test_every_bound_assigns_to_known_family():
    families = set(ParamFamily)
    for k, b in PARAM_BOUNDS.items():
        assert b.family in families, f"{k} → {b.family} not in ParamFamily"


def test_defaults_within_bounds():
    for k, b in PARAM_BOUNDS.items():
        assert b.min <= b.default <= b.max, f"{k} default outside bounds"


def test_clamp_respects_integer_flag():
    b = PARAM_BOUNDS["min_dte"]
    assert b.integer is True
    assert b.clamp(7.4) == 7
    assert b.clamp(-1) == b.min
    assert b.clamp(99) == b.max


def test_clamp_returns_float_for_continuous():
    b = PARAM_BOUNDS["r1_soft_cap_pct"]
    assert isinstance(b.clamp(0.015), float)
    assert b.clamp(0.5) == b.max
    assert b.clamp(0.0001) == b.min


def test_validate_payload_rejects_unknown_key():
    with pytest.raises(UnknownParamKey):
        _validate_payload({"this_key_does_not_exist": 1.0})


def test_validate_payload_rejects_frozen_hard_cap():
    for frozen in list(FROZEN_HARD_CAPS)[:3]:
        with pytest.raises(FrozenParamMutationAttempt):
            _validate_payload({frozen: 0.05})


def test_validate_payload_clamps_out_of_range():
    cleaned = _validate_payload({"r1_soft_cap_pct": 0.99})
    assert cleaned["r1_soft_cap_pct"] == PARAM_BOUNDS["r1_soft_cap_pct"].max


def test_resolver_falls_back_to_defaults():
    r = defaults_resolver()
    assert r.version_id is None
    assert r.status == "DEFAULTS"
    assert r.get("r1_soft_cap_pct") == PARAM_BOUNDS["r1_soft_cap_pct"].default
    assert r.get("min_dte") == PARAM_BOUNDS["min_dte"].default


def test_resolver_uses_overrides_when_present():
    r = ParamResolver(
        version_id=42, status="ACTIVE", scope="global", scope_key=None,
        values={"r1_soft_cap_pct": 0.012},
    )
    assert r.get("r1_soft_cap_pct") == 0.012
    assert r.get("min_dte") == PARAM_BOUNDS["min_dte"].default


def test_resolver_unknown_key_raises():
    r = defaults_resolver()
    with pytest.raises(UnknownParamKey):
        r.get("not_a_real_param")


def test_resolver_family_view():
    r = defaults_resolver()
    fam = r.family(ParamFamily.REGIME_SIZE_MULTIPLIERS)
    assert set(fam) == {
        "size_mult_bull_trend",
        "size_mult_range_low_vol",
        "size_mult_volatile_transition",
        "size_mult_bear_trend",
    }
    assert "size_mult_crisis" not in fam  # CRISIS is hard zero, never mutable


def test_crisis_size_multiplier_is_frozen():
    assert "size_mult_crisis" in FROZEN_HARD_CAPS


def test_real_money_toggle_is_frozen():
    assert "enable_real" in FROZEN_HARD_CAPS
    assert "trd_env" in FROZEN_HARD_CAPS


def test_audit_dict_round_trip():
    r = ParamResolver(
        version_id=7, status="CANARY", scope="global", scope_key=None,
        values={"r1_soft_cap_pct": 0.018},
    )
    d = r.to_audit_dict()
    assert d["version_id"] == 7
    assert d["status"] == "CANARY"
    assert d["values"]["r1_soft_cap_pct"] == 0.018
