"""Phase 2.8 — soak phase gate logic."""
from __future__ import annotations

import os

import pytest

from trading_agent.learning.soak import (
    SoakPhase,
    canary_routing_enabled,
    current_phase,
    is_new_entry_allowed,
    soak_audit_dict,
    tiny_paper_qty_cap,
)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("SOAK_PHASE", raising=False)


def test_default_phase_is_autonomous():
    assert current_phase() == SoakPhase.AUTONOMOUS


def test_recognized_phase_set_via_env(monkeypatch):
    monkeypatch.setenv("SOAK_PHASE", "tiny_paper")
    assert current_phase() == SoakPhase.TINY_PAPER


def test_unrecognized_phase_falls_back_to_autonomous(monkeypatch):
    monkeypatch.setenv("SOAK_PHASE", "🤷")
    assert current_phase() == SoakPhase.AUTONOMOUS


def test_read_only_blocks_entries():
    assert not is_new_entry_allowed(SoakPhase.READ_ONLY)


def test_other_phases_allow_entries():
    for p in (SoakPhase.TINY_PAPER, SoakPhase.CANARY_DISABLED,
              SoakPhase.CANARY_10PCT, SoakPhase.AUTONOMOUS):
        assert is_new_entry_allowed(p)


def test_tiny_paper_caps_qty_at_one():
    assert tiny_paper_qty_cap(SoakPhase.TINY_PAPER) == 1


def test_no_cap_outside_tiny_paper():
    for p in (SoakPhase.READ_ONLY, SoakPhase.CANARY_DISABLED,
              SoakPhase.CANARY_10PCT, SoakPhase.AUTONOMOUS):
        assert tiny_paper_qty_cap(p) is None


def test_canary_only_in_canary_10pct_and_autonomous():
    assert not canary_routing_enabled(SoakPhase.READ_ONLY)
    assert not canary_routing_enabled(SoakPhase.TINY_PAPER)
    assert not canary_routing_enabled(SoakPhase.CANARY_DISABLED)
    assert canary_routing_enabled(SoakPhase.CANARY_10PCT)
    assert canary_routing_enabled(SoakPhase.AUTONOMOUS)


def test_audit_dict_shape():
    d = soak_audit_dict(SoakPhase.TINY_PAPER)
    assert d["soak_phase"] == "tiny_paper"
    assert d["new_entries_allowed"] is True
    assert d["tiny_paper_qty_cap"] == 1
    assert d["canary_routing_enabled"] is False
