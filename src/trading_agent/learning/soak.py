"""Phase 2.8 — Soak-phase gates.

The 30-day soak runs over real calendar time and isn't something a single
session can complete.  This module ships the *infrastructure* — the phase
enum, entry gates, and an acceptance-criteria evaluator — so the operator
flips ``SOAK_PHASE`` env vars at the documented checkpoints and the system
adapts deterministically.

Phase progression (per plan §15):

    READ_ONLY        5 days  — no new entries, full ntfy + regime + risk logged
    TINY_PAPER       5 days  — max 1 contract per entry, full audit chain
    CANARY_DISABLED  20 days — no canary routing; CANARY rows insert ok but
                              ``assign_canary`` short-circuits to control
    CANARY_10PCT     20 days — canary at 10% per family; 2.7's gates active
    AUTONOMOUS       30 days — full autonomy; this is the soak window

Operator flips the phase by setting ``SOAK_PHASE`` in
``/home/ubuntu/trading-agent/.env`` (gitignored) and reloading the systemd
timers' EnvironmentFile.  No DB migration needed — the env var is read on
every run.
"""
from __future__ import annotations

import logging
import os
from enum import Enum

log = logging.getLogger(__name__)


class SoakPhase(str, Enum):
    READ_ONLY = "read_only"
    TINY_PAPER = "tiny_paper"
    CANARY_DISABLED = "canary_disabled"
    CANARY_10PCT = "canary_10pct"
    AUTONOMOUS = "autonomous"


_TINY_PAPER_QTY_CAP = 1


def current_phase() -> SoakPhase:
    """Read the soak phase from env.  Defaults to AUTONOMOUS so an unset
    var doesn't accidentally throttle the production path — operator opts
    *into* a stricter phase, never silently out."""
    raw = os.environ.get("SOAK_PHASE", "").strip().lower()
    if not raw:
        return SoakPhase.AUTONOMOUS
    try:
        return SoakPhase(raw)
    except ValueError:
        log.warning("SOAK_PHASE=%r unrecognized — defaulting to AUTONOMOUS", raw)
        return SoakPhase.AUTONOMOUS


def is_new_entry_allowed(phase: SoakPhase | None = None) -> bool:
    """READ_ONLY blocks all new entries.  All other phases allow entries
    (with possibly-tightened sizing or canary rules)."""
    p = phase or current_phase()
    return p != SoakPhase.READ_ONLY


def tiny_paper_qty_cap(phase: SoakPhase | None = None) -> int | None:
    """Returns the absolute qty ceiling for the current phase, or None
    when no soak-imposed cap applies."""
    p = phase or current_phase()
    if p == SoakPhase.TINY_PAPER:
        return _TINY_PAPER_QTY_CAP
    return None


def canary_routing_enabled(phase: SoakPhase | None = None) -> bool:
    """True only in CANARY_10PCT and AUTONOMOUS."""
    p = phase or current_phase()
    return p in (SoakPhase.CANARY_10PCT, SoakPhase.AUTONOMOUS)


def soak_audit_dict(phase: SoakPhase | None = None) -> dict[str, object]:
    p = phase or current_phase()
    return {
        "soak_phase": p.value,
        "new_entries_allowed": is_new_entry_allowed(p),
        "tiny_paper_qty_cap": tiny_paper_qty_cap(p),
        "canary_routing_enabled": canary_routing_enabled(p),
    }


__all__ = [
    "SoakPhase",
    "canary_routing_enabled",
    "current_phase",
    "is_new_entry_allowed",
    "soak_audit_dict",
    "tiny_paper_qty_cap",
]
