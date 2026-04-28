"""Phase 2.6 — Online learning: shadow → canary → promote.

This package owns the *mutable* parameter surface of the trading agent.
Hard risk caps (R1, R5, MAX_CONCURRENT_OPENS, CRISIS size_multiplier=0,
TrdEnv toggle) are NEVER reachable from here.

Modules:
    params   — parameter family allow-list, bounds, and ACTIVE-version resolver
    shadow   — at-decision-time counterfactual recorder (§13 / plan §8)
    outcome  — close-time enrichment writer for trade_outcome_features
    replay   — historical replay of decisions under candidate parameters

Phase 2.6 is shadow-only: parameters are recorded and replayed, never used to
drive a real order.  Canary routing arrives in Phase 2.7.
"""
from __future__ import annotations
