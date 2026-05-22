"""Deterministic exit rules — replaces the LLM exit_monitor on the hot path.

The hard_executor consumes the ExitPlan written at entry time and produces
an EXIT_* decision every intraday tick. See docs/position_management.md for
the full rule ladder.
"""
from trading_agent.exits.hard_executor import (
    ExitDecision,
    compute_dte_from_symbol,
    compute_intrinsic,
    hard_exit_decision,
)

__all__ = [
    "ExitDecision",
    "compute_dte_from_symbol",
    "compute_intrinsic",
    "hard_exit_decision",
]
