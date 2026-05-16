"""Typed LangGraph state for the autonomous orchestrator.

Following TradingAgents' pattern: a single TypedDict per subgraph plus
typed sub-objects for the cross-cutting concerns (regime, proposal, risk
decision). LangGraph's reducer model handles concurrent writes from
parallel nodes by merging via dict update on these top-level keys.

Thread IDs (Postgres checkpointer):
    f"{ticker}:{date}:{trigger}"           per-ticker runs (research, entry)
    f"global:{date}:{trigger}"             portfolio runs (regime, eod)
    f"hourly:{date}:{hour}"                health checks
"""
from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

# -----------------------------------------------------------------------------
# Decision and label primitives — keep these aligned with Postgres schema
# (see migrations/002_phase2.sql).
# -----------------------------------------------------------------------------

Decision = Literal["APPROVE", "DOWNSIZE", "VETO", "DEFER"]

RegimeLabel = Literal[
    "BULL_TREND",
    "RANGE_LOW_VOL",
    "VOLATILE_TRANSITION",
    "BEAR_TREND",
    "CRISIS",
]

Trigger = Literal[
    "premarket_scan",
    "candidate_entry",
    "intraday_monitor",
    "eod_review",
    "weekly_learning",
    "healthcheck",
]


# -----------------------------------------------------------------------------
# Sub-state shapes
# -----------------------------------------------------------------------------


class AgentBudget(TypedDict):
    """Per-tick cost ceiling. DEFER_NO_BUDGET kicks in at 80% of run cap."""

    run_usd_cap: float
    spent_usd: float
    model_calls: int


class RegimeState(TypedDict):
    state_id: int
    label: RegimeLabel
    probabilities: dict[str, float]
    confidence: float
    degradation_level: int  # 0=clean, 1=stale, 2=missing critical → safe-mode
    feature_snapshot_id: int
    model_version_id: int
    crisis_flags: list[str]
    gate: dict[str, Any]
    llm_review_id: NotRequired[int]


class TradeProposal(TypedDict):
    proposal_id: str  # ULID
    ticker: str
    symbol: str  # e.g. US.SPY260530C00720000
    asset_type: Literal["STK", "OPT"]
    direction: Literal["LONG", "LONG_CALL", "LONG_PUT"]
    side: Literal["BUY"]
    qty: float
    entry_price: float
    strategy_label: str
    thesis_id: int
    params_version_id: int
    stop: NotRequired[float]
    target: NotRequired[float]
    expected_return_pct: NotRequired[float]
    max_loss_pct: NotRequired[float]
    option_delta: NotRequired[float]
    option_dte: NotRequired[int]
    option_iv: NotRequired[float]


class RiskDecision(TypedDict):
    risk_decision_id: str  # ULID
    decision: Decision
    approved_qty: float
    max_entry_price: float
    reasons: list[str]
    hard_violations: list[str]
    risk_snapshot_id: int
    expires_at: str  # ISO 8601


class InvestDebateState(TypedDict):
    """Bull / Bear researcher pod state (TradingAgents pattern)."""

    ticker: str
    bull_history: list[str]
    bear_history: list[str]
    rounds_done: int
    consensus: NotRequired[str]


# -----------------------------------------------------------------------------
# Top-level graph state
# -----------------------------------------------------------------------------


class TradingGraphState(TypedDict):
    """The state shape every subgraph reads/writes.

    LangGraph merges concurrent writes per-key via dict update; nodes
    should write only the keys they own (e.g. analyst_pod writes `research`,
    risk_subgraph writes `risk`, etc.).
    """

    # Identity
    run_id: str
    trigger: Trigger
    ts: str  # ISO 8601 UTC

    # Cost governance
    budget: AgentBudget

    # Regime gate (set by premarket_scan, read by every entry / monitor)
    regime: NotRequired[RegimeState]

    # Scout / analyst output
    watchlist: list[str]
    market_data: dict[str, Any]
    candidates: list[dict[str, Any]]
    research: dict[str, Any]
    debate: NotRequired[dict[str, Any]]

    # Account / positions snapshot (Phase 2.5 — load_open_positions)
    account: NotRequired[dict[str, Any]]
    positions: NotRequired[list[dict[str, Any]]]

    # Trader → Risk → Executor pipeline
    proposal: NotRequired[TradeProposal]
    sizing: dict[str, Any]
    risk: NotRequired[RiskDecision]
    order: dict[str, Any]
    fill: dict[str, Any]

    # Shadow proposal log row id (set by build_trade_proposal; downstream
    # nodes UPDATE the same row with their decisions). NotRequired because
    # the DB write is best-effort and may fail without blocking the run.
    shadow_proposal_id: NotRequired[int]

    # Bookkeeping
    journal: dict[str, Any]
    learning: dict[str, Any]
    notifications: list[dict[str, Any]]
    errors: list[dict[str, Any]]


def empty_state(run_id: str, trigger: Trigger, ts: str) -> TradingGraphState:
    """Build a fresh empty state. Used as the initial state for a new run."""
    return {
        "run_id": run_id,
        "trigger": trigger,
        "ts": ts,
        "budget": {"run_usd_cap": 1.0, "spent_usd": 0.0, "model_calls": 0},
        "watchlist": [],
        "market_data": {},
        "candidates": [],
        "research": {},
        "sizing": {},
        "order": {},
        "fill": {},
        "journal": {},
        "learning": {},
        "notifications": [],
        "errors": [],
    }
