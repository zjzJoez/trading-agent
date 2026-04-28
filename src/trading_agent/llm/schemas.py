"""Pydantic output schemas for the 14 LLM agent roles.

These are the contracts the OAuth router validates against. Each role's
agent markdown definition declares the same shape; if a model returns
anything not matching, the router retries once then raises
LLMSchemaViolation, and the caller (typically the risk pipeline) defaults
to DEFER on schema failure.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    """Reject extra fields; coerce nothing."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# -----------------------------------------------------------------------------
# Regime
# -----------------------------------------------------------------------------


class RegimeReviewerOutput(_Strict):
    review_label: Literal[
        "CONFIRM",
        "DOWNGRADE_TO_TRANSITION",
        "DOWNGRADE_TO_CRISIS",
        "DATA_QUALITY_DEFER",
    ]
    confidence_adjustment: float = Field(ge=-0.4, le=0.0)
    risk_notes: list[str]
    must_defer_new_entries: bool


# -----------------------------------------------------------------------------
# Scout
# -----------------------------------------------------------------------------


class ScoutCandidate(_Strict):
    ticker: str
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=160)


class ScoutSkipped(_Strict):
    ticker: str
    reason: str = Field(max_length=160)


class ScoutOutput(_Strict):
    candidates: list[ScoutCandidate]
    skipped: list[ScoutSkipped] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Analysts (technical / news / sentiment / fundamental)
# -----------------------------------------------------------------------------


class TechnicalAnalystOutput(_Strict):
    ticker: str
    trend_1d: Literal["UP", "DOWN", "RANGE"]
    trend_1h: Literal["UP", "DOWN", "RANGE"]
    support_levels: list[float]
    resistance_levels: list[float]
    iv_skew_summary: str
    iv_atm_30d: float
    hv_30d: float
    setup_quality: float = Field(ge=0.0, le=1.0)
    primary_signal: str
    directional_bias: Literal["LONG", "SHORT", "NEUTRAL"]
    chase_risk: bool
    tech_report_md: str


class NewsAnalystOutput(_Strict):
    ticker: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    headline_count: int = Field(ge=0)
    filings_count: int = Field(ge=0)
    primary_catalysts: list[str]
    earnings_within_5d: bool
    summary_md: str


class SentimentAnalystOutput(_Strict):
    ticker: str
    insider_score: float = Field(ge=-1.0, le=1.0)
    insider_summary: str
    rag_lessons: list[str]
    rag_warnings: list[str]
    summary_md: str


class FundamentalAnalystOutput(_Strict):
    ticker: str
    fundamentals_score: float = Field(ge=-1.0, le=1.0)
    revenue_yoy_pct: float | None = None
    margin_trend: Literal["EXPANDING", "FLAT", "CONTRACTING", "UNKNOWN"]
    primary_risks: list[str]
    cash_health_summary: str
    fund_report_md: str


# -----------------------------------------------------------------------------
# Researchers (bull / bear)
# -----------------------------------------------------------------------------


class BullResearcherOutput(_Strict):
    round: int = Field(ge=1, le=2)
    thesis: str
    key_drivers: list[str]
    addressed_bear_points: list[str] = Field(default_factory=list)
    remaining_uncertainty: list[str]
    bull_case_md: str


class BearResearcherOutput(_Strict):
    round: int = Field(ge=1, le=2)
    thesis: str
    key_drivers: list[str]
    addressed_bull_points: list[str] = Field(default_factory=list)
    remaining_uncertainty: list[str]
    bear_case_md: str


# -----------------------------------------------------------------------------
# Trader / synthesizer
# -----------------------------------------------------------------------------


class TraderProposal(_Strict):
    ticker: str
    symbol: str
    asset_type: Literal["STK", "OPT"]
    direction: Literal["LONG", "LONG_CALL", "LONG_PUT"]
    strategy_label: str
    entry_price: float
    stop: float
    target: float
    expected_return_pct: float
    max_loss_pct: float
    option_delta: float | None = None
    option_dte: int | None = None
    option_iv: float | None = None
    qty_request: int = Field(ge=0)


class TraderSynthesizerOutput(_Strict):
    decline_to_trade: bool
    decline_reason: str = ""
    proposal: TraderProposal | None = None
    reflection_notes: list[str]
    proposal_notes: str


# -----------------------------------------------------------------------------
# Risk council
# -----------------------------------------------------------------------------


class RiskConservativeOutput(_Strict):
    decision: Literal["APPROVE", "DOWNSIZE", "VETO", "DEFER"]
    approved_qty_factor: float = Field(ge=0.0, le=1.0)
    primary_risks: list[str]
    concerns_md: str


class RiskOpportunityOutput(_Strict):
    decision: Literal["APPROVE", "DOWNSIZE", "VETO", "DEFER"]
    approved_qty_factor: float = Field(ge=0.0, le=1.0)
    primary_opportunities: list[str]
    concerns_about_conservative_review: list[str]
    rationale_md: str


class RiskArbiterOutput(_Strict):
    decision: Literal["APPROVE", "DOWNSIZE", "VETO", "DEFER"]
    approved_qty: int = Field(ge=0)
    max_entry_price: float = Field(ge=0.0)
    primary_risks: list[str]
    reason: str
    agreed_with: Literal["conservative", "opportunity", "neither", "both"]
    required_conditions: list[str]


# -----------------------------------------------------------------------------
# Exit monitor / journal / digest
# -----------------------------------------------------------------------------


class ExitMonitorOutput(_Strict):
    action: Literal[
        "HOLD",
        "EXIT_STOP",
        "EXIT_TARGET",
        "EXIT_TIME_DECAY",
        "EXIT_REGIME_FLIP",
        "EXIT_THESIS_BROKEN",
        "EXIT_CAUTIOUS",
    ]
    exit_qty_factor: float = Field(ge=0.0, le=1.0)
    reason: str


class JournalAgentOutput(_Strict):
    mode: Literal["FILL", "EXIT", "EOD"]
    title: str
    body_md: str
    tags: list[str]


class NtfyDigestOutput(_Strict):
    title: str = Field(max_length=80)
    body_md: str
    priority: int = Field(ge=1, le=5)
    tags: list[str]


# -----------------------------------------------------------------------------
# Learning critic
# -----------------------------------------------------------------------------


class LearningProposal(_Strict):
    param_name: str
    current_value: float
    proposed_value: float
    rationale: str
    expected_impact: str
    min_canary_trades: int = Field(ge=1)


class LearningCriticOutput(_Strict):
    n_proposed: int = Field(ge=0)
    proposals: list[LearningProposal]
    rejected_changes: list[str] = Field(default_factory=list)
    weekly_summary_md: str


# -----------------------------------------------------------------------------
# Lookup table
# -----------------------------------------------------------------------------


SCHEMA_FOR_ROLE: dict[str, type[BaseModel]] = {
    "regime_reviewer": RegimeReviewerOutput,
    "scout": ScoutOutput,
    "technical_analyst": TechnicalAnalystOutput,
    "news_analyst": NewsAnalystOutput,
    "sentiment_analyst": SentimentAnalystOutput,
    "fundamental_analyst": FundamentalAnalystOutput,
    "bull_researcher": BullResearcherOutput,
    "bear_researcher": BearResearcherOutput,
    "trader_synthesizer": TraderSynthesizerOutput,
    "risk_conservative": RiskConservativeOutput,
    "risk_opportunity": RiskOpportunityOutput,
    "risk_arbiter": RiskArbiterOutput,
    "exit_monitor": ExitMonitorOutput,
    "journal_agent": JournalAgentOutput,
    "ntfy_digest_composer": NtfyDigestOutput,
    "learning_critic": LearningCriticOutput,
}
