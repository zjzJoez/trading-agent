"""Pydantic output schemas for the 14 LLM agent roles.

These are the contracts the OAuth router validates against. Each role's
agent markdown definition declares the same shape; if a model returns
anything not matching, the router retries once then raises
LLMSchemaViolation, and the caller (typically the risk pipeline) defaults
to DEFER on schema failure.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    """Reject extra fields; coerce nothing."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# -----------------------------------------------------------------------------
# Exit plan — deterministic exit rules baked at entry time
#
# The intraday hard executor (trading_agent.exits.hard_executor) consumes
# this plan and produces deterministic EXIT_* decisions every 15-min tick.
# No LLM call in the exit path. The plan is built once at entry; never
# re-derived. See docs/position_management.md for the full rule table.
# -----------------------------------------------------------------------------


class ScaleOutRung(_Strict):
    """One rung in the scale-out ladder.

    When mark >= ``at_mark`` and this rung hasn't fired yet, close
    ``exit_factor`` of the *remaining* position. If ``then_engage_trail``
    is True, subsequent ticks start checking the trailing-stop rule for
    the residual quantity.
    """

    at_mark: float = Field(gt=0)
    exit_factor: float = Field(gt=0.0, le=1.0)
    then_engage_trail: bool = False


class TrailStopConfig(_Strict):
    """Trailing stop. Engaged when mark first touches ``engage_at_mark``.

    Once engaged, the trail tracks the highest mark seen on this position
    (sourced from ``journal_trades.mfe_so_far``, converted back to dollars
    via the trade's R-unit). Exit fires when mark falls below
    ``high_water * (1 - distance_pct)``. ``never_below`` floors the trail
    so it can never give back below break-even / entry / the original stop.
    """

    engage_at_mark: float = Field(gt=0)
    distance_pct: float = Field(gt=0, lt=1.0)
    never_below: Literal["break_even", "entry", "original_stop"] = "break_even"


class DteRulesConfig(_Strict):
    """DTE-aware exit rules for options.

    * ``force_exit_at_dte``: always exit when DTE <= this. Default 2 covers
      assignment risk + the gamma cliff near expiry.
    * ``force_exit_at_dte_5_if_delta_below``: at DTE <= 5, force exit when
      |delta| is below this threshold. OTM options near expiry are pure
      theta-decay traps; expected EV is negative.
    * ``switch_to_intrinsic_floor_at_dte``: at DTE <= this AND |delta|
      >= the threshold above, raise the effective stop to
      ``intrinsic - intrinsic_floor_buffer``. ITM/ATM options near expiry
      retain real intrinsic value; we lock that in instead of force-exiting.
    """

    force_exit_at_dte: int = Field(ge=0, default=2)
    force_exit_at_dte_5_if_delta_below: float = Field(ge=0.0, le=1.0, default=0.40)
    switch_to_intrinsic_floor_at_dte: int = Field(ge=0, default=5)
    intrinsic_floor_buffer: float = Field(ge=0.0, default=0.10)


class RegimeRulesConfig(_Strict):
    """Regime-driven kill switches."""

    exit_on_labels: list[str] = Field(default_factory=lambda: ["CRISIS"])
    downsize_50_on_labels: list[str] = Field(default_factory=list)


class ExitPlan(_Strict):
    """Full deterministic exit plan baked at entry time.

    All price-denominated fields are in the position's quote unit (dollars
    per share for stock, dollars per contract for option premiums).
    """

    version: int = 1
    hard_stop: float = Field(gt=0)
    hard_target: float = Field(gt=0)
    scale_out_ladder: list[ScaleOutRung] = Field(default_factory=list)
    trail_stop: TrailStopConfig | None = None
    dte_rules: DteRulesConfig = Field(default_factory=DteRulesConfig)
    regime_rules: RegimeRulesConfig = Field(default_factory=RegimeRulesConfig)
    time_in_trade_max_days: int = Field(ge=1, default=30)


# Minimum risk-reward floor for any opening trade. 1.5:1 means we must
# stand to make 1.5x what we risk on every entry. 5/22 audit: previous
# LLM-emitted trades had 0.6:1 (SPY 742C: stop -55%, target +33%); that's
# negative-EV at any win rate < 62%, far below our actual ~55% baseline.
MIN_RISK_REWARD = 1.5


def default_exit_plan(
    entry: float, stop: float, target: float, asset_type: str
) -> ExitPlan:
    """Generate a sensible default exit plan from (entry, stop, target).

    Used when the trader synthesizer emits only the legacy stop/target
    fields without a full ExitPlan. Designed so behaviour is reasonable
    even without explicit caller-supplied rungs.
    """
    if asset_type == "OPT":
        scale_at = entry + (target - entry) * 0.70
        return ExitPlan(
            hard_stop=stop,
            hard_target=target,
            scale_out_ladder=[
                ScaleOutRung(
                    at_mark=scale_at, exit_factor=0.5, then_engage_trail=True
                ),
            ],
            trail_stop=TrailStopConfig(
                engage_at_mark=scale_at,
                distance_pct=0.15,
                never_below="break_even",
            ),
            time_in_trade_max_days=30,
        )
    # STK
    scale_at = entry + (target - entry) * 0.70
    return ExitPlan(
        hard_stop=stop,
        hard_target=target,
        scale_out_ladder=[],  # whole-share atomicity, no partials
        trail_stop=TrailStopConfig(
            engage_at_mark=scale_at,
            distance_pct=0.10,
            never_below="break_even",
        ),
        time_in_trade_max_days=60,
    )


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
    # Optional — synthesizer may emit a fully-specified plan; otherwise
    # default_exit_plan() builds one from (entry, stop, target, asset_type)
    # in persist_trade_event.
    exit_plan: ExitPlan | None = None

    @model_validator(mode="after")
    def _validate_risk_reward(self) -> "TraderProposal":
        """R7 — minimum risk:reward floor enforced at LLM-output time.

        Forces synthesizer to either tighten stop or widen target so the
        portfolio's per-trade EV is at least neutral at our baseline win
        rate. SPY 742C-style 0.6:1 R:R proposals raise here and the LLM
        either retries with better numbers or declines the trade.
        """
        risk = abs(self.entry_price - self.stop)
        reward = abs(self.target - self.entry_price)
        if risk < 0.01:
            raise ValueError(
                f"stop ({self.stop}) too close to entry ({self.entry_price}); "
                f"risk=${risk:.4f}. Move stop further from entry."
            )
        if reward < 0.01:
            raise ValueError(
                f"target ({self.target}) too close to entry ({self.entry_price}); "
                f"reward=${reward:.4f}. Move target further from entry."
            )
        rr = reward / risk
        if rr < MIN_RISK_REWARD:
            raise ValueError(
                f"R:R {rr:.2f} below floor {MIN_RISK_REWARD} "
                f"(risk=${risk:.2f}, reward=${reward:.2f}). "
                f"Tighten stop OR widen target until reward/risk >= "
                f"{MIN_RISK_REWARD}."
            )
        # If exit_plan is supplied, its hard_stop/hard_target should match
        # the top-level stop/target. Catches divergence early.
        if self.exit_plan is not None:
            if abs(self.exit_plan.hard_stop - self.stop) > 0.01:
                raise ValueError(
                    f"exit_plan.hard_stop ({self.exit_plan.hard_stop}) "
                    f"!= proposal.stop ({self.stop})"
                )
            if abs(self.exit_plan.hard_target - self.target) > 0.01:
                raise ValueError(
                    f"exit_plan.hard_target ({self.exit_plan.hard_target}) "
                    f"!= proposal.target ({self.target})"
                )
        return self


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
