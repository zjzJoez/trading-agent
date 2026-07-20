"""Pydantic output schemas for the 14 LLM agent roles.

These are the contracts the OAuth router validates against. Each role's
agent markdown definition declares the same shape; if a model returns
anything not matching, the router retries once then raises
LLMSchemaViolation, and the caller (typically the risk pipeline) defaults
to DEFER on schema failure.
"""
from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

log = logging.getLogger(__name__)


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


class EventRulesConfig(_Strict):
    """Event-driven exits. Fires when an out-of-band signal marks the thesis
    broken (journal_theses.status == 'thesis_broken'), set by the operator via
    journal-mcp ``mark_thesis_broken`` or a future premarket news node. The
    intraday executor only READS the flag — no LLM/network on the hot path."""

    exit_on_thesis_broken: bool = True
    exit_factor: float = Field(gt=0.0, le=1.0, default=1.0)


class ExitPlan(_Strict):
    """Full deterministic exit plan baked at entry time.

    All price-denominated fields are in the position's quote unit (dollars
    per share for stock, dollars per contract for option premiums).

    ``direction`` controls the executor's comparison polarity:
      * LONG:  mark <= stop triggers, mark >= target triggers
      * SHORT: mark >= stop triggers, mark <= target triggers
        (you're short the option → price rising is bad)
    Defaults to LONG so legacy plans (no direction field) keep working.
    """

    version: int = 1
    direction: Literal["LONG", "SHORT"] = "LONG"
    hard_stop: float = Field(gt=0)
    hard_target: float = Field(gt=0)
    scale_out_ladder: list[ScaleOutRung] = Field(default_factory=list)
    trail_stop: TrailStopConfig | None = None
    dte_rules: DteRulesConfig = Field(default_factory=DteRulesConfig)
    regime_rules: RegimeRulesConfig = Field(default_factory=RegimeRulesConfig)
    event_rules: EventRulesConfig = Field(default_factory=EventRulesConfig)
    time_in_trade_max_days: int = Field(ge=1, default=30)


# ABSOLUTE minimum risk-reward floor for LONG opening trades — the floor
# for labels with no strategy spec. Labels mapped to a spec in
# strategy_specs.REGISTRY use the SPEC's min_risk_reward when higher (the
# convexity track demands 2.0): _validate_geometry resolves the effective
# floor via _effective_min_rr(), so a too-thin proposal fails HERE, inside
# the router's schema-retry loop where the LLM can widen the target
# in-flight — not after the full analyst/debate/synthesis pipeline has
# burned its tokens only to die at deterministic sizing's R7.
#
# History: 5/22 raised from 1.5 to 1.3 after the SPY 742C 0.6:1 audit,
# justified at the time by an UNCITED "~55% baseline win rate". The Phase-2
# strategy specs replaced that arithmetic: each spec declares a falsifiable
# expectancy profile with a friction-aware breakeven win rate
# (strategy_specs.py is canonical).
#
# SHORT premium positions are not gated by R7 — see _validate_risk_reward.
# R:R is structurally capped (reward = premium collected ≤ strike), so a
# CSP at strike 100 selling for $2 would always fail R7 1.3:1 even when
# the EV is clearly positive.
MIN_RISK_REWARD = 1.3


def _effective_min_rr(strategy_label: str | None) -> float:
    """Spec floor when the label maps to one, else the global floor.

    Never raises and never loosens: registry errors fall back to
    MIN_RISK_REWARD, and a spec can only RAISE the floor (max()).
    """
    try:
        from trading_agent.strategy_specs import spec_for_label
        spec = spec_for_label(strategy_label)
        if spec is not None and spec.min_risk_reward is not None:
            return max(MIN_RISK_REWARD, float(spec.min_risk_reward))
    except Exception:  # noqa: BLE001 — schema validation must never crash
        pass
    return MIN_RISK_REWARD


def default_exit_plan(
    entry: float, stop: float, target: float, asset_type: str,
    direction: str = "LONG",
) -> ExitPlan:
    """Generate a sensible default exit plan from (entry, stop, target).

    Used when the trader synthesizer emits only the legacy stop/target
    fields without a full ExitPlan.

    For LONG positions: target > entry > stop.
    For SHORT positions: stop > entry > target (you want premium to decay).
    """
    if direction == "SHORT":
        # Short option: collect premium, want it to decay to ~$0.
        # scale_at: capture ~50% of max profit (entry - 0.5×(entry-target))
        scale_at = entry - (entry - target) * 0.50
        return ExitPlan(
            direction="SHORT",
            hard_stop=stop,
            hard_target=target,
            scale_out_ladder=[
                ScaleOutRung(
                    at_mark=scale_at, exit_factor=0.5, then_engage_trail=True
                ),
            ],
            trail_stop=TrailStopConfig(
                engage_at_mark=scale_at,
                distance_pct=0.25,           # wider trail for short premium
                never_below="break_even",
            ),
            time_in_trade_max_days=30,
        )

    # LONG (stk or opt)
    if asset_type == "OPT":
        scale_at = entry + (target - entry) * 0.70
        return ExitPlan(
            direction="LONG",
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
    # STK long
    scale_at = entry + (target - entry) * 0.70
    return ExitPlan(
        direction="LONG",
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


def combo_exit_plan(net_credit: float, width: float) -> ExitPlan:
    """Exit plan for a defined-risk credit vertical, baked at OPEN time.

    The plan is written against the WHOLE spread: entry = net_credit, mark =
    spread value (short mark − long mark), direction SHORT. The hard executor
    then prices both M1-0.1 triggers with zero code change:

      * 50% profit-take: ``hard_target = 0.5 × net_credit`` — the SHORT
        target branch fires on ``mark <= target``.
      * 21-DTE force-close: ``force_exit_at_dte = 21`` — the SHORT DTE
        branch fires from the short leg's symbol (the synthetic combo
        position keeps the short-leg symbol for DTE parsing).

    ``hard_stop = width`` is deliberately inert: a vertical's spread value
    cannot exceed its width — the defined-risk structure IS the stop.

    Do NOT use ``default_exit_plan(direction="SHORT")`` here: it attaches a
    0.5 scale-out rung + trailing stop, which violates "whole units only /
    never leg apart" on 1–2 lot verticals (a rung would try to close half
    a spread). No ladder, no trail — full closes and P0b whole-unit trims
    only.
    """
    return ExitPlan(
        direction="SHORT",
        hard_stop=width,
        hard_target=round(0.5 * net_credit, 2),
        scale_out_ladder=[],
        trail_stop=None,
        dte_rules=DteRulesConfig(force_exit_at_dte=21),
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
    # LONG_PUT/CALL = buy premium; SHORT_PUT/CALL = sell premium (collect).
    # SHORT_PUT is the cash-secured-put case (CSP); SHORT_CALL is naked
    # (sizing enforces stress-buffered max_loss to bound the unbounded
    # theoretical risk).
    direction: Literal[
        "LONG", "LONG_CALL", "LONG_PUT", "SHORT_CALL", "SHORT_PUT"
    ]
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

    @property
    def is_short(self) -> bool:
        return self.direction in ("SHORT_CALL", "SHORT_PUT")

    @model_validator(mode="after")
    def _validate_geometry(self) -> "TraderProposal":
        """R7 R:R floor + plan/proposal consistency.

        LONG positions: target > entry > stop, R:R >= MIN_RISK_REWARD.
        SHORT positions: stop > entry > target (premium decays toward
            zero = profit); R7 is SKIPPED because R:R is structurally
            capped on premium-collecting trades (reward = premium ≤
            strike, risk = up to strike × 100). Sizing R5b/R1 stress
            buffer handles the real risk bound for shorts.
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

        # Direction-specific geometry: LONG wants target above entry;
        # SHORT wants target BELOW entry (premium decay).
        if self.is_short:
            if self.target >= self.entry_price:
                raise ValueError(
                    f"SHORT direction requires target ({self.target}) < "
                    f"entry ({self.entry_price}) — you profit from premium "
                    f"decaying toward zero."
                )
            if self.stop <= self.entry_price:
                raise ValueError(
                    f"SHORT direction requires stop ({self.stop}) > "
                    f"entry ({self.entry_price}) — you lose when the "
                    f"option price rises against you."
                )
        else:
            # LONG geometry + R7 R:R floor.
            if self.target <= self.entry_price:
                raise ValueError(
                    f"LONG direction requires target ({self.target}) > "
                    f"entry ({self.entry_price})."
                )
            if self.stop >= self.entry_price:
                raise ValueError(
                    f"LONG direction requires stop ({self.stop}) < "
                    f"entry ({self.entry_price})."
                )
            rr = reward / risk
            min_rr = _effective_min_rr(getattr(self, "strategy_label", None))
            if rr < min_rr:
                raise ValueError(
                    f"R:R {rr:.2f} below floor {min_rr} for "
                    f"strategy_label={getattr(self, 'strategy_label', None)!r} "
                    f"(risk=${risk:.2f}, reward=${reward:.2f}). "
                    f"Tighten stop OR widen target until reward/risk >= "
                    f"{min_rr}."
                )

        # Greeks are MANDATORY on OPT proposals (fail-closed). The spec-band
        # gate below, build_trade_proposal's backstop AND sizing's R5 all
        # skip a None dte/delta — so a model that simply drops the fields
        # would evade every band check. The retry loop feeds this error
        # back to the LLM, which always has the chain data to re-emit.
        if self.asset_type == "OPT":
            missing = [
                name for name in ("option_delta", "option_dte")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"{', '.join(missing)} missing: option_delta and "
                    f"option_dte are mandatory for OPT proposals — the "
                    f"DTE/delta band gates cannot run without them. "
                    f"Re-emit the proposal with the contract's delta and "
                    f"DTE from the option chain."
                )

        # Spec-band gate (Week-1 Step 6b): a label mapped to a StrategySpec
        # must sit inside the spec's declared DTE/delta bands; an UNMAPPED
        # label on an OPT proposal falls back to the global R5 band — no
        # label is a free pass. Raising HERE keeps the failure inside the
        # router's schema-retry loop (same rationale as _effective_min_rr):
        # the LLM can pick a compliant contract in-flight instead of dying
        # at build_trade_proposal's post-parse backstop.
        try:
            from trading_agent.strategy_specs import spec_band_violations
            band_violations = spec_band_violations(
                strategy_label=self.strategy_label,
                asset_type=self.asset_type,
                option_dte=self.option_dte,
                option_delta=self.option_delta,
            )
        except Exception as exc:  # noqa: BLE001 — schema validation must never crash
            log.warning(
                "spec-band schema check crashed, deferring to "
                "build_trade_proposal backstop: %s", exc,
            )
            band_violations = []
        if band_violations:
            raise ValueError(
                "; ".join(str(v["message"]) for v in band_violations)
                + ". Pick a contract inside the band."
            )

        # If exit_plan is supplied, its hard_stop/hard_target should match
        # the top-level stop/target, AND its direction must match.
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
            expected_dir = "SHORT" if self.is_short else "LONG"
            if self.exit_plan.direction != expected_dir:
                raise ValueError(
                    f"exit_plan.direction ({self.exit_plan.direction}) "
                    f"!= proposal direction ({expected_dir})"
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
