"""Schema validation tests for all 16 LLM-role Pydantic models.

Each test exercises a happy-path payload + at least one rejection. These
match the spec emitted by `.claude/agents/*.md` and `.codex/agents/*.md`.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from trading_agent.llm.schemas import (
    BearResearcherOutput,
    BullResearcherOutput,
    ExitMonitorOutput,
    FundamentalAnalystOutput,
    JournalAgentOutput,
    LearningCriticOutput,
    NewsAnalystOutput,
    NtfyDigestOutput,
    RegimeReviewerOutput,
    RiskArbiterOutput,
    RiskConservativeOutput,
    RiskOpportunityOutput,
    SCHEMA_FOR_ROLE,
    ScoutOutput,
    SentimentAnalystOutput,
    TechnicalAnalystOutput,
    TraderSynthesizerOutput,
)


def test_regime_reviewer_happy_path():
    obj = RegimeReviewerOutput.model_validate(
        {
            "review_label": "CONFIRM",
            "confidence_adjustment": 0.0,
            "risk_notes": [],
            "must_defer_new_entries": False,
        }
    )
    assert obj.review_label == "CONFIRM"


def test_regime_reviewer_rejects_positive_adjustment():
    """Reviewer can only DOWNGRADE confidence — positive adjustment is illegal."""
    with pytest.raises(ValidationError):
        RegimeReviewerOutput.model_validate(
            {
                "review_label": "CONFIRM",
                "confidence_adjustment": 0.1,  # > 0 disallowed
                "risk_notes": [],
                "must_defer_new_entries": False,
            }
        )


def test_scout_happy_path():
    obj = ScoutOutput.model_validate(
        {"candidates": [{"ticker": "SPY", "score": 0.7, "reason": "vwap reclaim"}]}
    )
    assert len(obj.candidates) == 1


def test_scout_rejects_score_over_1():
    with pytest.raises(ValidationError):
        ScoutOutput.model_validate(
            {"candidates": [{"ticker": "SPY", "score": 1.5, "reason": "x"}]}
        )


def test_technical_analyst_happy_path():
    TechnicalAnalystOutput.model_validate(
        {
            "ticker": "SPY",
            "trend_1d": "UP",
            "trend_1h": "RANGE",
            "support_levels": [400, 395],
            "resistance_levels": [410],
            "iv_skew_summary": "puts richer than calls",
            "iv_atm_30d": 0.18,
            "hv_30d": 0.12,
            "setup_quality": 0.6,
            "primary_signal": "20d MA support hold",
            "directional_bias": "LONG",
            "chase_risk": False,
            "tech_report_md": "...",
        }
    )


def test_news_sentiment_score_bounds():
    with pytest.raises(ValidationError):
        NewsAnalystOutput.model_validate(
            {
                "ticker": "SPY",
                "sentiment_score": 2.0,  # > 1
                "headline_count": 1,
                "filings_count": 0,
                "primary_catalysts": [],
                "earnings_within_5d": False,
                "summary_md": "...",
            }
        )


def test_trader_synthesizer_decline_path():
    TraderSynthesizerOutput.model_validate(
        {
            "decline_to_trade": True,
            "decline_reason": "regime is CRISIS",
            "proposal": None,
            "reflection_notes": [],
            "proposal_notes": "stand down",
        }
    )


def test_trader_synthesizer_with_proposal():
    obj = TraderSynthesizerOutput.model_validate(
        {
            "decline_to_trade": False,
            "decline_reason": "",
            "proposal": {
                "ticker": "SPY",
                "symbol": "US.SPY260530C00720000",
                "asset_type": "OPT",
                "direction": "LONG_CALL",
                "strategy_label": "long_call_30d",
                "entry_price": 1.5,
                "stop": 0.75,
                "target": 3.0,  # R:R = 1.5/0.75 = 2.0 (passes R7 ≥ 1.5)
                "expected_return_pct": 100.0,
                "max_loss_pct": 50.0,
                "option_delta": 0.4,
                "option_dte": 30,
                "option_iv": 0.18,
                "qty_request": 10,
            },
            "reflection_notes": ["IV high"],
            "proposal_notes": "...",
        }
    )
    assert obj.proposal is not None and obj.proposal.qty_request == 10


# ---------------------------------------------------------------------------
# R7 — Risk:reward floor on TraderProposal
# ---------------------------------------------------------------------------


def _proposal_payload(stop: float, target: float, entry: float = 10.0) -> dict:
    return {
        "ticker": "AAPL", "symbol": "US.AAPL",
        "asset_type": "STK", "direction": "LONG",
        "strategy_label": "trend",
        "entry_price": entry, "stop": stop, "target": target,
        "expected_return_pct": 10.0, "max_loss_pct": 10.0,
        "qty_request": 10,
    }


def test_proposal_r7_rejects_bad_rr():
    """SPY 742C-style 0.6:1 R:R must be rejected at schema time."""
    from pydantic import ValidationError
    from trading_agent.llm.schemas import TraderProposal

    # entry=10, stop=5 (risk=5), target=13 (reward=3) → R:R = 3/5 = 0.6
    with pytest.raises(ValidationError, match="R:R 0.60 below floor"):
        TraderProposal.model_validate(_proposal_payload(stop=5.0, target=13.0))


def test_proposal_r7_accepts_15_floor():
    """Exactly 1.5:1 R:R passes."""
    from trading_agent.llm.schemas import TraderProposal

    # entry=10, stop=8, target=13 → R:R = 3/2 = 1.5
    obj = TraderProposal.model_validate(_proposal_payload(stop=8.0, target=13.0))
    assert obj.stop == 8.0 and obj.target == 13.0


def test_proposal_r7_rejects_zero_risk():
    """stop == entry → no risk defined."""
    from pydantic import ValidationError
    from trading_agent.llm.schemas import TraderProposal

    with pytest.raises(ValidationError, match="too close to entry"):
        TraderProposal.model_validate(_proposal_payload(stop=10.0, target=12.0))


def test_proposal_r7_rejects_zero_reward():
    from pydantic import ValidationError
    from trading_agent.llm.schemas import TraderProposal

    with pytest.raises(ValidationError, match="too close to entry"):
        TraderProposal.model_validate(_proposal_payload(stop=8.0, target=10.0))


def test_proposal_with_exit_plan_must_match_stop_target():
    """If exit_plan is supplied, its hard_stop/target must match the
    top-level stop/target. Catches divergence early."""
    from pydantic import ValidationError
    from trading_agent.llm.schemas import TraderProposal

    payload = _proposal_payload(stop=8.0, target=13.0)
    payload["exit_plan"] = {
        "hard_stop": 7.0,  # ← does NOT match top-level stop=8.0
        "hard_target": 13.0,
    }
    with pytest.raises(ValidationError, match="exit_plan.hard_stop"):
        TraderProposal.model_validate(payload)


def test_proposal_exit_plan_field_optional():
    """Plan is optional — synthesizer can emit just stop/target."""
    from trading_agent.llm.schemas import TraderProposal

    obj = TraderProposal.model_validate(_proposal_payload(stop=8.0, target=13.0))
    assert obj.exit_plan is None


def test_risk_arbiter_happy_path():
    obj = RiskArbiterOutput.model_validate(
        {
            "decision": "APPROVE",
            "approved_qty": 5,
            "max_entry_price": 1.51,
            "primary_risks": ["concentration"],
            "reason": "within all caps",
            "agreed_with": "both",
            "required_conditions": [],
        }
    )
    assert obj.decision == "APPROVE"


def test_risk_arbiter_rejects_negative_qty():
    with pytest.raises(ValidationError):
        RiskArbiterOutput.model_validate(
            {
                "decision": "APPROVE",
                "approved_qty": -1,
                "max_entry_price": 1.0,
                "primary_risks": [],
                "reason": "x",
                "agreed_with": "neither",
                "required_conditions": [],
            }
        )


def test_exit_monitor_action_values():
    ExitMonitorOutput.model_validate({"action": "HOLD", "exit_qty_factor": 0.0, "reason": "ok"})
    with pytest.raises(ValidationError):
        ExitMonitorOutput.model_validate(
            {"action": "DOUBLE_DOWN", "exit_qty_factor": 0.0, "reason": "x"}
        )


def test_journal_mode_enum():
    JournalAgentOutput.model_validate(
        {"mode": "FILL", "title": "x", "body_md": "y", "tags": ["fill"]}
    )
    with pytest.raises(ValidationError):
        JournalAgentOutput.model_validate(
            {"mode": "BAD", "title": "x", "body_md": "y", "tags": []}
        )


def test_ntfy_digest_priority_bounds():
    NtfyDigestOutput.model_validate(
        {"title": "x", "body_md": "y", "priority": 3, "tags": ["bar_chart"]}
    )
    with pytest.raises(ValidationError):
        NtfyDigestOutput.model_validate(
            {"title": "x", "body_md": "y", "priority": 9, "tags": []}
        )


def test_learning_critic_proposals_count():
    LearningCriticOutput.model_validate(
        {
            "n_proposed": 1,
            "proposals": [
                {
                    "param_name": "min_setup_quality",
                    "current_value": 0.5,
                    "proposed_value": 0.55,
                    "rationale": "win-rate softness in low-quality bucket",
                    "expected_impact": "fewer false positives",
                    "min_canary_trades": 25,
                }
            ],
            "rejected_changes": [],
            "weekly_summary_md": "...",
        }
    )


def test_role_to_schema_map_complete():
    expected_roles = {
        "regime_reviewer", "scout", "technical_analyst", "news_analyst",
        "sentiment_analyst", "fundamental_analyst", "bull_researcher",
        "bear_researcher", "trader_synthesizer", "risk_conservative",
        "risk_opportunity", "risk_arbiter", "exit_monitor", "journal_agent",
        "ntfy_digest_composer", "learning_critic",
    }
    assert set(SCHEMA_FOR_ROLE.keys()) == expected_roles
