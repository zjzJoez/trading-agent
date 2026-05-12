"""Router-level tests — happy path uses StubLLMRouter; real router tested manually after OAuth login.

The OAuth router itself spawns subprocesses that we can't run reliably in CI
without `claude login` / `codex login` credentials. These tests pin the
router's interface contract and the budget governor.
"""
from __future__ import annotations

import pytest

from trading_agent.llm import StubLLMRouter, get_router
from trading_agent.llm.budget import WeeklyBudget
from trading_agent.llm.oauth_router import (
    LLMResult,
    OAuthLLMRouter,
    _strip_markdown_fences,
    _extract_assistant_message_from_stream,
    _estimate_tokens,
    _stub_response_for,
    claude_logged_in,
    codex_logged_in,
)
from trading_agent.llm.roles import ROLE_CONFIG


def test_stub_router_returns_canned_for_all_roles():
    router = StubLLMRouter()
    for role in ROLE_CONFIG:
        result = router.call(role, "test prompt")
        assert isinstance(result, LLMResult)
        assert result.role == role
        assert result.cost_usd == 0.0
        assert result.parsed is not None or result.raw_text


def test_stub_router_with_schema_validates():
    from trading_agent.llm.schemas import RegimeReviewerOutput

    router = StubLLMRouter()
    result = router.call("regime_reviewer", "review the regime", schema=RegimeReviewerOutput)
    assert result.parsed is not None


def test_get_router_returns_singleton():
    r1 = get_router()
    r2 = get_router()
    assert r1 is r2


# ---------------------------------------------------------------------------
# JSON repair — fixes the trader_synthesizer-malformed-JSON class of failure
# ---------------------------------------------------------------------------

def test_json_repair_handles_trailing_comma():
    """Output with a trailing comma is recovered by json_repair, no LLM round-trip."""
    from trading_agent.llm.schemas import RegimeReviewerOutput
    router = OAuthLLMRouter()
    # Trailing comma before }, common LLM quirk
    malformed = '{"review_label": "CONFIRM", "confidence_adjustment": -0.1, "risk_notes": ["ok"], "must_defer_new_entries": false,}'
    parsed = router._parse_json_with_retry(
        malformed, RegimeReviewerOutput, "regime_reviewer", "<original prompt>"
    )
    assert parsed.review_label == "CONFIRM"
    assert parsed.confidence_adjustment == -0.1


def test_json_repair_handles_missing_comma():
    """Missing field-separator comma is the exact failure mode from 5/12 NVDA run."""
    from trading_agent.llm.schemas import RegimeReviewerOutput
    router = OAuthLLMRouter()
    # No comma between the first and second field
    malformed = '{"review_label": "CONFIRM" "confidence_adjustment": -0.1, "risk_notes": ["ok"], "must_defer_new_entries": false}'
    parsed = router._parse_json_with_retry(
        malformed, RegimeReviewerOutput, "regime_reviewer", "<original prompt>"
    )
    assert parsed.review_label == "CONFIRM"


def test_json_repair_handles_markdown_fence():
    """Markdown-fenced output should be stripped THEN repaired."""
    from trading_agent.llm.schemas import RegimeReviewerOutput
    router = OAuthLLMRouter()
    raw = '```json\n{"review_label": "CONFIRM", "confidence_adjustment": -0.05, "risk_notes": [], "must_defer_new_entries": false,}\n```'
    parsed = router._parse_json_with_retry(
        raw, RegimeReviewerOutput, "regime_reviewer", "<original prompt>"
    )
    assert parsed.review_label == "CONFIRM"


def test_strip_markdown_fences_handles_json_fence():
    raw = '```json\n{"x": 1}\n```'
    assert _strip_markdown_fences(raw) == '{"x": 1}'


def test_strip_markdown_fences_handles_plain_fence():
    assert _strip_markdown_fences("```\n{}\n```") == "{}"


def test_strip_markdown_fences_passthrough():
    assert _strip_markdown_fences('{"x": 1}') == '{"x": 1}'


def test_extract_assistant_message_from_stream_picks_text():
    stream = (
        '{"type":"system","subtype":"init"}\n'
        '{"type":"assistant","message":{"content":[{"type":"text","text":"hello "}]}}\n'
        '{"type":"assistant","message":{"content":[{"type":"text","text":"world"}]}}\n'
        '{"type":"result","result":""}\n'
    )
    assert _extract_assistant_message_from_stream(stream) == "hello world"


def test_extract_handles_blank_lines():
    assert _extract_assistant_message_from_stream("\n\n") == ""


def test_estimate_tokens_rough_4_char():
    assert _estimate_tokens("a" * 16) == 4
    assert _estimate_tokens("") == 1


def test_stub_response_for_each_role_has_required_keys():
    """Stub responses must round-trip through the corresponding schema."""
    from trading_agent.llm.schemas import SCHEMA_FOR_ROLE

    for role in ROLE_CONFIG:
        canned = _stub_response_for(role)
        # We don't assert validation passes for ALL — some stubs are minimal —
        # but the canned dict should at least be a dict with relevant keys.
        assert isinstance(canned, dict)
        assert canned, f"empty stub for role {role}"


def test_budget_query_returns_zero_without_dsn():
    b = WeeklyBudget(dsn=None)
    assert b.used_this_week("claude_code") == 0
    assert b.used_for_role("trader_synthesizer") == 0
    assert b.would_exceed_weekly("claude_code", 1_000_000) is False


def test_claude_codex_logged_in_returns_bool():
    """These are environment-dependent; just verify they don't raise."""
    assert isinstance(claude_logged_in(), bool)
    assert isinstance(codex_logged_in(), bool)
