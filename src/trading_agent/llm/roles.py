"""Role → channel + model + agent-file mapping.

Concrete model identifiers are kept here, not hard-coded across the codebase.
Tier abstraction (`MODEL_DEEP`/`MODEL_QUICK`/`MODEL_CHEAP`) maps to specific
model IDs; if Anthropic/OpenAI rotate model names, only this file changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Concrete model IDs as of 2026-04-28
MODEL_OPUS = "opus"  # Claude Code accepts "opus" as a shortcut → claude-opus-4-7
MODEL_SONNET = "sonnet"  # → claude-sonnet-4-6
MODEL_HAIKU = "haiku"  # → claude-haiku-4-5
MODEL_GPT55 = "gpt-5.5"

Channel = Literal["claude_code", "codex"]


@dataclass(frozen=True)
class RoleConfig:
    channel: Channel
    model: str
    agent_name: str  # filename stem under .claude/agents/ or .codex/agents/
    weekly_token_cap: int = 100_000  # soft cap per role


# 14 LLM-using roles. Deterministic roles (Scheduler, Sizing, Executor, etc.)
# don't appear here — they don't call LLMs.
ROLE_CONFIG: dict[str, RoleConfig] = {
    # Claude Code (Max 20x plan)
    "regime_reviewer": RoleConfig("claude_code", MODEL_SONNET, "regime-reviewer", 80_000),
    "scout": RoleConfig("claude_code", MODEL_HAIKU, "scout", 200_000),
    "technical_analyst": RoleConfig("claude_code", MODEL_SONNET, "technical-analyst", 300_000),
    "news_analyst": RoleConfig("claude_code", MODEL_HAIKU, "news-analyst", 200_000),
    "sentiment_analyst": RoleConfig("claude_code", MODEL_HAIKU, "sentiment-analyst", 100_000),
    "bull_researcher": RoleConfig("claude_code", MODEL_SONNET, "bull-researcher", 150_000),
    "trader_synthesizer": RoleConfig("claude_code", MODEL_OPUS, "trader-synthesizer", 200_000),
    "risk_conservative": RoleConfig("claude_code", MODEL_SONNET, "risk-conservative", 100_000),
    "risk_arbiter": RoleConfig("claude_code", MODEL_OPUS, "risk-arbiter", 200_000),
    "exit_monitor": RoleConfig("claude_code", MODEL_HAIKU, "exit-monitor", 250_000),
    "journal_agent": RoleConfig("claude_code", MODEL_HAIKU, "journal-agent", 200_000),
    "ntfy_digest_composer": RoleConfig(
        "claude_code", MODEL_HAIKU, "ntfy-digest-composer", 50_000
    ),

    # Codex (Plus plan) — challenger / cross-family roles
    "fundamental_analyst": RoleConfig("codex", MODEL_GPT55, "fundamental-analyst", 80_000),
    "bear_researcher": RoleConfig("codex", MODEL_GPT55, "bear-researcher", 100_000),
    "risk_opportunity": RoleConfig("codex", MODEL_GPT55, "risk-opportunity", 80_000),
    "learning_critic": RoleConfig("codex", MODEL_GPT55, "learning-critic", 50_000),
}


# When a role's weekly cap or channel cap is hit, degrade to this fallback role.
# `None` = defer (skip the LLM call entirely; downstream defaults to safe path).
DEGRADE_TABLE: dict[str, str | None] = {
    "trader_synthesizer": "risk_conservative",  # Sonnet fallback for synthesis
    "risk_arbiter": "risk_conservative",  # if Opus is over-cap, use Sonnet conservative-only
    "bear_researcher": "risk_conservative",  # Cross-family unavailable → Claude Sonnet
    "fundamental_analyst": "news_analyst",  # closest equivalent on Claude
    "risk_opportunity": None,  # defer council (deterministic decision wins)
    "learning_critic": None,  # defer to next week
    # Other roles: no degrade — if their cap is hit, the entire run defers
}
