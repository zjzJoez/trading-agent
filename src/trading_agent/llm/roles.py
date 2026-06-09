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
MODEL_DEEPSEEK_V4_PRO = "deepseek-v4-pro"

Channel = Literal["claude_code", "codex", "deepseek"]


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
    # Scout moved Haiku → Sonnet on 2026-06-02. Haiku returned empty /
    # malformed / list-wrapped JSON on ~13 of 25 trading days (the
    # 40-ticker watchlist table → strict JSON is past Haiku's reliable
    # instruction-following). json_repair salvaged the garbage into a
    # structurally-valid but EMPTY candidates list, which silently fell
    # back to score=0.5 (below DISPATCH_MIN_SCORE 0.55) → 0 dispatches
    # for 11 days. Sonnet 4.6 follows the JSON contract far more reliably.
    "scout": RoleConfig("claude_code", MODEL_SONNET, "scout", 200_000),
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

    # DeepSeek (universal-fallback channel) — schema-matched degrade
    # targets for every role with a DEGRADE_TABLE entry.  Each reuses the
    # SAME agent file as its source role (looked up first under
    # .codex/agents/<name>.md, then .claude/agents/<name>.md) so the
    # produced JSON validates against the source role's pydantic schema.
    # Uses DEEPSEEK_API_KEY from env, OpenAI-compatible chat completions.
    "trader_synthesizer_deepseek": RoleConfig(
        "deepseek", MODEL_DEEPSEEK_V4_PRO, "trader-synthesizer", 200_000
    ),
    "risk_arbiter_deepseek": RoleConfig(
        "deepseek", MODEL_DEEPSEEK_V4_PRO, "risk-arbiter", 200_000
    ),
    "bear_researcher_deepseek": RoleConfig(
        "deepseek", MODEL_DEEPSEEK_V4_PRO, "bear-researcher", 100_000
    ),
    "fundamental_analyst_deepseek": RoleConfig(
        "deepseek", MODEL_DEEPSEEK_V4_PRO, "fundamental-analyst", 80_000
    ),
    "risk_opportunity_deepseek": RoleConfig(
        "deepseek", MODEL_DEEPSEEK_V4_PRO, "risk-opportunity", 80_000
    ),
    "learning_critic_deepseek": RoleConfig(
        "deepseek", MODEL_DEEPSEEK_V4_PRO, "learning-critic", 50_000
    ),

    # Claude (claude_code) degrade targets for the codex CHALLENGER roles.
    # When codex is unavailable (expired OAuth, quota, network) these roles
    # fall back to CLAUDE rather than DeepSeek — same agent file (so the source
    # role's pydantic schema still validates), Sonnet tier. The claude_code
    # primaries keep DeepSeek as their universal fallback; the codex challengers
    # fall back to Claude. (Set 2026-06-09 after codex OAuth + DeepSeek balance
    # both lapsed; Claude is the reliable always-on channel here.)
    "fundamental_analyst_claude": RoleConfig(
        "claude_code", MODEL_SONNET, "fundamental-analyst", 80_000
    ),
    "bear_researcher_claude": RoleConfig(
        "claude_code", MODEL_SONNET, "bear-researcher", 100_000
    ),
    "risk_opportunity_claude": RoleConfig(
        "claude_code", MODEL_SONNET, "risk-opportunity", 80_000
    ),
    "learning_critic_claude": RoleConfig(
        "claude_code", MODEL_SONNET, "learning-critic", 50_000
    ),
}


# When a role's weekly cap or channel cap is hit (OR codex itself returns a
# quota-shaped error), degrade to this fallback role.
#
# Updated 2026-04-30: every degrade now lands on a schema-matched DeepSeek
# variant. Single universal fallback channel keeps the topology simple —
# any LLM role that fails its primary path tries DeepSeek with the SAME
# agent file (so the same pydantic schema validates against the result).
# DeepSeek API key in EC2 .env (DEEPSEEK_API_KEY); model
# defaults to deepseek-v4-pro.
#
# `None` = defer (skip the LLM call entirely; downstream defaults to safe path).
DEGRADE_TABLE: dict[str, str | None] = {
    # claude_code primaries → DeepSeek (cross-family universal fallback).
    "trader_synthesizer": "trader_synthesizer_deepseek",
    "risk_arbiter": "risk_arbiter_deepseek",
    # codex challengers → Claude (DeepSeek is unreliable/unfunded; Claude is the
    # always-on channel). Changed 2026-06-09.
    "bear_researcher": "bear_researcher_claude",
    "fundamental_analyst": "fundamental_analyst_claude",
    "risk_opportunity": "risk_opportunity_claude",
    "learning_critic": "learning_critic_claude",
    # Other roles (regime_reviewer, scout, technical_analyst, etc.): no
    # degrade configured — if their cap is hit, the entire run defers.
    # Add `<role>_deepseek` entries above + map them here if needed.
}
