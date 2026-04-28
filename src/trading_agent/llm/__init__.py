"""Phase 2.4 — OAuth-based LLM router.

All LLM calls go through this package. Two channels:
- Claude Code (Max 20x plan) via `claude -p` subprocess
- Codex (Plus plan) via `codex exec` subprocess

No Anthropic/OpenAI API keys are used anywhere; OAuth login is a one-time
manual step the user does on EC2 (see `EC2_OAUTH_HANDOFF.md`).
"""
from trading_agent.llm.oauth_router import (  # noqa: F401
    OAuthLLMRouter,
    LLMSchemaViolation,
    StubLLMRouter,
    get_router,
)
from trading_agent.llm.budget import WeeklyBudget  # noqa: F401
from trading_agent.llm.roles import ROLE_CONFIG, RoleConfig  # noqa: F401
