"""OAuth-subprocess LLM router.

Wraps `claude -p` and `codex exec` CLI invocations behind a uniform
`router.call(role, prompt, schema)` interface. Until the user runs
`claude login` and `codex login` on EC2 (one-time, ~5min), the router runs
in stub mode and returns deterministic placeholder responses so the rest
of the system can run end-to-end.

Detection of "logged in" state:
- Claude Code: `claude --version` returns 0 AND a subsequent `-p` smoke call
  doesn't error with the auth-required message. We check via
  `claude config get oauth.refreshToken` if available.
- Codex: `codex auth status` (CLI subcommand). If absent, fall back to a
  trivial smoke-test exec.

The router is process-safe (asyncio.Semaphore caps concurrent CLI procs at
2 to match the t3.large 2-vCPU constraint).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from trading_agent.llm.budget import WeeklyBudget
from trading_agent.llm.roles import DEGRADE_TABLE, ROLE_CONFIG, RoleConfig

log = logging.getLogger(__name__)


class LLMSchemaViolation(RuntimeError):
    """Raised when a model's output cannot be parsed into the expected schema."""


@dataclass
class LLMResult:
    role: str
    raw_text: str
    parsed: BaseModel | dict | None
    tokens_estimated: int
    cost_usd: float
    degraded: bool = False
    degrade_target: str | None = None


# -----------------------------------------------------------------------------
# Concurrency: max 2 simultaneous CLI processes (matches 2 vCPU on t3.large)
# -----------------------------------------------------------------------------
_PROC_SEMAPHORE_LOCK = threading.Lock()
_PROC_SEMAPHORE: threading.BoundedSemaphore | None = None


def _proc_semaphore(max_concurrent: int = 2) -> threading.BoundedSemaphore:
    global _PROC_SEMAPHORE
    with _PROC_SEMAPHORE_LOCK:
        if _PROC_SEMAPHORE is None:
            _PROC_SEMAPHORE = threading.BoundedSemaphore(max_concurrent)
    return _PROC_SEMAPHORE


# -----------------------------------------------------------------------------
# Auth detection helpers
# -----------------------------------------------------------------------------


def claude_logged_in() -> bool:
    if not shutil.which("claude"):
        return False
    try:
        # cheapest check: --version succeeds even when not logged in, so we
        # try a tiny --help-style smoke
        proc = subprocess.run(
            ["claude", "--version"], capture_output=True, timeout=5, text=True
        )
        if proc.returncode != 0:
            return False
    except Exception:
        return False
    # Look for credentials file (canonical for Claude Code)
    for path in (
        os.path.expanduser("~/.claude/.credentials.json"),
        os.path.expanduser("~/.config/claude/credentials.json"),
    ):
        if os.path.exists(path):
            return True
    return False


def codex_logged_in() -> bool:
    if not shutil.which("codex"):
        return False
    # Codex CLI stores credentials under ~/.codex
    for path in (
        os.path.expanduser("~/.codex/credentials.json"),
        os.path.expanduser("~/.codex/auth.json"),
    ):
        if os.path.exists(path):
            return True
    try:
        proc = subprocess.run(
            ["codex", "auth", "status"], capture_output=True, timeout=5, text=True
        )
        return proc.returncode == 0 and ("logged in" in proc.stdout.lower())
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Stub router — used until OAuth is configured on EC2
# -----------------------------------------------------------------------------


class StubLLMRouter:
    """Deterministic placeholder. Returns canned responses; never calls a CLI."""

    def __init__(self):
        self.budget = WeeklyBudget()

    def call(
        self,
        role: str,
        user_prompt: str,
        schema: type[BaseModel] | None = None,
        max_tokens: int = 4000,
        timeout_s: int = 120,
    ) -> LLMResult:
        cfg = ROLE_CONFIG[role]
        log.info("[stub_router] role=%s channel=%s model=%s prompt_len=%d",
                 role, cfg.channel, cfg.model, len(user_prompt))
        canned = _stub_response_for(role)
        parsed: BaseModel | dict | None = canned
        if schema is not None:
            try:
                parsed = schema.model_validate(canned)
            except ValidationError:
                parsed = canned  # leave raw dict; caller decides
        return LLMResult(
            role=role,
            raw_text=json.dumps(canned),
            parsed=parsed,
            tokens_estimated=0,
            cost_usd=0.0,
        )


# -----------------------------------------------------------------------------
# Real router
# -----------------------------------------------------------------------------


class OAuthLLMRouter:
    """OAuth-subscription router. Spawns claude/codex CLI subprocesses."""

    # Patterns we treat as "this channel is over quota / unavailable".
    # Conservative — a false positive only burns one extra subprocess; a
    # false negative makes us re-spawn slow CLIs that will keep failing.
    _CODEX_QUOTA_PATTERNS: tuple[str, ...] = (
        "rate limit",
        "rate_limit",
        "quota",
        "usage limit",
        "weekly limit",
        "you have exceeded",
        "exceeded your",
        "limit reached",
        "429",
    )

    def __init__(self, max_concurrent: int = 2):
        self.budget = WeeklyBudget(dsn=os.environ.get("TRADING_PG_DSN"))
        self._sem = _proc_semaphore(max_concurrent)
        # Per-process "channel unhealthy" cache.  Once a codex call returns
        # a quota-shaped error, every subsequent codex role call inside the
        # same Python process immediately degrades without spawning the CLI
        # (each subprocess attempt costs ~3-10 s startup + 2x the timeout).
        self._channel_unhealthy: dict[str, str] = {}
        # Operator override: LLM_DISABLE_CODEX=1 forces every codex role to
        # degrade immediately. Documented in plan §11.5.
        if os.environ.get("LLM_DISABLE_CODEX", "").strip() in ("1", "true", "yes"):
            self._channel_unhealthy["codex"] = "disabled by LLM_DISABLE_CODEX env"

    # -------------- public API --------------

    def call(
        self,
        role: str,
        user_prompt: str,
        schema: type[BaseModel] | None = None,
        max_tokens: int = 4000,
        timeout_s: int = 120,
    ) -> LLMResult:
        cfg = ROLE_CONFIG[role]

        # Pre-flight: budget & degrade
        if self.budget.would_exceed_weekly(cfg.channel, max_tokens):
            return self._degrade(role, user_prompt, schema, max_tokens, timeout_s)
        if self.budget.would_exceed_role(role, cfg.weekly_token_cap, max_tokens):
            return self._degrade(role, user_prompt, schema, max_tokens, timeout_s)
        # Pre-flight: per-process channel-unhealthy cache (set by an
        # earlier failed codex call that matched _CODEX_QUOTA_PATTERNS).
        if cfg.channel in self._channel_unhealthy:
            log.warning(
                "[%s] channel %s flagged unhealthy (%s) — degrading without CLI spawn",
                role, cfg.channel, self._channel_unhealthy[cfg.channel],
            )
            return self._degrade(role, user_prompt, schema, max_tokens, timeout_s)

        try:
            with self._sem:
                if cfg.channel == "claude_code":
                    raw = self._invoke_cc(cfg, user_prompt, timeout_s)
                elif cfg.channel == "codex":
                    raw = self._invoke_codex(cfg, user_prompt, timeout_s)
                else:
                    raise RuntimeError(f"Unknown channel: {cfg.channel}")
        except RuntimeError as e:
            # Detect quota / rate-limit shaped failures from the subprocess
            # and route through the degrade table.  Other RuntimeError shapes
            # (network, parsing, real bugs) propagate so we see them.
            err_str = str(e).lower()
            if cfg.channel == "codex" and any(p in err_str for p in self._CODEX_QUOTA_PATTERNS):
                log.warning(
                    "[%s] codex returned quota-shaped error: %s — flagging codex "
                    "unhealthy for this process and degrading",
                    role, str(e)[:200],
                )
                self._channel_unhealthy["codex"] = str(e)[:200]
                return self._degrade(role, user_prompt, schema, max_tokens, timeout_s)
            raise

        parsed: BaseModel | dict | None = None
        if schema is not None:
            parsed = self._parse_json_with_retry(raw, schema, role, user_prompt)
        else:
            parsed = {"text": raw}

        tokens = _estimate_tokens(user_prompt) + _estimate_tokens(raw)
        self.budget.record(role=role, channel=cfg.channel, tokens=tokens)

        return LLMResult(
            role=role,
            raw_text=raw,
            parsed=parsed,
            tokens_estimated=tokens,
            cost_usd=0.0,  # OAuth subscription = no per-call cost
        )

    # -------------- channel invocations --------------

    def _invoke_cc(self, cfg: RoleConfig, user_prompt: str, timeout_s: int) -> str:
        # claude -p --output-format=json (single-shot JSON, no --verbose needed)
        cmd = [
            "claude", "-p", user_prompt,
            "--model", cfg.model,
            "--agent", cfg.agent_name,
            "--output-format", "json",
            "--dangerously-skip-permissions",
        ]
        env = {**os.environ, "CLAUDE_CODE_LOG_LEVEL": "warn"}
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s, text=True, env=env)
        if proc.returncode != 0:
            log.error("claude returncode=%d stderr=%s", proc.returncode, proc.stderr[:500])
            raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr[:200]}")
        return _extract_text_from_claude_json(proc.stdout)

    def _invoke_codex(self, cfg: RoleConfig, user_prompt: str, timeout_s: int) -> str:
        # Codex CLI has no --system-from-file flag — read the agent file and
        # inline the system prompt at the top of the user prompt.
        system_prompt = _read_codex_system_prompt(cfg.agent_name)
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        cmd = [
            "codex", "exec",
            "--model", cfg.model,
            "--skip-git-repo-check",
            "--ephemeral",
            "-",  # read prompt from stdin (cleaner for long prompts)
        ]
        proc = subprocess.run(
            cmd, input=full_prompt, capture_output=True,
            timeout=timeout_s, text=True,
        )
        if proc.returncode != 0:
            log.error("codex returncode=%d stderr=%s", proc.returncode, proc.stderr[:500])
            raise RuntimeError(f"codex exited {proc.returncode}: {proc.stderr[:200]}")
        return _extract_codex_last_message(proc.stdout)

    # -------------- schema validation --------------

    def _parse_json_with_retry(
        self, raw: str, schema: type[BaseModel], role: str, original_prompt: str
    ) -> BaseModel:
        for attempt in (1, 2):
            try:
                cleaned = _strip_markdown_fences(raw)
                obj = json.loads(cleaned)
                return schema.model_validate(obj)
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == 1:
                    log.warning("[%s] schema retry: %s", role, e)
                    retry_prompt = (
                        original_prompt
                        + "\n\nYour previous response failed validation:\n"
                        + str(e)[:500]
                        + "\n\nReturn ONLY the JSON object — no fences, no preamble."
                    )
                    cfg = ROLE_CONFIG[role]
                    if cfg.channel == "claude_code":
                        raw = self._invoke_cc(cfg, retry_prompt, 120)
                    else:
                        raw = self._invoke_codex(cfg, retry_prompt, 120)
                    continue
                log.error("[%s] schema violation after retry: %s", role, e)
                raise LLMSchemaViolation(f"role={role}: {e}") from e
        raise LLMSchemaViolation(f"role={role}: unreachable")

    # -------------- degrade --------------

    def _degrade(
        self,
        role: str,
        user_prompt: str,
        schema: type[BaseModel] | None,
        max_tokens: int,
        timeout_s: int,
    ) -> LLMResult:
        target = DEGRADE_TABLE.get(role)
        if target is None:
            log.warning("[%s] over budget, no degrade target — deferring", role)
            try:
                from trading_agent.notify import send as ntfy_send

                ntfy_send(
                    "learning",
                    title=f"Budget cap hit: {role}",
                    body=f"Role {role} exceeded weekly cap; deferring this run.",
                    priority=3,
                )
            except Exception:
                pass
            return LLMResult(
                role=role,
                raw_text="",
                parsed=None,
                tokens_estimated=0,
                cost_usd=0.0,
                degraded=True,
                degrade_target=None,
            )
        log.info("[%s] degrading → %s", role, target)
        result = self.call(target, user_prompt, schema, max_tokens, timeout_s)
        result.degraded = True
        result.degrade_target = target
        return result


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    """Strip ```json ... ``` or ``` ... ``` fences if present."""
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def _extract_assistant_message_from_stream(stream_json_output: str) -> str:
    """Parse Claude Code stream-json output and return the final assistant text."""
    parts: list[str] = []
    for line in stream_json_output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "assistant" and evt.get("message"):
            for block in evt["message"].get("content", []):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
        elif evt.get("type") == "result" and evt.get("result"):
            parts.append(evt["result"])
    return "".join(parts).strip()


def _extract_text_from_claude_json(json_output: str) -> str:
    """Parse `claude -p --output-format=json` single-result envelope.

    Claude Code returns a single JSON object with shape
    `{"type": "result", "subtype": "success", "result": "<text>", ...}`.
    """
    try:
        obj = json.loads(json_output.strip())
    except json.JSONDecodeError:
        return json_output.strip()
    if isinstance(obj, dict):
        if "result" in obj and isinstance(obj["result"], str):
            return obj["result"].strip()
        # Fallback: some versions return {"text": ...}
        if "text" in obj and isinstance(obj["text"], str):
            return obj["text"].strip()
    return json_output.strip()


def _read_codex_system_prompt(agent_name: str) -> str:
    """Read .codex/agents/<agent_name>.md and strip YAML frontmatter."""
    paths = [
        f".codex/agents/{agent_name}.md",
        os.path.expanduser(f"~/trading-agent/.codex/agents/{agent_name}.md"),
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                content = f.read()
            return _strip_yaml_frontmatter(content)
    log.warning("codex agent file not found for %s; falling back to empty system prompt", agent_name)
    return ""


def _strip_yaml_frontmatter(text: str) -> str:
    """Strip leading `---\\n...\\n---\\n` block if present."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].lstrip("\n")
    return text


def _extract_codex_last_message(stdout: str) -> str:
    """Codex `exec` prints reasoning / progress lines then the final message.

    The last non-empty line block (after a `codex` event marker) is the model's
    final response. Simplest robust heuristic: split on '[<timestamp>] codex'
    markers and take everything after the last one. Falls back to last
    non-empty paragraph.
    """
    text = stdout.strip()
    if not text:
        return ""
    # Codex 0.125 format: lines like "[2026-04-28T01:23:45] codex" then content
    parts = re.split(r"\n\[[\d\-T:.Z]+\]\s+codex\s*\n", text)
    if len(parts) > 1:
        last = parts[-1].strip()
        # Drop any trailing "tokens used: N" footer
        last = re.sub(r"\n\s*tokens used:.*$", "", last, flags=re.IGNORECASE | re.DOTALL).strip()
        if last:
            return last
    # Fallback: take everything after the last blank-line separator
    chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
    return chunks[-1] if chunks else text


def _estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


# -----------------------------------------------------------------------------
# Stub responses keyed off role (fast-path until OAuth is wired)
# -----------------------------------------------------------------------------


def _stub_response_for(role: str) -> dict:
    if role == "regime_reviewer":
        return {
            "review_label": "CONFIRM",
            "confidence_adjustment": 0.0,
            "risk_notes": ["stub_phase_2_4_no_llm"],
            "must_defer_new_entries": False,
        }
    if role == "scout":
        return {"candidates": [], "skipped": [{"ticker": "stub", "reason": "phase_2_4_stub"}]}
    if role in ("technical_analyst", "news_analyst", "sentiment_analyst", "fundamental_analyst"):
        return {"summary_md": "(stub: phase 2.4 not yet wired to LLM)"}
    if role in ("bull_researcher", "bear_researcher"):
        return {
            "round": 1,
            "thesis": "stub",
            "key_drivers": [],
            "addressed_bear_points": [],
            "remaining_uncertainty": ["stub_no_llm"],
            "bull_case_md": "(stub)",
            "bear_case_md": "(stub)",
        }
    if role == "trader_synthesizer":
        return {
            "decline_to_trade": True,
            "decline_reason": "stub_phase_2_4_no_llm",
            "reflection_notes": [],
            "proposal_notes": "(stub)",
        }
    if role in ("risk_conservative", "risk_opportunity"):
        return {
            "decision": "APPROVE",
            "approved_qty_factor": 1.0,
            "primary_risks": [],
            "concerns_md": "(stub)",
        }
    if role == "risk_arbiter":
        return {
            "decision": "APPROVE",
            "approved_qty": 0,
            "max_entry_price": 0.0,
            "primary_risks": [],
            "reason": "stub_phase_2_4_no_llm",
            "agreed_with": "neither",
            "required_conditions": [],
        }
    if role == "exit_monitor":
        return {"action": "HOLD", "exit_qty_factor": 0.0, "reason": "stub"}
    if role == "journal_agent":
        return {"mode": "FILL", "title": "stub", "body_md": "(stub)", "tags": []}
    if role == "ntfy_digest_composer":
        return {
            "title": "Daily digest (stub)",
            "body_md": "(LLM not yet wired in Phase 2.4)",
            "priority": 2,
            "tags": ["bar_chart"],
        }
    if role == "learning_critic":
        return {"n_proposed": 0, "proposals": [], "rejected_changes": [], "weekly_summary_md": "(stub)"}
    return {"text": "(stub)"}


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------


_ROUTER_SINGLETON: OAuthLLMRouter | StubLLMRouter | None = None


def get_router() -> OAuthLLMRouter | StubLLMRouter:
    """Return the singleton router. Falls back to StubLLMRouter when CLIs not logged in."""
    global _ROUTER_SINGLETON
    if _ROUTER_SINGLETON is not None:
        return _ROUTER_SINGLETON
    if claude_logged_in() or codex_logged_in():
        log.info("[oauth_router] OAuth detected → using real router")
        _ROUTER_SINGLETON = OAuthLLMRouter()
    else:
        log.info("[oauth_router] no OAuth credentials yet → using stub router")
        _ROUTER_SINGLETON = StubLLMRouter()
    return _ROUTER_SINGLETON
