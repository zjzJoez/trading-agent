"""Global PreToolUse hook — reject any tool input that looks like a real-trading attempt.

Belt-and-suspenders to the code-level TrdEnv.SIMULATE hard-coding in moomoo/server.py.
If someone edits the server, or an LLM tries to pass undocumented params, this hook
blocks the call before it reaches the MCP transport.

Runs on EVERY tool call (matcher: ".*" in .claude/settings.json). Must be cheap.

Blocks on exit code 2. Appends an audit line to data/hook_audit.log regardless of
outcome so we can review bypass attempts.

Contract (Claude Code hook protocol):
- stdin: JSON with tool_name, tool_input, etc.
- exit 0: allow; exit 2: block (stderr message shown to user+model).
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Patterns scanned in tool_input (recursively). Case-insensitive substring or
# key-name match. These are the red flags that should NEVER appear in a paper-only
# MVP — if they do, someone is trying something we said we wouldn't support.
FORBIDDEN_KEYS = {
    "trd_env", "trade_env", "trading_env",   # explicit env override
    "trading_password", "trd_password",      # real-account unlock
    "security_firm",                         # changing firm routing
}
FORBIDDEN_VALUES = {
    # Uppercase match; values are normalized to upper before comparing.
    "REAL", "LIVE", "PRODUCTION", "TRDENV.REAL",
}
# Tokens that appear as *substrings* of values — more permissive; catches
# things like "trd_env=REAL" hidden inside a larger string blob.
FORBIDDEN_SUBSTRINGS = [
    re.compile(r"\btrd_env\s*=\s*['\"]?(REAL|LIVE)\b", re.IGNORECASE),
    re.compile(r"\btrading_password\b", re.IGNORECASE),
    re.compile(r"TrdEnv\.REAL\b"),
]

AUDIT_PATH = Path.home() / "Desktop" / "trading-agent" / "data" / "hook_audit.log"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit(record: dict) -> None:
    try:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_PATH.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        # Never let audit failure block the hook; we'd rather allow a call than
        # silently drop it because of a disk issue.
        pass


def _scan(obj, path: str = "$") -> list[str]:
    """Walk tool_input and collect policy violations. Returns list of reason strings."""
    reasons: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in FORBIDDEN_KEYS:
                reasons.append(f"forbidden key {path}.{k}")
            reasons.extend(_scan(v, f"{path}.{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            reasons.extend(_scan(v, f"{path}[{i}]"))
    elif isinstance(obj, str):
        upper = obj.strip().upper()
        if upper in FORBIDDEN_VALUES:
            reasons.append(f"forbidden literal at {path}: {obj!r}")
        for pat in FORBIDDEN_SUBSTRINGS:
            if pat.search(obj):
                reasons.append(f"forbidden pattern at {path}: {pat.pattern}")
    return reasons


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as e:
        # Malformed payload: allow by default (hook infra problem, not a policy violation).
        sys.stderr.write(f"[reject_real_env] bad payload, allowing: {e}\n")
        return 0

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {})
    reasons = _scan(tool_input)

    audit = {
        "ts": _now(),
        "hook": "reject_real_env",
        "tool": tool_name,
        "blocked": bool(reasons),
        "reasons": reasons,
    }
    _audit(audit)

    if reasons:
        sys.stderr.write(
            "REFUSED by reject_real_env: tool input contains real-trading markers.\n"
            f"  tool: {tool_name}\n  reasons: {reasons}\n"
            "This project is paper-only. If you believe this is wrong, edit the allowlist\n"
            "in src/trading_agent/hooks/reject_real_env.py, not the caller.\n"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
