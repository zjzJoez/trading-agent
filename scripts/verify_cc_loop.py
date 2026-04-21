"""Automated restart-free validation of the full Claude Code tool loop.

What this proves (without you touching anything):
  1. A fresh `claude -p` subprocess spawns the three project MCP servers
     from an explicit --mcp-config.
  2. The model successfully discovers + calls an MCP tool
     (mcp__moomoo-mcp__get_quote → live paper account).
  3. The PreToolUse hook `pretool_order_guard.py` fires on real CC
     dispatch and blocks a no-thesis order (exit 2 surfaces as
     permission_denials[]).

Runs two headless CC sessions back-to-back, ~2 minutes total.

Run (from the repo root):
    .venv/bin/python scripts/verify_cc_loop.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_BIN = ROOT / ".venv" / "bin"

MCP_CONFIG = {
    "mcpServers": {
        "moomoo-mcp":  {"command": str(VENV_BIN / "moomoo-mcp"),  "args": [], "env": {}},
        "edgar-mcp":   {"command": str(VENV_BIN / "edgar-mcp"),   "args": [], "env": {}},
        "journal-mcp": {"command": str(VENV_BIN / "journal-mcp"), "args": [], "env": {}},
    }
}

PASS = "[PASS]"
FAIL = "[FAIL]"
results: list[tuple[str, str, str]] = []


def record(outcome: str, label: str, detail: str = "") -> None:
    results.append((outcome, label, detail))
    dots = "." * max(0, 56 - len(label))
    print(f"  {outcome} {label}{dots}{detail}")


def run_headless(prompt: str, timeout_s: int = 180) -> dict:
    """Launches `claude -p ...` with our MCP config, returns parsed JSON result."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(MCP_CONFIG, f)
        cfg = f.name
    try:
        proc = subprocess.run(
            ["claude",
             "--mcp-config", cfg,
             "--dangerously-skip-permissions",
             "--output-format", "json",
             "-p", prompt],
            capture_output=True, text=True, timeout=timeout_s,
            cwd=str(ROOT),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr[:500]}")
        return json.loads(proc.stdout)
    finally:
        os.unlink(cfg)


def test_happy_path() -> None:
    """A fresh CC should be able to call get_quote via MCP + hooks pass."""
    print("\n[1/2] Happy path — mcp__moomoo-mcp__get_quote on US.AAPL")
    t0 = time.time()
    res = run_headless(
        "Your task: fetch the current quote for US.AAPL using the "
        "moomoo-mcp server and output ONLY its last_price as a number.\n\n"
        "Procedure:\n"
        "1. Use ToolSearch with query 'select:mcp__moomoo-mcp__get_quote' "
        "to load the tool schema.\n"
        "2. If the search reports the server is still connecting, wait 3 "
        "seconds (by doing a trivial Bash `sleep 3`) and ToolSearch again. "
        "Retry up to 5 times.\n"
        "3. Once the tool loads, call mcp__moomoo-mcp__get_quote with "
        "symbols=['US.AAPL'].\n"
        "4. Extract last_price from the first row and output ONLY that "
        "number (e.g. 267.25). Nothing else.\n"
        "5. Only if after 5 retries the server is still not connected, "
        "output 'TOOL_UNAVAILABLE'.",
        timeout_s=240,
    )
    dur = time.time() - t0
    ans = (res.get("result") or "").strip()
    denials = res.get("permission_denials", [])
    try:
        price = float(ans)
        record(PASS, "get_quote US.AAPL via fresh CC",
               f" last=${price:.2f} ({dur:.0f}s)")
    except ValueError:
        record(FAIL, "get_quote US.AAPL via fresh CC",
               f" got: {ans!r} denials={denials}")


def test_hook_block() -> None:
    """A fresh CC should have the PreToolUse hook block a no-thesis order."""
    print("\n[2/2] Hook block — place_paper_order without a thesis")
    t0 = time.time()
    res = run_headless(
        "Call mcp__moomoo-mcp__place_paper_order once with symbol='US.ZZZTEST', "
        "side='BUY', qty=1, price=1.0, thesis_id=99999. There is no open "
        "thesis for ZZZTEST, so the PreToolUse hook pretool_order_guard.py "
        "should block this call (it exits with code 2 and returns a "
        "permission denial). Do NOT retry or try anything else. After the "
        "call is blocked, output exactly 'BLOCKED' if the call was blocked "
        "by the hook, or 'NOT_BLOCKED' if the order was actually placed.",
        timeout_s=180,
    )
    dur = time.time() - t0
    ans = (res.get("result") or "").strip().upper()
    denials = res.get("permission_denials", [])
    blocked_by_hook = any(
        d.get("tool_name", "").endswith("place_paper_order")
        and (d.get("tool_input") or {}).get("symbol") == "US.ZZZTEST"
        for d in denials
    )
    if "BLOCKED" in ans and blocked_by_hook:
        record(PASS, "no-thesis order blocked by pretool hook",
               f" ({dur:.0f}s, denials={len(denials)})")
    else:
        record(FAIL, "no-thesis order blocked by pretool hook",
               f" ans={ans!r} blocked_by_hook={blocked_by_hook} denials={denials}")


def main() -> int:
    print("=" * 64)
    print(" Claude Code tool-loop verification (restart-free)")
    print(f" cwd: {ROOT}")
    print("=" * 64)

    try:
        test_happy_path()
        test_hook_block()
    except subprocess.TimeoutExpired as e:
        print(f"\n[verify] headless CC timeout: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"\n[verify] FATAL: {e!r}", file=sys.stderr)
        return 2

    p = sum(1 for r in results if r[0] == PASS)
    f = sum(1 for r in results if r[0] == FAIL)
    print("\n" + "=" * 64)
    print(f" {p} PASS / {f} FAIL")
    print("=" * 64)
    if f:
        print("\nFailures:")
        for o, lbl, det in results:
            if o == FAIL:
                print(f"  - {lbl}: {det}")
    return 0 if f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
