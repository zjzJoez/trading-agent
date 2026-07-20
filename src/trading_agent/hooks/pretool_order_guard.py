"""PreToolUse gate for paper orders — thesis freshness + sizing hard rules.

Runs on every invocation of moomoo-mcp's order-placing tools. Blocks
(exit 2) when:

  1. No open thesis for this ticker within the last 10 minutes
     (see order_guard.THESIS_FRESHNESS_MIN). Thesis table is source of truth.
  2. The order would OPEN a short option position (SELL-to-open hard block;
     SELL-to-close is exempt).
  3. An OPENING option order fails the R5d liquidity gate against a live
     snapshot: quoted spread <= 5% of mid AND open interest >= 500 at the
     strike; no obtainable bid/ask blocks as R5d_unquotable. Closes are
     exempt — the gate never traps an exit.
  4. Any sizing rule in trading_agent.sizing returns a "block"-severity
     violation (R1-R7; warns pass through).

All evaluation logic lives in ``trading_agent.order_guard`` — the SAME
engine the moomoo MCP server runs inside place_paper_order /
place_paper_option_order. This hook is the early, client-side layer (fast
feedback inside Claude Code); the server-side call is the authoritative
choke point that also covers the autonomous graph and scripts. Keeping
one engine means a rule fix lands in both layers at once (the R1
close-exemption, d089349, previously had to be hook-only).

Matcher is kept permissive so both the Claude Code hyphen-preserving
form (`mcp__moomoo-mcp__...`) and the normalize-to-underscore form
(`mcp__moomoo_mcp__...`) match.

Contract (Claude Code hook protocol):
  stdin  → JSON with tool_name + tool_input
  stdout → ignored on exit 0
  stderr → shown to user + model on exit 2
  exit 0 → allow
  exit 2 → block

Fail-closed design:
  - moomoo SDK unreachable for equity lookup → exit 2 ("cannot verify sizing")
  - DB corrupt / schema missing → exit 2
  - Unknown tool_name pattern → exit 0 (not our concern)
  - sectors.csv missing → R4 degrades to warn (still exit 0 if no blockers)
"""
from __future__ import annotations

import json
import re
import sys
import traceback

from trading_agent.config import CONFIG
from trading_agent.order_guard import (
    GuardDecision,
    audit_decision,
    evaluate_combo,
    evaluate_order,
)

GUARD_NAME = "pretool_order_guard"

ORDER_TOOL_RE = re.compile(
    r"^mcp__moomoo[_-]mcp__(place_paper_order|place_paper_option_order"
    r"|place_paper_option_combo)$"
)


def _fetch_equity_from_sdk() -> tuple[float, float | None]:
    """Query Moomoo paper account for (total_assets, cash). Fail-closed:
    raises on any error. Cash component is best-effort — returns None if
    the field isn't on the response.

    Cash is needed by R5b to validate cash-secured-put collateral.
    """
    # Lazy import to keep cold start fast when the tool isn't an order.
    from moomoo import OpenSecTradeContext, SecurityFirm, TrdEnv, TrdMarket

    ctx = OpenSecTradeContext(
        filter_trdmarket=TrdMarket.US,
        host=CONFIG.opend_host,
        port=CONFIG.opend_port,
        security_firm=SecurityFirm.FUTUSG,
    )
    try:
        ret, df = ctx.accinfo_query(trd_env=TrdEnv.SIMULATE)
    finally:
        try:
            ctx.close()
        except Exception:
            pass
    if ret != 0:  # moomoo.RET_OK == 0
        raise RuntimeError(f"accinfo_query failed: {df!r}")
    if df is None or df.empty:
        raise RuntimeError("accinfo_query returned no rows")
    # total_assets is USD market value + cash; use as equity denominator.
    val = df.iloc[0].get("total_assets")
    try:
        equity = float(val)
    except (TypeError, ValueError):
        raise RuntimeError(f"unparseable total_assets from accinfo_query: {val!r}")
    # Cash field naming varies by plan tier — try a couple of likely names.
    cash: float | None = None
    for key in ("cash", "available_funds", "available_cash"):
        v = df.iloc[0].get(key)
        if v is None:
            continue
        try:
            cash = float(v)
            break
        except (TypeError, ValueError):
            continue
    return equity, cash


def _emit_block(decision: GuardDecision, tool_name: str) -> int:
    sys.stderr.write(
        f"REFUSED by {GUARD_NAME}: {decision.reason}\n"
        f"  tool: {tool_name}\n"
    )
    for v in decision.violations_json():
        tag = "BLOCK" if v["severity"] == "block" else "warn "
        sys.stderr.write(f"  [{tag}] {v['rule']}: {v['message']}\n")
    sys.stderr.write(
        "\nFix and retry. See data/hook_audit.log and the hook_audit_log "
        "table in data/trader.db for full context. The moomoo server runs "
        "the same gate — bypassing this hook will not place the order.\n"
    )
    return 2


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as e:
        sys.stderr.write(f"[{GUARD_NAME}] bad payload: {e}\n")
        return 0  # infra problem, not a policy violation

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {}) or {}

    m = ORDER_TOOL_RE.match(tool_name)
    if not m:
        # Not our business.
        return 0
    tool = m.group(1)
    is_combo = tool == "place_paper_option_combo"
    kind = "option" if tool == "place_paper_option_order" else "stock"

    # Equity + cash from live SDK (paper account). Fail-closed on any SDK
    # issue. Cash is best-effort (None falls back to equity in R5b).
    try:
        equity, cash = _fetch_equity_from_sdk()
    except Exception as e:
        decision = GuardDecision(
            allowed=False,
            reason="cannot verify sizing — moomoo equity lookup failed",
            symbol=str(tool_input.get("option_symbol") or tool_input.get("symbol") or ""),
        )
        audit_decision(decision, guard_name=GUARD_NAME, tool_name=tool_name)
        sys.stderr.write(
            f"REFUSED by {GUARD_NAME}: {decision.reason}\n  detail: {e!r}\n"
        )
        return 2

    try:
        decision = (
            evaluate_combo(tool_input, equity=equity, cash=cash)
            if is_combo
            else evaluate_order(kind, tool_input, equity=equity, cash=cash)
        )
    finally:
        # R5d's quote fetch lazily imports the moomoo server and creates its
        # singleton OpenQuoteContext. That SDK context starts NON-DAEMON
        # threads, and without close() they keep this hook process alive
        # past sys.exit (the same hang server.shutdown()'s docstring cites
        # from a py-spy dump of the orchestrator). Only touch the module if
        # the fetch actually imported it — keep cold start fast otherwise.
        srv = sys.modules.get("trading_agent.mcp_servers.moomoo.server")
        if srv is not None:
            try:
                srv.shutdown()
            except Exception:
                pass
    audit_decision(decision, guard_name=GUARD_NAME, tool_name=tool_name)

    if not decision.allowed:
        return _emit_block(decision, tool_name)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        # Unhandled exception = fail-closed. The order must not silently go
        # through just because the hook crashed.
        sys.stderr.write(
            f"[pretool_order_guard] CRASH — failing closed:\n"
            f"{traceback.format_exc()}\n"
        )
        sys.exit(2)
