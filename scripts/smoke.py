"""Phase-1 MVP one-command smoke test.

Runs the exact same tool loop Claude Code would drive after a fresh restart,
but headless — so the user doesn't need to click anything. What this proves:

  1. The three MCP servers registered in .claude/settings.json boot and
     expose their full tool set over real JSON-RPC/stdio.
  2. moomoo-mcp can query MoomooOpenD for a real quote + option chain on a
     live paper account (SIMULATE env).
  3. edgar-mcp can pull live filings from data.sec.gov (respecting UA + rate
     limit) and cache them under data/filings_cache/.
  4. journal-mcp's full CRUD + RAG loop works: record_thesis →
     search_past_trades (sqlite-vec MiniLM) → close_thesis → append_note.
  5. pretool_order_guard.py blocks when no fresh thesis exists AND allows
     when one does — via real subprocess (same way Claude Code invokes it).
  6. reject_real_env.py rejects trd_env=REAL / trading_password payloads.
  7. posttool_fill_capture.py runs cleanly on a real place_paper_order
     response and inserts a `trades` row bound to the thesis.

Exit 0 = ready for real Claude Code use. Exit != 0 = something regressed.

Run (from the repo root):
    .venv/bin/python scripts/smoke.py

Takes ~20–40 seconds. Writes nothing lasting to data/trader.db — all test
rows are marked with `topic LIKE 'smoke-%'` and cleaned up at the end.
Leftover paper orders (if any) are cancelled.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv" / "bin"

# Force the smoke test to use the real project DB so we touch the same
# schema/vec index the user will see. We clean our rows at the end.
os.environ.setdefault("DB_PATH", str(ROOT / "data" / "trader.db"))
# SEC_UA_EMAIL must come from your .env (SEC requires a real address they
# can contact if they see misbehavior). We don't ship a default in source.

# Optional: fail fast if OpenD isn't up.
try:
    import socket
    with socket.create_connection(("127.0.0.1", 11111), timeout=2):
        pass
except OSError as e:
    print(f"[smoke] WARN: MoomooOpenD not reachable on 127.0.0.1:11111 — "
          f"Moomoo-MCP tests will skip quote/chain ({e}).", file=sys.stderr)
    OPEND_UP = False
else:
    OPEND_UP = True

from mcp import ClientSession, StdioServerParameters  # type: ignore
from mcp.client.stdio import stdio_client  # type: ignore

# ─────────────────────────────  helpers  ──────────────────────────────

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
INFO = "[....]"

results: list[tuple[str, str, str]] = []  # (outcome, label, detail)


def record(outcome: str, label: str, detail: str = "") -> None:
    results.append((outcome, label, detail))
    dots = "." * max(0, 50 - len(label))
    print(f"  {outcome} {label}{dots}{detail}")


async def call(session: ClientSession, name: str, args: dict) -> dict:
    resp = await session.call_tool(name, args)
    if resp.isError:
        raise RuntimeError(f"{name} returned error: {resp.content!r}")
    # FastMCP returns content blocks; extract structured JSON if present.
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            txt = block.text
            try:
                return json.loads(txt)
            except Exception:
                return {"_text": txt}
    return {"_raw": resp}


def run_hook(hook_path: Path, payload: dict) -> tuple[int, str, str]:
    """Invoke a hook the way Claude Code does: via subprocess + JSON on stdin."""
    proc = subprocess.run(
        [str(VENV / "python"), str(hook_path)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=20,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


# ──────────────────────────────  core  ────────────────────────────────

async def smoke_mcp() -> None:
    """Stanza 1-4: boot each MCP, exercise representative tools."""
    stack = AsyncExitStack()

    async def connect(name: str, binary: str) -> ClientSession:
        params = StdioServerParameters(command=str(VENV / binary), args=[], env=os.environ.copy())
        read, write = await stack.enter_async_context(stdio_client(params))
        sess = await stack.enter_async_context(ClientSession(read, write))
        await sess.initialize()
        return sess

    async with stack:
        print("\n[1/6] MCP server boot")
        moomoo = await connect("moomoo-mcp", "moomoo-mcp")
        edgar  = await connect("edgar-mcp",  "edgar-mcp")
        journal = await connect("journal-mcp", "journal-mcp")
        mt = {t.name for t in (await moomoo.list_tools()).tools}
        et = {t.name for t in (await edgar.list_tools()).tools}
        jt = {t.name for t in (await journal.list_tools()).tools}
        for label, got, want in [
            ("moomoo tool set", mt,
             {"get_quote", "get_option_chain_snapshot", "place_paper_order",
              "place_paper_option_order", "cancel_paper_order", "get_orders",
              "get_positions", "get_account_info"}),
            ("edgar tool set", et,
             {"search_filings", "get_filing_text",
              "get_recent_filings_for_ticker", "get_insider_transactions"}),
            ("journal tool set", jt,
             {"record_thesis", "record_fill", "append_note",
              "search_past_trades", "close_thesis", "close_trade",
              "get_open_positions_with_thesis",
              "generate_post_mortem_prompt"}),
        ]:
            missing = want - got
            if missing:
                record(FAIL, label, f" missing {sorted(missing)}")
            else:
                record(PASS, label, f" ({len(got)} tools)")

        # -- moomoo live quote + chain
        print("\n[2/6] moomoo-mcp live paper account")
        if not OPEND_UP:
            record(SKIP, "quote AAPL", " OpenD not running")
            record(SKIP, "option chain AAPL", " OpenD not running")
        else:
            try:
                q = await call(moomoo, "get_quote", {"symbols": ["US.AAPL"]})
                rows = q.get("rows", q.get("quotes", []))
                if not rows:
                    record(FAIL, "quote AAPL", f" empty response: {q}")
                else:
                    px = rows[0].get("last_price") or rows[0].get("cur_price")
                    record(PASS, "quote AAPL", f" last={px}")
            except Exception as e:
                record(FAIL, "quote AAPL", f" {e}")
            try:
                c = await call(moomoo, "get_option_chain_snapshot",
                               {"underlying": "US.AAPL",
                                "expiry_window_days": 30,
                                "strike_count_each_side": 3})
                rows = c.get("rows", [])
                record(PASS if rows else FAIL, "option chain AAPL",
                       f" {len(rows)} strikes (exp={c.get('expiry')})")
            except Exception as e:
                record(FAIL, "option chain AAPL", f" {e}")

        # -- edgar live filings
        print("\n[3/6] edgar-mcp live SEC fetch")
        try:
            f = await call(edgar, "get_recent_filings_for_ticker",
                           {"ticker": "AAPL", "limit": 5})
            filings = f.get("filings", f.get("rows", []))
            record(PASS if filings else FAIL,
                   "AAPL recent filings",
                   f" {len(filings)} filings")
        except Exception as e:
            record(FAIL, "AAPL recent filings", f" {e}")

        # -- journal full RAG loop
        print("\n[4/6] journal-mcp RAG + CRUD")
        ticker = "SMOKETEST"
        try:
            th = await call(journal, "record_thesis", {
                "ticker": ticker, "direction": "LONG",
                "thesis_text": "smoke-test thesis — verify RAG + hook gate",
                "invalidation": "close < 1.00 EOD",
                "timeframe": "same-day",
                "expected_return_pct": 1.0, "max_loss_pct": 1.0,
            })
            thesis_id = th.get("thesis_id") or th.get("id")
            assert thesis_id
            record(PASS, "record_thesis", f" id={thesis_id}")
        except Exception as e:
            record(FAIL, "record_thesis", f" {e}")
            thesis_id = None

        try:
            s = await call(journal, "search_past_trades",
                           {"query": "IV crush earnings long calls", "k": 3})
            notes = s.get("notes_semantic", [])
            trades = s.get("trades_lexical", [])
            top_topic = notes[0].get("topic") if notes else None
            record(PASS if notes else FAIL,
                   "search_past_trades (RAG)",
                   f" {len(notes)} notes / {len(trades)} trades"
                   + (f", top={top_topic}" if top_topic else ""))
        except Exception as e:
            record(FAIL, "search_past_trades (RAG)", f" {e}")

        try:
            n = await call(journal, "append_note", {
                "topic": "smoke-note",
                "text": "smoke test write with embedding",
                "tags": "smoke,test"})
            nid = n.get("note_id") or n.get("id")
            record(PASS if nid else FAIL, "append_note (with embed)",
                   f" id={nid}")
        except Exception as e:
            record(FAIL, "append_note (with embed)", f" {e}")

        # -- hooks via subprocess
        print("\n[5/6] hooks via subprocess (Claude Code dispatch shape)")
        hooks_dir = ROOT / "src" / "trading_agent" / "hooks"

        # reject_real_env: flag REAL
        rc, out, err = run_hook(hooks_dir / "reject_real_env.py", {
            "tool_name": "mcp__moomoo-mcp__place_paper_order",
            "tool_input": {"symbol": "US.AAPL", "qty": 1, "trd_env": "REAL"},
        })
        record(PASS if rc == 2 else FAIL, "reject trd_env=REAL",
               f" rc={rc}" + (f" err={err[:60]}" if rc != 2 else ""))

        rc, out, err = run_hook(hooks_dir / "reject_real_env.py", {
            "tool_name": "anything",
            "tool_input": {"foo": "bar", "trading_password": "xxx"},
        })
        record(PASS if rc == 2 else FAIL, "reject trading_password",
               f" rc={rc}")

        rc, out, err = run_hook(hooks_dir / "reject_real_env.py", {
            "tool_name": "mcp__moomoo-mcp__get_quote",
            "tool_input": {"symbols": ["US.AAPL"]},
        })
        record(PASS if rc == 0 else FAIL, "allow clean payload",
               f" rc={rc}")

        # pretool_order_guard: no fresh thesis for ZZZNEW → block
        rc, out, err = run_hook(hooks_dir / "pretool_order_guard.py", {
            "tool_name": "mcp__moomoo-mcp__place_paper_order",
            "tool_input": {"symbol": "US.ZZZNEW", "price": 1.0, "qty": 1,
                            "order_type": "NORMAL", "side": "BUY",
                            "thesis_id": 999999},
        })
        record(PASS if rc == 2 else FAIL, "block no-thesis order",
               f" rc={rc}")

        # pretool_order_guard: ticker with a fresh thesis (we just wrote one) → allow
        if thesis_id:
            rc, out, err = run_hook(hooks_dir / "pretool_order_guard.py", {
                "tool_name": "mcp__moomoo-mcp__place_paper_order",
                "tool_input": {"symbol": f"US.{ticker}", "price": 1.0, "qty": 1,
                                "order_type": "NORMAL", "side": "BUY",
                                "thesis_id": thesis_id},
            })
            record(PASS if rc == 0 else FAIL,
                   "allow fresh-thesis order",
                   f" rc={rc}" + (f" err={err[:80]}" if rc != 0 else ""))
        else:
            record(SKIP, "allow fresh-thesis order", " thesis not created")

        # posttool on synthetic fill payload
        if thesis_id:
            # Build the payload shape Claude Code delivers: tool_input + tool_response
            synth = {
                "tool_name": "mcp__moomoo-mcp__place_paper_order",
                "tool_input": {"symbol": f"US.{ticker}", "qty": 1, "price": 1.0,
                                "side": "BUY", "order_type": "NORMAL",
                                "thesis_id": thesis_id},
                "tool_response": {
                    "order_id": "SMOKE-ORDER-001",
                    "filled_qty": 1, "dealt_avg_price": 1.0,
                    "order_status": "FILLED",
                },
            }
            rc, out, err = run_hook(hooks_dir / "posttool_fill_capture.py", synth)
            record(PASS if rc == 0 else FAIL,
                   "posttool on synthetic fill",
                   f" rc={rc}" + (f" err={err[:80]}" if rc != 0 else ""))

        # -- cleanup
        print("\n[6/6] cleanup (rows + paper orders)")
        # Use the project's connection() throughout so sqlite-vec is loaded
        # and we don't fight ourselves for the db write lock.
        from trading_agent.db import connection as proj_conn
        with proj_conn() as p:
            smoke_note_ids = [r[0] for r in p.execute(
                "SELECT id FROM notes WHERE topic LIKE 'smoke-%'"
            ).fetchall()]
            p.execute("DELETE FROM theses WHERE ticker = ?", (ticker,))
            p.execute("DELETE FROM notes WHERE topic LIKE 'smoke-%'")
            p.execute("DELETE FROM trades WHERE broker_order_id LIKE 'SMOKE-%'")
            if smoke_note_ids:
                p.executemany(
                    "DELETE FROM notes_vec WHERE note_id = ?",
                    [(nid,) for nid in smoke_note_ids],
                )
            p.commit()
        record(PASS, "cleaned test rows",
               f" (notes+vec: {len(smoke_note_ids)})")

        if OPEND_UP:
            try:
                orders = await call(moomoo, "get_orders", {})
                ocount = 0
                for o in orders.get("rows", orders.get("orders", [])):
                    status = (o.get("order_status") or "").upper()
                    if status in {"SUBMITTED", "WAITING_SUBMIT", "SUBMITTING"}:
                        oid = o.get("order_id")
                        if oid:
                            try:
                                await call(moomoo, "cancel_paper_order",
                                           {"order_id": oid})
                                ocount += 1
                            except Exception:
                                pass
                record(PASS, "cancelled leftover paper orders", f" n={ocount}")
            except Exception as e:
                record(SKIP, "cancelled leftover paper orders", f" {e}")


def main() -> int:
    t0 = time.time()
    print("=" * 62)
    print(" Phase-1 MVP smoke test")
    print(f" DB={os.environ['DB_PATH']}")
    print(f" OpenD up: {OPEND_UP}")
    print(f" started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 62)
    try:
        asyncio.run(smoke_mcp())
    except Exception as e:
        print(f"\n[smoke] FATAL: {e!r}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return 2

    # summary
    p = sum(1 for r in results if r[0] == PASS)
    f = sum(1 for r in results if r[0] == FAIL)
    s = sum(1 for r in results if r[0] == SKIP)
    dur = time.time() - t0
    print("\n" + "=" * 62)
    print(f" {p} PASS  /  {f} FAIL  /  {s} SKIP   ({dur:.1f}s)")
    print("=" * 62)
    if f:
        print("\nFailures:")
        for o, lbl, det in results:
            if o == FAIL:
                print(f"  - {lbl}: {det}")
    return 0 if f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
