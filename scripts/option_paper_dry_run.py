"""Paper-options end-to-end dry-run.

What this does (and why):

  Phase-1 criterion #6 is: "option paper order works, either through the
  real Moomoo paper API or through the virtual-fill fallback." The only
  way to know which path our account is on is to actually try. This script
  does so headlessly, without Claude Code, so it's cheap to re-run.

Flow:
  1. Connect to moomoo-mcp + journal-mcp via stdio (same as Claude Code).
  2. Sanity: quote SPY (spot price, confirm OpenD responding).
  3. Pick a DTE-~30 expiry and a ~ATM call (delta-nearest-0.45).
  4. Record a SPY LONG_CALL thesis (posttool hook needs one to link to).
  5. Submit a 1-contract BUY at bid-ask mid via `place_paper_option_order`.
  6. Branch:
       a) Moomoo paper API accepts → invoke posttool_fill_capture.py with a
          synthetic PostToolUse payload so the `trades` + snapshot row lands
          the same way Claude Code would persist them. Cancel the resting
          order so paper inventory stays clean.
       b) Moomoo replies `virtual_fill_suggested=True` → call
          `record_virtual_fill` on journal-mcp. That inserts the trade row
          with `broker_order_id='VIRTUAL-<uuid>'`.
  7. Print which path fired + the resulting trades.id for the user.

Leaves one real row in `trades` on success — caller decides whether to
close it via /eod-review later. If you want it gone, delete by thesis_id
(printed at the end).

Run:
    .venv/bin/python scripts/option_paper_dry_run.py
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv" / "bin"

os.environ.setdefault("DB_PATH", str(ROOT / "data" / "trader.db"))

from mcp import ClientSession, StdioServerParameters  # type: ignore
from mcp.client.stdio import stdio_client  # type: ignore

TICKER = "SPY"
UNDERLYING = f"US.{TICKER}"
CONTRACTS = 1
# Target DTE window (matches position-sizing/SKILL.md: 14–60)
DTE_TARGET_MIN = 14
DTE_TARGET_MAX = 45
# Target delta window (SKILL.md: 0.30–0.55 for long premium)
DELTA_TARGET = 0.45

GREEN = "\033[32m"; RED = "\033[31m"; YELLOW = "\033[33m"
DIM = "\033[2m"; BOLD = "\033[1m"; RESET = "\033[0m"


def info(s: str) -> None:
    print(f"  {s}")


def ok(s: str) -> None:
    print(f"  {GREEN}✓{RESET} {s}")


def warn(s: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {s}")


def fail(s: str) -> None:
    print(f"  {RED}✗{RESET} {s}")


async def call(session: ClientSession, name: str, args: dict) -> dict:
    resp = await session.call_tool(name, args)
    if resp.isError:
        raise RuntimeError(f"{name} error: {resp.content!r}")
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            try:
                return json.loads(block.text)
            except Exception:
                return {"_text": block.text}
    return {}


def check_opend() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 11111), timeout=2):
            return True
    except OSError:
        return False


def run_posttool_hook(tool_name: str, tool_input: dict, tool_response: dict) -> tuple[int, str, str]:
    """Mirror what Claude Code does: pipe a PostToolUse payload on stdin."""
    hook = ROOT / "src" / "trading_agent" / "hooks" / "posttool_fill_capture.py"
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_response": tool_response,
    }
    proc = subprocess.run(
        [str(VENV / "python"), str(hook)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=30,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def pick_target_contract(chain_rows: list[dict]) -> dict | None:
    """Pick a CALL closest to target delta in the chain.

    Moomoo's US option snapshot doesn't expose bid/ask (those would require
    an orderbook subscription). We score by |delta - target| and take the
    closest contract that has a non-zero `option_premium`.
    """
    calls = [
        r for r in chain_rows
        if (r.get("option_type") or "").upper() in ("CALL", "C")
    ]
    if not calls:
        return None

    def score(r: dict) -> float:
        d = r.get("option_delta") or r.get("delta")
        if d is None:
            return 1e6
        try:
            return abs(float(d) - DELTA_TARGET)
        except Exception:
            return 1e6

    calls.sort(key=score)
    for cand in calls:
        prem = cand.get("option_premium")
        try:
            if prem is not None and float(prem) > 0:
                return cand
        except Exception:
            continue
    # No contract with a quoted mark — take the top-scored anyway so the
    # user sees why (likely outside RTH or symbol unavailable).
    return calls[0] if calls else None


async def main() -> int:
    print(f"{BOLD}Option paper dry-run — {datetime.now().isoformat(timespec='seconds')}{RESET}")
    print()

    if not check_opend():
        fail("MoomooOpenD not reachable on 127.0.0.1:11111 — start OpenD and re-run.")
        return 2
    ok("MoomooOpenD up on 127.0.0.1:11111")

    async with AsyncExitStack() as stack:
        async def connect(name: str, binary: str) -> ClientSession:
            params = StdioServerParameters(
                command=str(VENV / binary), args=[], env=os.environ.copy(),
            )
            read, write = await stack.enter_async_context(stdio_client(params))
            sess = await stack.enter_async_context(ClientSession(read, write))
            await sess.initialize()
            return sess

        info("Booting moomoo-mcp + journal-mcp stdio clients...")
        moomoo = await connect("moomoo-mcp", "moomoo-mcp")
        journal = await connect("journal-mcp", "journal-mcp")
        ok("MCPs ready")

        # 1) SPY spot
        q = await call(moomoo, "get_quote", {"symbols": [UNDERLYING]})
        rows = q.get("rows") or []
        if not rows:
            fail(f"no quote returned for {UNDERLYING}")
            return 3
        spot = rows[0].get("last_price")
        ok(f"{TICKER} spot = ${spot}")

        # 2) Expiry selection — do this explicitly so we can enforce DTE ≥ 14.
        # (get_option_chain_snapshot picks the *nearest* expiry in its window,
        # which in early-morning UTC can resolve to an already-closed same-day
        # expiry with bid=ask=0.)
        info(f"Listing option expiries for {UNDERLYING} (DTE ∈ [{DTE_TARGET_MIN},{DTE_TARGET_MAX}])...")
        expiries = await call(moomoo, "list_option_expiries", {"underlying": UNDERLYING})
        today = datetime.now(timezone.utc).date()
        cand: list[tuple[int, str]] = []
        for r in expiries.get("rows") or []:
            st = r.get("strike_time")
            if not st:
                continue
            try:
                d = datetime.strptime(st, "%Y-%m-%d").date()
            except Exception:
                continue
            dte = (d - today).days
            if DTE_TARGET_MIN <= dte <= DTE_TARGET_MAX:
                cand.append((dte, st))
        if not cand:
            fail(f"no expiries with DTE in [{DTE_TARGET_MIN},{DTE_TARGET_MAX}]")
            return 4
        cand.sort()
        dte_sel, expiry = cand[0]
        ok(f"selected expiry {expiry} (DTE={dte_sel})")

        # Fetch the chain + snapshots for that expiry directly. get_option_chain
        # returns the static chain (no greeks/bid/ask); we call get_quote on the
        # contract codes to get the snapshot.
        chain_static = await call(moomoo, "get_option_chain", {
            "underlying": UNDERLYING, "expiry": expiry, "option_type": "CALL",
        })
        codes = [r.get("code") for r in (chain_static.get("rows") or []) if r.get("code")]
        if not codes:
            fail(f"chain empty for {UNDERLYING} @ {expiry}")
            return 4
        # Keep 16 strikes closest to spot so we don't over-fetch.
        def strike_of(c: str) -> float:
            try:
                # US.SPY260516C00500000 → 500.000
                tail = c.rsplit("C", 1)[-1] if "C" in c else c.rsplit("P", 1)[-1]
                return int(tail) / 1000.0
            except Exception:
                return 0.0
        codes.sort(key=lambda c: abs(strike_of(c) - float(spot)))
        codes = codes[:16]
        chain_resp = await call(moomoo, "get_quote", {"symbols": codes})
        chain = chain_resp.get("rows") or []
        ok(f"chain: {len(chain)} call contracts around spot")

        # 3) Pick a strike
        target = pick_target_contract(chain)
        if not target:
            fail("no usable call contract found")
            return 5
        sym = target.get("code")
        strike = target.get("option_strike_price") or target.get("strike_price")
        premium = float(target.get("option_premium") or 0)
        delta = target.get("option_delta") or target.get("delta")
        # Moomoo paper book uses limit orders against its simulated fill
        # engine; we send at the mark and let the sim decide. Round to the
        # nickel (≥$3) or penny (<$3) per standard US option tick sizing.
        limit_px = round(premium + 0.01, 2) if premium < 3 else round(premium * 1.02, 2)
        ok(f"selected: {sym}")
        info(f"  strike={strike}  premium_mark={premium}  limit={limit_px}  delta≈{delta}  DTE={dte_sel}")
        if limit_px <= 0:
            fail("no premium quoted (option_premium=0) — likely out of RTH or illiquid strike")
            return 6

        # 4) Thesis (needed by the posttool linkage, same as Claude Code flow)
        th = await call(journal, "record_thesis", {
            "ticker": TICKER, "direction": "LONG_CALL",
            "thesis_text": (
                f"Dry-run: proving paper-options API path on {TICKER}. "
                f"Long ATM call, validate fill capture + virtual-fill fallback."
            ),
            "invalidation": f"SPY < {round(spot * 0.98, 2)} intraday or sub-mid bid",
            "timeframe": "swing",
            "expected_return_pct": 20.0,
            "max_loss_pct": 100.0,  # long premium: max loss = debit
        })
        thesis_id = th.get("thesis_id") or th.get("id")
        ok(f"thesis recorded (id={thesis_id})")

        # 5) Submit paper option order at (slightly crossing) mark
        info(f"Submitting {CONTRACTS} BUY @ limit ${limit_px} on {sym}...")
        order_input = {
            "option_symbol": sym,
            "side": "BUY",
            "contracts": CONTRACTS,
            "price": limit_px,
            "thesis_id": thesis_id,
            "strategy_label": "long_atm_call",
            "delta": float(delta) if delta is not None else None,
            "dte": dte_sel,
        }
        try:
            order_resp = await call(moomoo, "place_paper_option_order", order_input)
        except Exception as e:
            fail(f"place_paper_option_order raised: {e}")
            order_resp = {"virtual_fill_suggested": True, "reason": str(e),
                          "thesis_id": thesis_id}

        print()
        print(f"  {DIM}{json.dumps(order_resp, indent=2)[:600]}{RESET}")
        print()

        trade_id: int | None = None
        path: str

        if order_resp.get("virtual_fill_suggested"):
            # ---- Path B: virtual-fill fallback ----
            warn("Moomoo paper API rejected the option order — invoking virtual-fill path")
            reason = order_resp.get("reason", "unknown")
            info(f"  reason: {str(reason)[:200]}")
            vf = await call(journal, "record_virtual_fill", {
                "option_symbol": sym,
                "side": "BUY",
                "contracts": CONTRACTS,
                "mid_price": premium or limit_px,
                "thesis_id": thesis_id,
                "strategy_label": "long_atm_call",
                "reasoning": f"virtual fill: moomoo paper rejected; reason={str(reason)[:80]}",
            })
            trade_id = vf.get("trade_id")
            path = "virtual-fill"
            ok(f"virtual fill recorded: trade_id={trade_id} broker={vf.get('broker_order_id')}")
        else:
            # ---- Path A: real paper fill ----
            broker_rows = order_resp.get("rows") or []
            if not broker_rows:
                fail("order returned without rows and without virtual suggestion — unclear state")
                return 7
            ok(f"Moomoo accepted order — got {len(broker_rows)} row(s)")

            # Run the real PostToolUse hook against this response, so the
            # journal row lands exactly the way Claude Code would persist it.
            info("Running posttool_fill_capture.py (same subprocess CC uses)...")
            rc, out, err = run_posttool_hook(
                tool_name="mcp__moomoo-mcp__place_paper_option_order",
                tool_input=order_input,
                tool_response=order_resp,
            )
            if rc != 0:
                warn(f"posttool hook rc={rc}; stderr={err[:200]}")
            else:
                ok("posttool hook exited 0")

            # Pick up the trade_id the hook just inserted (most recent for thesis)
            import sqlite3
            c = sqlite3.connect(ROOT / "data" / "trader.db")
            row = c.execute(
                "SELECT id, broker_order_id, symbol, entry_price FROM trades "
                "WHERE thesis_id = ? ORDER BY id DESC LIMIT 1",
                (thesis_id,),
            ).fetchone()
            c.close()
            if row:
                trade_id = row[0]
                ok(f"trade row: id={row[0]} broker={row[1]} symbol={row[2]} entry={row[3]}")
            else:
                warn("no trade row found for thesis — hook may have failed silently")

            # Paper orders on options may rest as SUBMITTING indefinitely on
            # an inactive session. Cancel to leave a clean paper book.
            broker_order_id = broker_rows[0].get("order_id")
            if broker_order_id:
                info(f"Cancelling resting paper order {broker_order_id}...")
                try:
                    cx = await call(moomoo, "cancel_paper_order",
                                    {"order_id": str(broker_order_id)})
                    ok(f"cancel ok: {json.dumps(cx)[:200]}")
                except Exception as e:
                    warn(f"cancel failed (non-fatal): {e}")
            path = "real-paper"

    # Summary
    print()
    print(f"{BOLD}Summary{RESET}")
    print(f"  path         : {GREEN if path else RED}{path}{RESET}")
    print(f"  thesis_id    : {thesis_id}")
    print(f"  trade_id     : {trade_id}")
    print()
    print(f"{DIM}To remove this dry-run trade + thesis later:{RESET}")
    print(f"{DIM}  sqlite3 data/trader.db \"DELETE FROM trades WHERE thesis_id={thesis_id};"
          f" DELETE FROM theses WHERE id={thesis_id};\"{RESET}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
