"""PostToolUse capture for paper order fills.

Fires after moomoo-mcp order tools return. Persists a `trades` row (one
per response row) and a `market_snapshots` payload, then exits 0.

Why posttool rather than inside the MCP tool itself:
  - Keeps moomoo/server.py a thin transport wrapper. Anything DB-ish
    lives in trading_agent.db / trading_agent.mcp_servers.journal.
  - The fill/thesis/snapshot bundle is exactly what the post-mortem
    flow consumes; having the hook own the join keeps the schema-level
    contract in one place.

This hook is NOT a gate. Any failure here is a telemetry problem, not
a trading problem — exit 0 in all cases and log to data/hook_audit.log.

Submit-time reality: MoomooOpenD typically returns order_status
SUBMITTING with dealt_qty=0 on the first response. Per plan decision
(MVP pick (a)): record at submit-time with entry_price=limit_price so
the journal row is never missing. If a subsequent FILLED_ALL event
ever comes in with a different dealt_avg_price, the ledger drift is
accepted for MVP (documented in data/hook_audit.log).
"""
from __future__ import annotations

import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_agent.config import CONFIG, ensure_dirs
from trading_agent.db import connection

ORDER_TOOL_RE = re.compile(
    r"^mcp__moomoo[_-]mcp__(place_paper_order|place_paper_option_order"
    r"|place_paper_option_combo)$"
)
OCC_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{4,8})$")
AUDIT_PATH = CONFIG.data_dir / "hook_audit.log"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit(rec: dict) -> None:
    try:
        ensure_dirs()
        with AUDIT_PATH.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


def _extract_dict(tool_response: Any) -> dict | None:
    """MCP tool responses can arrive as the bare return dict OR wrapped in
    `{"content": [{"type":"text","text":"<json>"}], "isError": False}`.
    Handle both shapes; give up gracefully on anything else."""
    if isinstance(tool_response, dict) and "content" not in tool_response:
        return tool_response
    if isinstance(tool_response, dict) and "content" in tool_response:
        items = tool_response.get("content") or []
        for item in items:
            if isinstance(item, dict) and item.get("type") == "text":
                try:
                    return json.loads(item.get("text") or "")
                except Exception:
                    continue
    return None


def _side_from_trd_side(x: Any) -> str | None:
    """moomoo SDK returns trd_side as 'BUY'/'SELL' or the int enum. Normalize."""
    if x is None:
        return None
    s = str(x).upper()
    if "BUY" in s:
        return "BUY"
    if "SELL" in s:
        return "SELL"
    return None


def _ticker_from_symbol(symbol: str) -> str | None:
    if not symbol:
        return None
    s = symbol.upper()
    if s.startswith("US."):
        s = s[3:]
    m = OCC_RE.match(s)
    if m:
        return m.group(1)
    return s


def _insert_trade_row(
    *,
    thesis_id: int | None,
    symbol: str,
    asset_type: str,
    strategy_label: str | None,
    side: str | None,
    qty: float,
    entry_price: float,
    stop: float | None,
    target: float | None,
    broker_order_id: str,
    reasoning: str | None,
) -> int:
    opened_at = _now()
    with connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO trades (thesis_id, symbol, asset_type, strategy_label,
                side, qty, entry_price, stop, target, opened_at, reasoning,
                outcome, broker_order_id, is_paper)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, 1)
            """,
            (thesis_id, symbol, asset_type, strategy_label, side, qty,
             entry_price, stop, target, opened_at, reasoning, broker_order_id),
        )
        trade_id = cur.lastrowid
        conn.commit()
    return int(trade_id)


def _insert_snapshot(trade_id: int, payload: dict) -> None:
    with connection() as conn:
        conn.execute(
            "INSERT INTO market_snapshots (trade_id, taken_at, payload) "
            "VALUES (?, ?, ?)",
            (trade_id, _now(), json.dumps(payload, default=str)),
        )
        conn.commit()


def _insert_combo_pg_row(
    *,
    short_sym: str,
    long_sym: str,
    contracts: float,
    net_credit: float,
    width: float,
    strategy_label: str | None,
    order_ids: list[str],
    snapshot_payload: dict,
) -> None:
    """Dual-write one engine-visible combo row into Postgres journal_trades.

    The row mirrors the SQLite combo row (symbol=short leg, side=SELL,
    entry_price=net_credit) and additionally carries:
      * ``exit_plan`` = combo_exit_plan(net_credit, width) — 50%-PT + 21-DTE
        priced on the whole spread (nothing else can attach a plan to a
        combo; without it the triggers can never fire — CONFLICTS C4);
      * ``broker_fill_json.combo`` = the canonical payload (the ONE Postgres
        detection rule, same shape as the SQLite snapshot);
      * 2-leg entry fees (a vertical pays commission on BOTH legs — C6).

    NOTE: order_ids[0] is the LONG leg's order id (place_paper_option_combo
    places the protective long first) — broker_order_id is a provenance
    pointer only, never "the" order; both ids live in combo.broker_order_ids.
    sync_fill_status's combo branch resolves legs from the combo payload,
    never from this column.

    Raises on any failure — the caller audits and swallows (telemetry hook).
    """
    from trading_agent.execution_costs import fees_per_side
    from trading_agent.llm.schemas import combo_exit_plan
    from trading_agent.store.postgres import cursor

    plan = combo_exit_plan(net_credit, width)
    fill_json = {
        "combo": snapshot_payload,
        "strategy_label": strategy_label,
        "provenance": "agent",
        "entry_fees": fees_per_side(contracts, "OPT") * 2,
    }
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO journal_trades
                (symbol, asset_type, side, qty, entry_price, stop, target,
                 outcome, broker_order_id, is_paper, opened_at,
                 exit_plan, broker_fill_json)
            VALUES (%s, 'OPT', 'SELL', %s, %s, NULL, NULL,
                    'OPEN', %s, TRUE, NOW(), %s::jsonb, %s::jsonb)
            """,
            (short_sym, contracts, net_credit,
             order_ids[0] if order_ids else "",
             json.dumps(plan.model_dump(), default=str),
             json.dumps(fill_json, default=str)),
        )


def _best_effort_snapshot(symbol: str) -> dict:
    """Try to grab a quote (and for options a short-window chain context).
    Best-effort — any failure returns a payload marked 'unavailable'. The
    hook swallows exceptions so the trade row always lands.
    """
    try:
        from moomoo import OpenQuoteContext
        ctx = OpenQuoteContext(host=CONFIG.opend_host, port=CONFIG.opend_port)
        try:
            ret, df = ctx.get_market_snapshot([symbol])
        finally:
            try:
                ctx.close()
            except Exception:
                pass
        if ret != 0 or df is None or df.empty:
            return {"symbol": symbol, "quote": None, "note": f"snapshot ret={ret}"}
        row = df.iloc[0].to_dict()
        # dataframe values may include numpy types; JSON-encode defensively upstream.
        return {"symbol": symbol, "quote": row}
    except Exception as e:
        return {"symbol": symbol, "quote": None, "note": f"snapshot failed: {e!r}"}


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as e:
        sys.stderr.write(f"[posttool_fill_capture] bad payload: {e}\n")
        return 0

    tool_name = payload.get("tool_name", "")
    if not ORDER_TOOL_RE.match(tool_name):
        return 0

    tool_input = payload.get("tool_input", {}) or {}
    tool_response_raw = payload.get("tool_response", {}) or {}
    resp = _extract_dict(tool_response_raw)
    if resp is None:
        _audit({"ts": _now(), "hook": "posttool_fill_capture", "tool": tool_name,
                "decision": "noop", "reason": "unparseable tool_response",
                "response_shape": type(tool_response_raw).__name__})
        return 0

    # Virtual-fill suggestion path: option tool returned without placing.
    # Skill will invoke journal.record_virtual_fill next — we don't double-write.
    if resp.get("virtual_fill_suggested"):
        _audit({"ts": _now(), "hook": "posttool_fill_capture", "tool": tool_name,
                "decision": "defer_virtual", "reason": resp.get("reason"),
                "thesis_id": resp.get("thesis_id")})
        return 0

    # Combo (defined-risk vertical): journal the spread as ONE OPT position so
    # R2 counts it once and the post-mortem scores it as a unit. entry_price is
    # the net credit; both leg symbols + broker order ids + width/max_loss live
    # in the snapshot. NOTE (area A limitation): at the broker-position level
    # the short leg still reads as a naked short to portfolio heat — that
    # over-counts (conservative; blocks more, never less). Netting vertical
    # pairs in heat, and a first-class combo-CLOSE path, are flagged follow-ups.
    if resp.get("combo"):
        combo_rows = resp.get("combo_rows") or []
        order_ids = [str(r.get("order_id") or "") for r in combo_rows]
        short_sym = tool_input.get("short_leg_symbol", "")
        long_sym = tool_input.get("long_leg_symbol", "")
        net_credit = resp.get("net_credit")
        width = resp.get("width")
        max_loss = resp.get("max_loss")
        combo_thesis = resp.get("thesis_id") or tool_input.get("thesis_id")
        combo_label = (resp.get("strategy_label")
                       or tool_input.get("strategy_label")
                       or "credit_put_spread_30_45")
        contracts = float(tool_input.get("contracts", 0) or 0)
        reasoning = (f"combo defined-risk vertical: short={short_sym} "
                     f"long={long_sym} net_credit={net_credit} width={width} "
                     f"max_loss={max_loss}")
        try:
            trade_id = _insert_trade_row(
                thesis_id=combo_thesis, symbol=short_sym, asset_type="OPT",
                strategy_label=combo_label, side="SELL", qty=contracts,
                entry_price=float(net_credit or 0), stop=None, target=None,
                broker_order_id=order_ids[0] if order_ids else "",
                reasoning=reasoning,
            )
            _insert_snapshot(trade_id, {
                "combo": True, "short_leg": short_sym, "long_leg": long_sym,
                "broker_order_ids": order_ids, "net_credit": net_credit,
                "width": width, "max_loss": max_loss, "contracts": contracts,
            })
        except Exception as e:
            _audit({"ts": _now(), "hook": "posttool_fill_capture", "tool": tool_name,
                    "decision": "error", "reason": f"combo insert failed: {e!r}"})
            return 0
        # D2 dual-write: the exit engine enriches/resolves EXCLUSIVELY from
        # Postgres get_open_journal_trades — a combo journaled only in SQLite
        # never reaches detect_exit_triggers at all (both legs would surface
        # as unmanaged "manual" broker positions). Best-effort: Postgres
        # unreachable (the Phase-1 Mac) → audit line, exit 0. The hook is
        # telemetry, never a gate.
        try:
            _insert_combo_pg_row(
                short_sym=short_sym, long_sym=long_sym,
                contracts=contracts, net_credit=float(net_credit or 0),
                width=float(width or 0), strategy_label=combo_label,
                order_ids=order_ids,
                snapshot_payload={
                    "combo": True, "short_leg": short_sym, "long_leg": long_sym,
                    "broker_order_ids": order_ids, "net_credit": net_credit,
                    "width": width, "max_loss": max_loss,
                    "contracts": contracts,
                },
            )
            pg_written = True
        except Exception as e:
            pg_written = False
            _audit({"ts": _now(), "hook": "posttool_fill_capture",
                    "tool": tool_name, "decision": "warn",
                    "phase": "combo_pg_dual_write", "error": repr(e),
                    "note": "combo row exists in SQLite only — the exit "
                            "engine (Postgres-fed) will NOT manage it"})
        _audit({"ts": _now(), "hook": "posttool_fill_capture", "tool": tool_name,
                "decision": "combo_insert", "trade_id": trade_id,
                "short_leg": short_sym, "long_leg": long_sym,
                "broker_order_ids": order_ids, "pg_dual_write": pg_written})
        return 0

    is_option = tool_name.endswith("__place_paper_option_order")
    asset_type = "OPT" if is_option else "STK"

    thesis_id = resp.get("thesis_id") or tool_input.get("thesis_id")
    strategy_label = resp.get("strategy_label") or tool_input.get("strategy_label")
    stop = resp.get("stop") if not is_option else None
    target = resp.get("target") if not is_option else None
    # For options: tool_input carries delta + dte (echoed into resp too).
    reasoning_bits: list[str] = []
    if is_option:
        if resp.get("delta") is not None:
            reasoning_bits.append(f"delta={resp.get('delta')}")
        if resp.get("dte") is not None:
            reasoning_bits.append(f"dte={resp.get('dte')}")
    reasoning = ", ".join(reasoning_bits) or None

    rows = resp.get("rows") or []
    if not rows:
        _audit({"ts": _now(), "hook": "posttool_fill_capture", "tool": tool_name,
                "decision": "noop", "reason": "no rows in response",
                "resp_keys": list(resp.keys())})
        return 0

    inserted: list[dict] = []
    for row in rows:
        symbol = (
            row.get("code")
            or tool_input.get("symbol")
            or tool_input.get("option_symbol")
            or ""
        )
        side = _side_from_trd_side(row.get("trd_side") or tool_input.get("side"))

        # Submit-time: dealt_qty often 0, order_status SUBMITTING. Per MVP
        # decision, record at submit with entry=limit_price so the journal
        # row never lags the broker. If dealt_avg_price is non-zero, prefer
        # it (partial/immediate fills).
        dealt_qty = float(row.get("dealt_qty", 0) or 0)
        dealt_avg_price = float(row.get("dealt_avg_price", 0) or 0)
        limit_price = float(tool_input.get("price", 0) or 0)
        entry_price = dealt_avg_price if dealt_qty > 0 and dealt_avg_price > 0 else limit_price

        qty = float(
            row.get("qty")
            or tool_input.get("qty")
            or tool_input.get("contracts")
            or 0
        )
        broker_order_id = str(row.get("order_id") or "")

        try:
            trade_id = _insert_trade_row(
                thesis_id=(int(thesis_id) if thesis_id is not None else None),
                symbol=symbol,
                asset_type=asset_type,
                strategy_label=strategy_label,
                side=side,
                qty=qty,
                entry_price=entry_price,
                stop=(float(stop) if stop is not None else None),
                target=(float(target) if target is not None else None),
                broker_order_id=broker_order_id,
                reasoning=reasoning,
            )
        except Exception as e:
            _audit({"ts": _now(), "hook": "posttool_fill_capture", "tool": tool_name,
                    "decision": "error", "phase": "insert_trade",
                    "error": repr(e), "row": row})
            continue

        # Best-effort snapshot — never raises.
        snap_payload = {
            "raw_order_row": row,
            "quote": _best_effort_snapshot(symbol),
            "order_status_at_submit": row.get("order_status"),
        }
        try:
            _insert_snapshot(trade_id, snap_payload)
        except Exception as e:
            _audit({"ts": _now(), "hook": "posttool_fill_capture", "tool": tool_name,
                    "decision": "warn", "phase": "insert_snapshot",
                    "error": repr(e), "trade_id": trade_id})

        inserted.append({"trade_id": trade_id, "broker_order_id": broker_order_id,
                          "symbol": symbol, "qty": qty, "entry_price": entry_price})

    _audit({"ts": _now(), "hook": "posttool_fill_capture", "tool": tool_name,
            "decision": "captured", "thesis_id": thesis_id,
            "inserted": inserted, "row_count": len(rows)})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        sys.stderr.write(
            f"[posttool_fill_capture] crashed (non-blocking):\n"
            f"{traceback.format_exc()}\n"
        )
        # Posttool hooks must never block; exit 0 even on crash.
        sys.exit(0)
