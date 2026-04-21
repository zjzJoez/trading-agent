"""PreToolUse gate for paper orders — thesis freshness + sizing hard rules.

Runs on every invocation of moomoo-mcp's order-placing tools. Blocks
(exit 2) when:

  1. No open thesis for this ticker within the last 10 minutes
     (see THESIS_FRESHNESS_MIN). Thesis table is the source of truth.
  2. Any sizing rule in trading_agent.sizing returns a "block"-severity
     violation (R1-R6; warns pass through).

Matcher is kept permissive so both the Claude Code hyphen-preserving
form (`mcp__moomoo-mcp__...`) and the normalize-to-underscore form
(`mcp__moomoo_mcp__...`) match. The first invocation in production
will write the exact tool_name to the audit log; tighten later if
desired.

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

import csv
import json
import re
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# These imports hit trading_agent.* only — deliberately NOT mcp_servers.*,
# which would drag in FastMCP + sentence-transformers.
from trading_agent.config import CONFIG, ensure_dirs
from trading_agent.db import connection
from trading_agent.sizing import (
    OpenPosition,
    ProposedTrade,
    SizingContext,
    blockers,
    check,
    summarize,
)

# ---- config constants ----

ORDER_TOOL_RE = re.compile(
    r"^mcp__moomoo[_-]mcp__(place_paper_order|place_paper_option_order)$"
)
OCC_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{4,8})$")
THESIS_FRESHNESS_MIN = 10

AUDIT_PATH = CONFIG.data_dir / "hook_audit.log"
SECTORS_CSV = CONFIG.data_dir / "sectors.csv"


# ---- small utils ----

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit(rec: dict) -> None:
    try:
        ensure_dirs()
        with AUDIT_PATH.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        # Audit must never block the hook from completing.
        pass


def _block(reason: str, tool_name: str, detail: Any = None) -> int:
    _audit({
        "ts": _now(),
        "hook": "pretool_order_guard",
        "tool": tool_name,
        "decision": "block",
        "reason": reason,
        "detail": detail,
    })
    sys.stderr.write(
        f"REFUSED by pretool_order_guard: {reason}\n"
        f"  tool: {tool_name}\n"
    )
    if detail:
        sys.stderr.write(f"  detail: {detail}\n")
    sys.stderr.write(
        "\nFix and retry. See data/hook_audit.log for full context. "
        "To override (never in production), edit "
        "src/trading_agent/hooks/pretool_order_guard.py.\n"
    )
    return 2


def _allow(tool_name: str, note: str = "", warns: list[str] | None = None) -> int:
    _audit({
        "ts": _now(),
        "hook": "pretool_order_guard",
        "tool": tool_name,
        "decision": "allow",
        "note": note,
        "warns": warns or [],
    })
    return 0


# ---- symbol parsing ----

def _bare_ticker(symbol: str) -> str | None:
    """US.AAPL → AAPL. Returns None if unparseable."""
    if not symbol:
        return None
    s = symbol.strip().upper()
    if s.startswith("US."):
        s = s[3:]
    if not s:
        return None
    # Reject obvious option codes handed to a stock tool.
    if OCC_RE.match(s):
        return None
    return s


def _parse_option_symbol(option_symbol: str) -> tuple[str, int] | None:
    """Moomoo option code → ("AAPL", dte_from_today). Returns None on bad input.

    Moomoo format: US.<TICKER><YYMMDD><C|P><strike*1000> where the strike
    is unpadded (e.g., US.AAPL260515C267500 = $267.500 strike). The regex
    accepts 4-8 strike digits to handle common US equity option strikes
    (penny stocks → mega-caps).

    Ticker-with-dot names (BRK.B, BF.B) aren't in moomoo's option universe
    the same way; MVP doesn't trade those, and the regex here won't match
    them. Flagged as a known limitation.
    """
    s = option_symbol.strip().upper()
    if s.startswith("US."):
        s = s[3:]
    m = OCC_RE.match(s)
    if not m:
        return None
    ticker = m.group(1)
    yymmdd = m.group(2)
    try:
        expiry = datetime.strptime(yymmdd, "%y%m%d").date()
    except ValueError:
        return None
    today = datetime.now(timezone.utc).date()
    dte = (expiry - today).days
    return ticker, dte


# ---- data sources ----

def _load_sector_map() -> dict[str, str]:
    if not SECTORS_CSV.exists():
        return {}
    out: dict[str, str] = {}
    with SECTORS_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("ticker") or "").strip().upper()
            sec = (row.get("sector") or "").strip()
            if t and sec:
                out[t] = sec
    return out


def _fetch_equity_from_sdk() -> float:
    """Query Moomoo paper account for total_assets. Fail-closed: raise on any
    error. Called inside try/except at the call site — the hook upgrades
    the error into an exit-2 block."""
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
    # Some plans return only `total_assets` as string — coerce defensively.
    val = df.iloc[0].get("total_assets")
    try:
        return float(val)
    except (TypeError, ValueError):
        raise RuntimeError(f"unparseable total_assets from accinfo_query: {val!r}")


def _load_open_positions(sector_map: dict[str, str]) -> list[OpenPosition]:
    """Read trades rows with outcome='OPEN' from the local DB. The DB is
    source of truth — posttool_fill_capture writes here after each fill,
    so the hook sees exactly what the journal considers live."""
    out: list[OpenPosition] = []
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT t.symbol, t.asset_type, t.side, t.qty, t.entry_price,
                   t.stop, t.strategy_label, th.ticker
            FROM trades t
            LEFT JOIN theses th ON th.id = t.thesis_id
            WHERE t.outcome = 'OPEN'
            """
        ).fetchall()
    for r in rows:
        symbol = r["symbol"]
        # Trust the trade's symbol over the thesis ticker — a symbol is
        # what actually trades; a thesis can be wrong or reused.
        ticker = _ticker_from_any_symbol(symbol) or r["ticker"] or symbol
        ticker_u = ticker.upper()
        out.append(OpenPosition(
            symbol=symbol,
            ticker=ticker_u,
            asset_type=r["asset_type"],
            qty=float(r["qty"] or 0),
            entry_price=float(r["entry_price"] or 0),
            stop=(float(r["stop"]) if r["stop"] is not None else None),
            sector=sector_map.get(ticker_u),
            strategy_label=r["strategy_label"],
        ))
    return out


def _ticker_from_any_symbol(symbol: str) -> str | None:
    """Best-effort: US.AAPL → AAPL, US.AAPL250117C00200000 → AAPL."""
    if not symbol:
        return None
    s = symbol.upper()
    if s.startswith("US."):
        s = s[3:]
    m = OCC_RE.match(s)
    if m:
        return m.group(1)
    return s


def _has_fresh_open_thesis(ticker: str) -> tuple[bool, int | None]:
    """True iff there's an open thesis for `ticker` created within the last
    THESIS_FRESHNESS_MIN minutes. Returns (ok, thesis_id).

    NOTE: We parse created_at in Python rather than compare via SQLite's
    datetime('now','-X minutes'). The theses table stores timezone-aware
    ISO-8601 strings ('2026-04-22T01:10:00+00:00'); SQLite's datetime(...)
    returns space-separated, timezone-naive strings. Lexicographic compare
    of the two forms is broken ('T' > ' '), so every open thesis looked
    fresh regardless of age. Parsing in Python avoids the string pitfall.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=THESIS_FRESHNESS_MIN)
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at FROM theses
            WHERE ticker = ? AND status = 'open'
            ORDER BY created_at DESC
            """,
            (ticker.upper(),),
        ).fetchall()
    for r in rows:
        try:
            created = datetime.fromisoformat(r["created_at"])
        except Exception:
            continue
        # Defensive: naive strings get interpreted as UTC.
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if created >= cutoff:
            return True, int(r["id"])
    return False, None


# ---- main flow ----

def _build_proposed(
    tool_name: str,
    tool_input: dict,
    sector_map: dict[str, str],
) -> ProposedTrade | str:
    """Returns a ProposedTrade, or a string error message to block with."""
    is_option = tool_name.endswith("__place_paper_option_order")
    side = str(tool_input.get("side", "")).upper()
    if side not in {"BUY", "SELL"}:
        return f"unknown side {side!r}"

    strategy_label = tool_input.get("strategy_label")

    if is_option:
        opt_sym = tool_input.get("option_symbol", "")
        parsed = _parse_option_symbol(opt_sym)
        if not parsed:
            return (
                f"unparseable option_symbol {opt_sym!r}; expected "
                "'US.<TICKER><YYMMDD><C|P><strike*1000 8d>'"
            )
        ticker, dte_from_symbol = parsed
        # Caller-provided dte wins (lets the skill pass explicit values);
        # fall back to date-derived DTE otherwise.
        dte = tool_input.get("dte")
        if dte is None:
            dte = dte_from_symbol
        delta = tool_input.get("delta")
        contracts = float(tool_input.get("contracts", 0))
        price = float(tool_input.get("price", 0))
        return ProposedTrade(
            ticker=ticker,
            asset_type="OPT",
            side=side,  # type: ignore[arg-type]
            qty=contracts,
            entry_price=price,
            stop=None,
            strategy_label=strategy_label,
            sector=sector_map.get(ticker.upper()),
            delta=(float(delta) if delta is not None else None),
            dte=(int(dte) if dte is not None else None),
            earnings_dte=tool_input.get("earnings_dte"),
        )

    # stock branch
    sym = tool_input.get("symbol", "")
    ticker = _bare_ticker(sym)
    if not ticker:
        return f"unparseable stock symbol {sym!r}"
    stop = tool_input.get("stop")
    qty = float(tool_input.get("qty", 0))
    price = float(tool_input.get("price", 0))
    return ProposedTrade(
        ticker=ticker,
        asset_type="STK",
        side=side,  # type: ignore[arg-type]
        qty=qty,
        entry_price=price,
        stop=(float(stop) if stop is not None else None),
        strategy_label=strategy_label,
        sector=sector_map.get(ticker),
        earnings_dte=tool_input.get("earnings_dte"),
    )


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as e:
        sys.stderr.write(f"[pretool_order_guard] bad payload: {e}\n")
        return 0  # infra problem, not a policy violation

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {}) or {}

    if not ORDER_TOOL_RE.match(tool_name):
        # Not our business.
        return 0

    # ---- parse proposed trade ----
    sector_map = _load_sector_map()
    proposed = _build_proposed(tool_name, tool_input, sector_map)
    if isinstance(proposed, str):
        return _block(proposed, tool_name, detail={"tool_input": tool_input})

    ticker = proposed.ticker
    side = proposed.side

    # ---- thesis freshness (opening trades only) ----
    # SELLs that close existing positions don't need a fresh thesis; the
    # original thesis was recorded when the position was opened. But we
    # still require that the position's thesis exists on file.
    if side == "BUY":
        ok, thesis_id = _has_fresh_open_thesis(ticker)
        if not ok:
            return _block(
                f"no open thesis for {ticker} in the last "
                f"{THESIS_FRESHNESS_MIN} minutes",
                tool_name,
                detail={"ticker": ticker},
            )
    else:
        thesis_id = None  # SELL path; skip gate

    # ---- sizing ----
    # Equity from live SDK (paper account). Fail-closed on any SDK issue.
    try:
        equity = _fetch_equity_from_sdk()
    except Exception as e:
        return _block(
            "cannot verify sizing — moomoo equity lookup failed",
            tool_name,
            detail={"error": repr(e)},
        )

    opens = _load_open_positions(sector_map)
    ctx = SizingContext(
        equity=equity,
        opens=tuple(opens),
        sector_lookup_available=bool(sector_map),
    )
    vs = check(ctx, proposed)
    bs = blockers(vs)
    warns = [f"{v.rule}: {v.message}" for v in vs if v.severity == "warn"]

    if bs:
        return _block(
            "sizing rules violated",
            tool_name,
            detail={
                "violations": [
                    {"rule": v.rule, "message": v.message, "severity": v.severity}
                    for v in vs
                ],
                "proposed": {
                    "ticker": proposed.ticker,
                    "asset_type": proposed.asset_type,
                    "side": proposed.side,
                    "qty": proposed.qty,
                    "entry_price": proposed.entry_price,
                    "stop": proposed.stop,
                    "sector": proposed.sector,
                    "strategy_label": proposed.strategy_label,
                    "dte": proposed.dte,
                    "delta": proposed.delta,
                },
                "equity": equity,
                "open_count": len(opens),
            },
        )

    return _allow(
        tool_name,
        note=f"thesis_id={thesis_id} ticker={ticker} equity=${equity:,.2f} "
             f"opens={len(opens)}",
        warns=warns,
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        # Unhandled exception = fail-closed. The order must not silently go
        # through just because the hook crashed.
        sys.stderr.write(
            f"[pretool_order_guard] CRASH — failing closed:\n"
            f"{traceback.format_exc()}\n"
        )
        sys.exit(2)
