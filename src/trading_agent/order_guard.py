"""Shared order-guard engine — thesis freshness + R1-R7 sizing at order time.

One implementation, two enforcement layers:

  - ``hooks/pretool_order_guard.py`` (client-side, Claude Code PreToolUse)
  - ``mcp_servers/moomoo/server.py``  (server-side, inside the order tools)

Until 2026-06 the guard lived ONLY in the hook, so the autonomous graph,
direct MCP calls from other clients, and scripts could place orders with
zero evaluation (verified: MRVL 2x and AAPL 8x spreads entered with no
guard row in hook_audit_log). The server-side call makes the order tools
themselves the choke point; the hook stays as an earlier, friendlier layer.
Keeping both on this module means a sizing fix (e.g. d089349's R1
close-exemption) lands in every layer at once.

Also owned here:

  - SHORT-OPTION-OPEN HARD BLOCK: SELL-to-open option legs are refused
    outright (``R_NO_SHORT_OPEN``). Per-order sizing math cannot see a
    multi-leg combo, so a "spread" legged in as two orders evaluates its
    short leg as a naked short — the AAPL 8x SELL-to-open 300P leg carried
    ~$235k assignment exposure that registered zero portfolio heat. Until
    combos are first-class (atomic width-minus-credit sizing), no SELL may
    open an option position. SELL-to-close stays exempt (intent='close',
    same journal-lookup logic d089349 relies on).

  - AUDIT: every evaluation writes a row to the ``hook_audit_log`` table in
    trader.db (and a JSONL line in data/hook_audit.log). The nightly
    ``jobs/reconcile_order_guard.py`` joins opening fills against these rows
    and alerts on any fill that no guard layer evaluated.

Import-light by design: trading_agent.{config,db,sizing} only — no moomoo
SDK, no FastMCP — so the PreToolUse hook keeps its fast cold start.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from trading_agent.config import CONFIG, ensure_dirs
from trading_agent.db import connection
from trading_agent.sizing import (
    OpenPosition,
    ProposedTrade,
    SizingContext,
    SizingViolation,
    blockers,
    check,
)

OCC_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{4,8})$")
THESIS_FRESHNESS_MIN = 10

# Temporary hard policy (see module docstring): no SELL order may OPEN an
# option position until multi-leg combos get atomic sizing. Grep-able rule
# code, same shape as the R* codes in sizing.py.
R_NO_SHORT_OPEN = "R_short_option_open_blocked"

AUDIT_PATH = CONFIG.data_dir / "hook_audit.log"
SECTORS_CSV = CONFIG.data_dir / "sectors.csv"


# ---- symbol parsing ----

def bare_ticker(symbol: str) -> str | None:
    """US.AAPL → AAPL. Returns None if unparseable or an option code."""
    if not symbol:
        return None
    s = symbol.strip().upper()
    if s.startswith("US."):
        s = s[3:]
    if not s:
        return None
    if OCC_RE.match(s):
        return None
    return s


def parse_option_symbol(option_symbol: str) -> tuple[str, int, str, float] | None:
    """Moomoo option code → (ticker, dte_from_today, right, strike).

    Returns None on bad input. ``right`` ∈ {"C","P"}; ``strike`` is in
    dollars (the OCC integer is divided by 1000).

    Moomoo format: US.<TICKER><YYMMDD><C|P><strike*1000> where the strike
    is unpadded (e.g., US.AAPL260515C267500 = $267.500 strike).
    """
    s = option_symbol.strip().upper()
    if s.startswith("US."):
        s = s[3:]
    m = OCC_RE.match(s)
    if not m:
        return None
    ticker = m.group(1)
    yymmdd = m.group(2)
    right = m.group(3)
    strike_int = m.group(4)
    try:
        expiry = datetime.strptime(yymmdd, "%y%m%d").date()
        strike = int(strike_int) / 1000.0
    except ValueError:
        return None
    today = datetime.now(timezone.utc).date()
    dte = (expiry - today).days
    return ticker, dte, right, strike


def ticker_from_any_symbol(symbol: str) -> str | None:
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


# ---- data sources ----

def load_sector_map() -> dict[str, str]:
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


def load_open_positions(sector_map: dict[str, str]) -> list[OpenPosition]:
    """Read trades rows with outcome='OPEN' from the local DB. The DB is
    source of truth — posttool_fill_capture writes here after each fill,
    so the guard sees exactly what the journal considers live.

    Provenance filter: only the system's OWN positions (agent + virtual)
    count toward R2/R3/R4. The journal also carries 54 'real_mirror' shadow
    rows (operator real-account fills mirrored for post-mortems, permanently
    OPEN by design) plus backfills/dry-runs — counting those blocked EVERY
    opening order on R2_max_open ('already 58 open positions; cap is 6')
    the moment the provenance migration and the in-tool guard merged."""
    out: list[OpenPosition] = []
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT t.symbol, t.asset_type, t.side, t.qty, t.entry_price,
                   t.stop, t.strategy_label, th.ticker
            FROM trades t
            LEFT JOIN theses th ON th.id = t.thesis_id
            WHERE t.outcome = 'OPEN'
              AND COALESCE(t.provenance, 'agent') IN ('agent', 'virtual')
            """
        ).fetchall()
    for r in rows:
        symbol = r["symbol"]
        # Trust the trade's symbol over the thesis ticker — a symbol is
        # what actually trades; a thesis can be wrong or reused.
        ticker = ticker_from_any_symbol(symbol) or r["ticker"] or symbol
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


def has_fresh_open_thesis(ticker: str) -> tuple[bool, int | None]:
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


def has_existing_open_for(broker_symbol: str) -> bool:
    """True iff a journal (SQLite OR Postgres) has an OPEN trade for the
    given broker symbol. Used to compute ``intent`` on the ProposedTrade:
    SELL of an option we don't already hold = opening a SHORT; SELL of
    one we do hold = closing the LONG.

    BOTH stores must be consulted: interactive /enter fills land in the
    SQLite journal, but autonomous-graph entries are journaled ONLY in
    Postgres journal_trades (persist_trade_event). With the SQLite-only
    lookup, every autonomous position's SELL-to-close was classified as
    SELL-to-open and hard-blocked by R_short_option_open_blocked — i.e.
    the guard would have prevented the system from exiting its own trades.
    """
    if not broker_symbol:
        return False
    sqlite_says_no = False
    try:
        with connection() as conn:
            # Same provenance filter as load_open_positions: a real_mirror
            # row is the OPERATOR's real holding — the paper account holds
            # nothing, so a SELL against it is an opening short, not a close.
            row = conn.execute(
                "SELECT 1 FROM trades WHERE symbol = ? AND outcome = 'OPEN' "
                "AND COALESCE(provenance, 'agent') IN ('agent', 'virtual') "
                "LIMIT 1",
                (broker_symbol,),
            ).fetchone()
            if row is not None:
                return True
            sqlite_says_no = True
    except Exception:
        pass  # SQLite unknown — fall through to Postgres, then fail open
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                "SELECT 1 FROM journal_trades "
                "WHERE symbol = %s AND outcome = 'OPEN' LIMIT 1",
                (broker_symbol,),
            )
            if cur.fetchone() is not None:
                return True
    except Exception:
        # Postgres unreachable is NORMAL on the Phase-1 Mac (no PG server).
        # If SQLite answered definitively "no open trade", trust it and
        # classify as opening — returning True here would mark every SELL
        # as a close and silently disable the SELL-to-open block in exactly
        # the environment the operator trades interactively.
        pass
    if sqlite_says_no:
        return False
    # SQLite errored and Postgres found nothing / errored: fail open to
    # close (more conservative — SHORT-specific R5b/R5c won't fire
    # spuriously on a real close).
    return True


# ---- proposed-trade construction ----

def build_proposed(
    kind: Literal["stock", "option"],
    params: dict,
    sector_map: dict[str, str],
) -> ProposedTrade | str:
    """Build a ProposedTrade from order-tool arguments.

    ``params`` carries the tool_input of place_paper_order (stock) or
    place_paper_option_order (option). Returns a string error message
    (caller blocks with it) when the symbol is unparseable.
    """
    side = str(params.get("side", "")).upper()
    if side not in {"BUY", "SELL"}:
        return f"unknown side {side!r}"

    strategy_label = params.get("strategy_label")

    if kind == "option":
        opt_sym = params.get("option_symbol", "")
        parsed = parse_option_symbol(opt_sym)
        if not parsed:
            return (
                f"unparseable option_symbol {opt_sym!r}; expected "
                "'US.<TICKER><YYMMDD><C|P><strike*1000 8d>'"
            )
        ticker, dte_from_symbol, right, strike = parsed
        # Caller-provided dte wins (lets the skill pass explicit values);
        # fall back to date-derived DTE otherwise.
        dte = params.get("dte")
        if dte is None:
            dte = dte_from_symbol
        delta = params.get("delta")
        contracts = float(params.get("contracts", 0))
        price = float(params.get("price", 0))
        # Intent: SELL is OPEN (new short) iff no existing OPEN position
        # has matching option_symbol; otherwise it's CLOSE of a long.
        # BUY is OPEN (new long) iff no existing position; otherwise it's
        # CLOSE of a short (buying back).
        opens_for_symbol = has_existing_open_for(opt_sym)
        intent = "close" if opens_for_symbol else "open"
        stop = params.get("stop")  # callers MAY pass stop; R5c needs it
        target = params.get("target")  # CSP / strategy code may pass it
        return ProposedTrade(
            ticker=ticker,
            asset_type="OPT",
            side=side,  # type: ignore[arg-type]
            qty=contracts,
            entry_price=price,
            stop=(float(stop) if stop is not None else None),
            target=(float(target) if target is not None else None),
            strategy_label=strategy_label,
            sector=sector_map.get(ticker.upper()),
            delta=(float(delta) if delta is not None else None),
            dte=(int(dte) if dte is not None else None),
            earnings_dte=params.get("earnings_dte"),
            right=right,  # type: ignore[arg-type]
            strike=strike,
            intent=intent,  # type: ignore[arg-type]
        )

    # stock branch
    sym = params.get("symbol", "")
    ticker = bare_ticker(sym)
    if not ticker:
        return f"unparseable stock symbol {sym!r}"
    stop = params.get("stop")
    qty = float(params.get("qty", 0))
    price = float(params.get("price", 0))
    # Intent: SELL of a stock we hold is an exit (skip thesis gate +
    # opening-only rules — blocking a risk-reducing close is the d089349
    # failure mode). SELL without a held position opens a short; BUY is
    # always an open (a new long OR an add-on, both increase exposure).
    intent = "close" if (side == "SELL" and has_existing_open_for(sym)) else "open"
    return ProposedTrade(
        ticker=ticker,
        asset_type="STK",
        side=side,  # type: ignore[arg-type]
        qty=qty,
        entry_price=price,
        stop=(float(stop) if stop is not None else None),
        target=(float(params.get("target")) if params.get("target") is not None else None),
        strategy_label=strategy_label,
        sector=sector_map.get(ticker),
        earnings_dte=params.get("earnings_dte"),
        intent=intent,  # type: ignore[arg-type]
    )


# ---- evaluation ----

@dataclass(frozen=True)
class GuardDecision:
    """Outcome of one guard evaluation, ready for audit + caller dispatch."""
    allowed: bool
    reason: str                          # short machine-greppable summary
    symbol: str                          # broker symbol as submitted
    ticker: str | None = None
    intent: str | None = None            # open/close (None if parse failed)
    thesis_id: int | None = None
    violations: tuple[SizingViolation, ...] = ()
    warns: tuple[str, ...] = ()
    equity: float | None = None
    open_count: int | None = None
    proposed: dict | None = None          # serialized ProposedTrade summary

    def violations_json(self) -> list[dict]:
        return [
            {"rule": v.rule, "message": v.message, "severity": v.severity}
            for v in self.violations
        ]


def evaluate_order(
    kind: Literal["stock", "option"],
    params: dict,
    *,
    equity: float,
    cash: float | None,
) -> GuardDecision:
    """Run the full order gate: parse → intent → thesis freshness →
    short-option-open hard block → R1-R7 sizing. Pure DB reads; the caller
    supplies live equity/cash (each layer has its own broker transport).
    """
    symbol = str(params.get("option_symbol") or params.get("symbol") or "")
    sector_map = load_sector_map()
    proposed = build_proposed(kind, params, sector_map)
    if isinstance(proposed, str):
        return GuardDecision(allowed=False, reason=proposed, symbol=symbol)

    proposed_summary = {
        "ticker": proposed.ticker,
        "asset_type": proposed.asset_type,
        "side": proposed.side,
        "qty": proposed.qty,
        "entry_price": proposed.entry_price,
        "stop": proposed.stop,
        "target": proposed.target,
        "sector": proposed.sector,
        "strategy_label": proposed.strategy_label,
        "dte": proposed.dte,
        "delta": proposed.delta,
        "intent": proposed.intent,
    }

    # ---- thesis freshness (opening trades only) ----
    # Any OPEN (long buy OR short sell) needs a fresh thesis. CLOSE
    # orders (selling a long / buying back a short) don't, because the
    # original thesis was recorded when the position was opened.
    thesis_id: int | None = None
    if proposed.intent == "open":
        ok, thesis_id = has_fresh_open_thesis(proposed.ticker)
        if not ok:
            return GuardDecision(
                allowed=False,
                reason=(
                    f"no open thesis for {proposed.ticker} in the last "
                    f"{THESIS_FRESHNESS_MIN} minutes"
                ),
                symbol=symbol,
                ticker=proposed.ticker,
                intent=proposed.intent,
                proposed=proposed_summary,
            )

    # ---- short-option-open hard block ----
    if proposed.is_opening_option_short:
        v = SizingViolation(
            R_NO_SHORT_OPEN,
            f"SELL-to-open option leg refused: {symbol}. Per-order sizing "
            f"cannot bound a short leg's assignment exposure (a legged-in "
            f"'spread' evaluates as a naked short). Short premium and "
            f"multi-leg combos are blocked at the order layer until combos "
            f"get atomic width-minus-credit sizing. SELL-to-close of an "
            f"existing long is unaffected.",
            "block",
        )
        return GuardDecision(
            allowed=False,
            reason="short option open blocked",
            symbol=symbol,
            ticker=proposed.ticker,
            intent=proposed.intent,
            thesis_id=thesis_id,
            violations=(v,),
            proposed=proposed_summary,
        )

    # ---- sizing (R1-R7) ----
    opens = load_open_positions(sector_map)
    ctx = SizingContext(
        equity=equity,
        cash=cash,
        opens=tuple(opens),
        sector_lookup_available=bool(sector_map),
    )
    vs = check(ctx, proposed)
    bs = blockers(vs)
    warns = tuple(f"{v.rule}: {v.message}" for v in vs if v.severity == "warn")

    if bs:
        return GuardDecision(
            allowed=False,
            reason="sizing rules violated",
            symbol=symbol,
            ticker=proposed.ticker,
            intent=proposed.intent,
            thesis_id=thesis_id,
            violations=tuple(vs),
            warns=warns,
            equity=equity,
            open_count=len(opens),
            proposed=proposed_summary,
        )

    return GuardDecision(
        allowed=True,
        reason="ok",
        symbol=symbol,
        ticker=proposed.ticker,
        intent=proposed.intent,
        thesis_id=thesis_id,
        violations=tuple(vs),
        warns=warns,
        equity=equity,
        open_count=len(opens),
        proposed=proposed_summary,
    )


# ---- audit ----

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def audit_jsonl(rec: dict) -> None:
    """Append one line to data/hook_audit.log. Never raises."""
    try:
        ensure_dirs()
        with AUDIT_PATH.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


def audit_decision(
    decision: GuardDecision,
    *,
    guard_name: str,
    tool_name: str,
) -> None:
    """Persist a guard evaluation to hook_audit_log (DB) + JSONL.

    The DB row is what jobs/reconcile_order_guard.py joins fills against —
    its payload MUST carry the broker symbol. Failure to write is reported
    to stderr but never blocks: a missing 'allow' row makes the nightly
    reconciliation alert (fail-safe direction), while failing the order on
    an audit hiccup would be a new outage mode.
    """
    payload = {
        "symbol": decision.symbol,
        "ticker": decision.ticker,
        "intent": decision.intent,
        "thesis_id": decision.thesis_id,
        "equity": decision.equity,
        "open_count": decision.open_count,
        "proposed": decision.proposed,
        "violations": decision.violations_json(),
        "warns": list(decision.warns),
    }
    verdict = "allow" if decision.allowed else "block"
    try:
        with connection() as conn:
            conn.execute(
                "INSERT INTO hook_audit_log "
                "(created_at, hook_name, tool_name, decision, reason, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (_now(), guard_name, tool_name, verdict, decision.reason,
                 json.dumps(payload, default=str)),
            )
    except Exception as e:
        print(
            f"[{guard_name}] hook_audit_log write failed (order still "
            f"{verdict}ed; nightly reconcile will flag it): {e!r}",
            file=sys.stderr,
        )
    audit_jsonl({
        "ts": _now(),
        "hook": guard_name,
        "tool": tool_name,
        "decision": verdict,
        "reason": decision.reason,
        "detail": payload,
    })
