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

  - R5d OPTION LIQUIDITY GATE: an OPENING option order must trade a liquid
    contract (quoted spread <= 5% of mid, OI >= 500 at the strike, checked
    against a live snapshot fetched at order time). Until 2026-06 the only
    spread/OI screening was prose in the options-strategy skill — nothing
    deterministic — so the autonomous synthesizer bought at the ask with
    zero spread check. Closes are ALWAYS exempt: the guard must never trap
    an exit (the d089349 failure mode); a wide spread on the way out is a
    cost the journal measures, not a reason to hold.

  - R8 DAILY -2R CIRCUIT BREAKER: once today's realized losing trades sum
    to −2R (R = sizing.MAX_SINGLE_RISK_PCT × live equity, the same R1 risk
    unit), NEW opening orders are refused for the rest of the trading day.
    Closes are always exempt. Both journals are consulted (SQLite trades +
    Postgres journal_trades — disjoint populations); one store down → warn
    and use the other, both down → block opens (fail-closed on blindness).

  - AUDIT: every evaluation writes a row to the ``hook_audit_log`` table in
    trader.db (and a JSONL line in data/hook_audit.log). The nightly
    ``jobs/reconcile_order_guard.py`` joins opening fills against these rows
    and alerts on any fill that no guard layer evaluated.

Import-light by design: trading_agent.{config,db,sizing} only — no moomoo
SDK, no FastMCP — so the PreToolUse hook keeps its fast cold start. The one
exception is R5d's quote fetch, which lazily imports the moomoo server
INSIDE the function and only for OPENING option orders; module import stays
cheap. (Hook processes that trigger it must call the server's shutdown() —
the SDK's quote context starts non-daemon threads that otherwise outlive
sys.exit; see mcp_servers/moomoo/server.py shutdown docstring.)
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
    MAX_SINGLE_RISK_PCT,
    R5E,
    ComboLeg,
    OpenPosition,
    ProposedCombo,
    ProposedTrade,
    SizingContext,
    SizingViolation,
    blockers,
    check,
    check_combo,
)

OCC_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{4,8})$")
THESIS_FRESHNESS_MIN = 10

# Temporary hard policy (see module docstring): no SELL order may OPEN an
# option position until multi-leg combos get atomic sizing. Grep-able rule
# code, same shape as the R* codes in sizing.py.
R_NO_SHORT_OPEN = "R_short_option_open_blocked"

# Rule R5d — liquidity gate for OPENING option orders. The options-strategy
# skill carried these as prose only (spread < 10% of mid, OI > 100), which
# the autonomous synthesizer never read at order time: it bought at the ask
# with zero spread check, and wide-spread entries are the instant drawdown
# execution_costs.py now measures but nothing prevented. Tighter-than-prose
# thresholds, enforced here so every layer (hook, server, graph) gets them.
# Closing option orders are ALWAYS exempt — never block an exit.
LIQ_MAX_SPREAD_PCT_OF_MID = 0.05   # quoted (ask − bid) must be <= 5% of mid
LIQ_MIN_OPEN_INTEREST = 500        # contracts of OI required at the strike
R5D = "R5d_illiquid"               # spread or OI failed the thresholds
R5D_UNQUOTABLE = "R5d_unquotable"  # no usable live bid/ask → block (fail-closed)
R5D_OI_UNKNOWN = "R5d_oi_unknown"  # snapshot omits OI → warn (spread is primary)

# Rule R8 — daily -2R circuit breaker (revival plan Week 1 Step 7).
# Deterministic and in the ORDER path so it binds every layer (hook, MCP
# server, autonomous graph) even when the LLM channel is down or degraded —
# the 6/20→7/8 OAuth incident proved graph-only guards can go silently
# offline. R is the SAME per-trade risk unit the sizing layer budgets for
# R1: sizing.MAX_SINGLE_RISK_PCT × live equity (no hardcoded dollar figure;
# equity moves, so does the trip wire). Once TODAY's realized losing trades
# sum to −2R or worse, NEW opening orders are refused for the rest of the
# trading day. Closes are NEVER touched — reducing risk is always allowed.
DAILY_LOSS_BREAKER_R_MULT = 2.0
R8 = "R8_daily_loss_breaker"
R8_STORE_BLIND = "R8_store_blind"        # BOTH journals unreachable → block opens
R8_STORE_DEGRADED = "R8_store_degraded"  # one journal unreachable → warn, use other

# Rows that are not the system's own live results must not trip (or pad) the
# breaker: real_mirror = operator's real-account shadow rows, virtual_backfill
# = reconstructed history, dry_run = rehearsals. agent + virtual count.
_BREAKER_EXCLUDED_PROVENANCE = ("real_mirror", "virtual_backfill", "dry_run")

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
                   t.stop, t.strategy_label, th.ticker,
                   (SELECT ms.payload FROM market_snapshots ms
                    WHERE ms.trade_id = t.id
                    ORDER BY ms.taken_at DESC LIMIT 1) AS snapshot_payload
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
        # Defined-risk combos are journaled with entry_price=net_credit (for
        # P&L); their true exposure is the max_loss stored in the snapshot.
        # Without this override R3/R4 would under-count the spread as just the
        # small credit collected.
        risk_notional: float | None = None
        sp = r["snapshot_payload"] if "snapshot_payload" in r.keys() else None
        if sp:
            try:
                payload = json.loads(sp) if isinstance(sp, str) else sp
                if isinstance(payload, dict) and payload.get("combo"):
                    ml = payload.get("max_loss")
                    if ml is not None:
                        risk_notional = float(ml)
            except (ValueError, TypeError):
                pass
        out.append(OpenPosition(
            symbol=symbol,
            ticker=ticker_u,
            asset_type=r["asset_type"],
            qty=float(r["qty"] or 0),
            entry_price=float(r["entry_price"] or 0),
            stop=(float(r["stop"]) if r["stop"] is not None else None),
            sector=sector_map.get(ticker_u),
            strategy_label=r["strategy_label"],
            risk_notional=risk_notional,
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


#: Sentinel returned by ``open_position_side_for`` when a journal row is
#: known/presumed to exist but its side could not be read (store errors,
#: NULL side column). Callers must fail CONSERVATIVELY on it: a SELL keeps
#: today's close classification (R5b/R5c must not fire on a real close),
#: while a BUY classifies as OPEN so every opening gate still applies.
SIDE_UNKNOWN = "UNKNOWN"


def open_position_side_for(broker_symbol: str) -> str | None:
    """Side of the journal's OPEN trade for ``broker_symbol``, or None.

    Returns:
      'BUY'        — an open LONG position exists (we hold it)
      'SELL'       — an open SHORT position exists (e.g. the short leg of a
                     defined-risk combo, journaled side=SELL)
      SIDE_UNKNOWN — a row exists (or the stores are unreachable and we
                     cannot prove one doesn't) but the side is unknown
      None         — definitively no open position in either journal

    Used to compute ``intent`` on the ProposedTrade. Side-awareness matters
    for BUY orders: BUY against an existing SHORT is a buy-back (close),
    but BUY against an existing LONG is an ADD-ON that increases exposure
    and must run every opening gate (status gate, thesis freshness, R5d,
    sizing). Pre-fix, ANY existing row classified a BUY as 'close' — the
    one remaining single-leg bypass of the convexity retirement.

    BOTH stores must be consulted: interactive /enter fills land in the
    SQLite journal, but autonomous-graph entries are journaled ONLY in
    Postgres journal_trades (persist_trade_event). With the SQLite-only
    lookup, every autonomous position's SELL-to-close was classified as
    SELL-to-open and hard-blocked by R_short_option_open_blocked — i.e.
    the guard would have prevented the system from exiting its own trades.
    """
    if not broker_symbol:
        return None
    sqlite_says_no = False
    try:
        with connection() as conn:
            # Same provenance filter as load_open_positions: a real_mirror
            # row is the OPERATOR's real holding — the paper account holds
            # nothing, so a SELL against it is an opening short, not a close.
            row = conn.execute(
                "SELECT side FROM trades WHERE symbol = ? AND outcome = 'OPEN' "
                "AND COALESCE(provenance, 'agent') IN ('agent', 'virtual') "
                "LIMIT 1",
                (broker_symbol,),
            ).fetchone()
            if row is not None:
                side = row["side"]
                return str(side).strip().upper() if side else SIDE_UNKNOWN
            sqlite_says_no = True
    except Exception:
        pass  # SQLite unknown — fall through to Postgres, then fail open
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                "SELECT side FROM journal_trades "
                "WHERE symbol = %s AND outcome = 'OPEN' LIMIT 1",
                (broker_symbol,),
            )
            row = cur.fetchone()
            if row is not None:
                side = row.get("side") if isinstance(row, dict) else row[0]
                return str(side).strip().upper() if side else SIDE_UNKNOWN
    except Exception:
        # Postgres unreachable is NORMAL on the Phase-1 Mac (no PG server).
        # If SQLite answered definitively "no open trade", trust it and
        # classify as opening — treating unknown as an existing position
        # would mark every SELL as a close and silently disable the
        # SELL-to-open block in exactly the environment the operator
        # trades interactively.
        pass
    # Combo long wings: a vertical is journaled as ONE row keyed by the SHORT
    # leg's symbol, so a direct-symbol lookup can never see the protective
    # LONG leg. Without this, every SELL-to-close of the wing (the atomic
    # combo close's second leg, and every fill-confirm re-placement of it)
    # classifies as SELL-to-OPEN and is hard-blocked by R_NO_SHORT_OPEN.
    try:
        from trading_agent.combo import open_combo_long_legs
        if broker_symbol in open_combo_long_legs():
            return True
    except Exception:
        pass
    if sqlite_says_no:
        return None
    # SQLite errored and Postgres found nothing / errored: a position may
    # exist but we cannot see it — SIDE_UNKNOWN. For SELL that fails open
    # to close (more conservative — SHORT-specific R5b/R5c won't fire
    # spuriously on a real close); for BUY it fails CLOSED to open (all
    # opening gates apply).
    return SIDE_UNKNOWN


def has_existing_open_for(broker_symbol: str) -> bool:
    """True iff a journal (SQLite OR Postgres) has an OPEN trade for the
    given broker symbol — or the stores are unreachable and we cannot prove
    one doesn't (fail open to 'close' for SELL classification). Thin
    wrapper over ``open_position_side_for``; see it for the dual-store
    rationale."""
    return open_position_side_for(broker_symbol) is not None


# ---- R5d option liquidity ----

_QUOTE_FETCH_TIMEOUT_S = 5.0


def fetch_option_quote(option_code: str,
                       timeout_s: float = _QUOTE_FETCH_TIMEOUT_S) -> dict | None:
    """Live snapshot row for one option code, or None on ANY failure.

    Lazy import keeps this module import-light for the PreToolUse hook's
    cold start (see module docstring); the moomoo SDK loads only when an
    OPENING option order actually needs a liquidity read — and the broker
    order itself needs OpenD anyway, so this adds no availability
    dependency the order didn't already have. The thread+timeout mirrors
    learning/excursion.py: get_market_snapshot can hang on option
    contracts that aren't subscribed (no auto-subscribe on the free OpenD
    path), and a hang here would stall the order tool itself.
    """
    import concurrent.futures
    try:
        from trading_agent.mcp_servers.moomoo.server import get_quote
    except Exception:
        return None
    # NOT a context manager: with-block exit calls shutdown(wait=True), which
    # JOINS a hung get_quote worker — stalling the order tool/hook in exactly
    # the unsubscribed-contract hang this timeout exists for. On timeout we
    # abandon the worker thread (shutdown(wait=False)) and fail closed to
    # R5d_unquotable; the hook's moomoo server.shutdown() + hard process exit
    # reap what's left.
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = ex.submit(get_quote, [option_code])
        try:
            result = future.result(timeout=timeout_s)
        except Exception:
            return None
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    rows = (result or {}).get("rows") or []
    return rows[0] if rows else None


def _pos_float(*vals) -> float | None:
    """First value that parses to a float > 0, else None. Zero is None on
    purpose: a 0 bid/ask in a snapshot means 'no quote on that side'."""
    for v in vals:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0:
            return f
    return None


def check_option_liquidity(option_code: str,
                           quote: dict | None) -> list[SizingViolation]:
    """Rule R5d for one OPENING option order, given a live snapshot row.

    Returns SizingViolation entries (same shape as R1-R7) so the caller
    folds them into the standard blockers/warns flow and every outcome
    lands in hook_audit_log like any other rule.

    Field spellings vary across moomoo snapshot payloads: bid/ask arrive
    as 'bid_price'/'ask_price' or 'bid'/'ask'; open interest as
    'option_open_interest' or 'open_interest' (graph/nodes/trade_nodes.py
    and learning/excursion.py handle the same dual spellings). Failure
    semantics: no quote at all, or a missing/zero bid or ask, is
    unquotable → BLOCK — an entry you can't price honestly is an entry
    you skip. A payload that quotes a tight spread but omits OI is a
    warn only — some snapshots omit OI, and spread is the primary
    protection.
    """
    if quote is None:
        return [SizingViolation(
            R5D_UNQUOTABLE,
            f"no live quote for {option_code} (fetch failed, timed out, or "
            f"returned no rows). An opening option order you cannot price "
            f"honestly is an order you skip; closes are exempt from R5d.",
            "block",
        )]
    bid = _pos_float(quote.get("bid_price"), quote.get("bid"))
    ask = _pos_float(quote.get("ask_price"), quote.get("ask"))
    if bid is None or ask is None:
        return [SizingViolation(
            R5D_UNQUOTABLE,
            f"{option_code} has no two-sided market (bid={bid}, ask={ask} "
            f"after parsing; missing/zero side = unquotable). Opening order "
            f"refused; closes are exempt from R5d.",
            "block",
        )]
    out: list[SizingViolation] = []
    mid = (bid + ask) / 2.0
    spread = max(0.0, ask - bid)  # crossed quotes (ask < bid) count as zero
    spread_pct = spread / mid
    if spread_pct > LIQ_MAX_SPREAD_PCT_OF_MID:
        out.append(SizingViolation(
            R5D,
            f"quoted spread ${spread:.2f} is {spread_pct:.1%} of mid "
            f"${mid:.2f} on {option_code}; R5d cap is "
            f"{LIQ_MAX_SPREAD_PCT_OF_MID:.0%}. Crossing a wide spread is an "
            f"instant, certain drawdown — pick a more liquid strike/expiry.",
            "block",
        ))
    oi: int | None = None
    for key in ("option_open_interest", "open_interest"):
        v = quote.get(key)
        if v is None:
            continue
        try:
            oi = int(float(v))
            break
        except (TypeError, ValueError):
            continue
    if oi is None:
        out.append(SizingViolation(
            R5D_OI_UNKNOWN,
            f"snapshot for {option_code} omits open interest; spread check "
            f"is the primary protection, so allowing with a warn.",
            "warn",
        ))
    elif oi < LIQ_MIN_OPEN_INTEREST:
        out.append(SizingViolation(
            R5D,
            f"open interest {oi} < {LIQ_MIN_OPEN_INTEREST} at this strike "
            f"({option_code}); thin OI is a roach motel — easy to get in, "
            f"hard to get out.",
            "block",
        ))
    return out


# ---- R8 daily -2R circuit breaker ----

def _sqlite_todays_losses(today_utc: str) -> float:
    """Sum (≤ 0) of today's realized LOSING P&L in the SQLite journal.

    Raises on any DB failure — the caller decides the fail direction.
    Losses only, not net P&L: a green morning must not re-arm a red
    afternoon (fail-closed beats fail-open on a live loop).
    ``substr(closed_at, 1, 10)`` compares the UTC date prefix, which works
    for both aware ('...T...+00:00') and naive ISO strings in that column.
    """
    with connection() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0.0) FROM trades "
            "WHERE outcome != 'OPEN' AND pnl IS NOT NULL AND pnl < 0 "
            "AND substr(closed_at, 1, 10) = ? "
            "AND COALESCE(provenance, 'agent') NOT IN (?, ?, ?)",
            (today_utc, *_BREAKER_EXCLUDED_PROVENANCE),
        ).fetchone()
    return float(row[0] or 0.0)


def _pg_todays_losses() -> float:
    """Sum (≤ 0) of today's (UTC) realized LOSING P&L in journal_trades.

    Raises when Postgres is unreachable — caller decides the fail
    direction. journal_trades has no pnl column, so dollars are derived
    the same way learning/outcome.py does: (exit − entry) × qty × contract
    multiplier, sign flipped for SELL-side entries. Provenance lives in
    broker_fill_json (persist_trade_event stamps 'agent'); the exclusion
    matches the SQLite filter.

    The two journals hold DISJOINT populations (see has_existing_open_for:
    interactive /enter fills → SQLite only, autonomous-graph entries →
    Postgres only), so summing both never double-counts a trade.
    """
    from trading_agent.store.postgres import cursor
    with cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(SUM(pnl_usd), 0.0) FROM (
                SELECT (exit_price - entry_price) * qty
                       * (CASE WHEN asset_type = 'OPT' THEN 100 ELSE 1 END)
                       * (CASE WHEN side = 'SELL' THEN -1 ELSE 1 END)
                       AS pnl_usd
                FROM journal_trades
                WHERE outcome <> 'OPEN'
                  AND exit_price IS NOT NULL
                  AND (closed_at AT TIME ZONE 'UTC')::date
                      = (NOW() AT TIME ZONE 'UTC')::date
                  AND COALESCE(broker_fill_json->>'provenance', 'agent')
                      NOT IN (%s, %s, %s)
            ) x
            WHERE pnl_usd < 0
            """,
            _BREAKER_EXCLUDED_PROVENANCE,
        )
        row = cur.fetchone()
    return float(row[0] or 0.0)


def check_daily_loss_breaker(equity: float, intent: str) -> list[SizingViolation]:
    """Rule R8: block NEW opens once today's realized losses reach −2R.

    Returns SizingViolation entries in the standard R1-R7 shape so callers
    fold them into the blockers/warns flow and hook_audit_log.

    Closes NEVER evaluate R8 — a breaker that traps exits is the d089349
    failure mode all over again, and a dead DB must not brick a close.
    Store-failure semantics for opens:
      * one journal unreachable  → WARN + evaluate against the reachable
        one (fail-closed to the store we can still see)
      * BOTH journals unreachable → BLOCK (total blindness on today's
        losses means we cannot prove the breaker hasn't tripped)
    """
    if intent != "open":
        return []
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    losses = 0.0
    reachable: list[str] = []
    degraded: list[str] = []
    try:
        losses += _sqlite_todays_losses(today_utc)
        reachable.append("sqlite")
    except Exception:
        degraded.append("sqlite")
    try:
        losses += _pg_todays_losses()
        reachable.append("postgres")
    except Exception:
        # Postgres unreachable is NORMAL on the Phase-1 Mac (no PG server);
        # the warn below still surfaces it in hook_audit_log.
        degraded.append("postgres")

    if not reachable:
        return [SizingViolation(
            R8_STORE_BLIND,
            "both journals (SQLite trades + Postgres journal_trades) are "
            "unreachable — today's realized losses are unknowable, so the "
            "R8 daily-loss breaker cannot be proven un-tripped. Opening "
            "order refused fail-closed; closes are exempt from R8.",
            "block",
        )]

    out: list[SizingViolation] = []
    if degraded:
        out.append(SizingViolation(
            R8_STORE_DEGRADED,
            f"journal(s) unreachable for R8: {', '.join(degraded)}; daily "
            f"loss evaluated against {', '.join(reachable)} only.",
            "warn",
        ))

    # R = the R1 per-trade risk budget at CURRENT equity. equity <= 0 makes
    # the trip wire meaningless (limit 0 would block every open on a flat
    # day); R1/R5 sizing already refuses anything sane at zero equity.
    limit = -DAILY_LOSS_BREAKER_R_MULT * MAX_SINGLE_RISK_PCT * equity
    if limit < 0 and losses <= limit:
        out.append(SizingViolation(
            R8,
            f"today's realized losses ${losses:,.2f} have reached the daily "
            f"circuit breaker of -{DAILY_LOSS_BREAKER_R_MULT:g}R "
            f"(${limit:,.2f}; R = {MAX_SINGLE_RISK_PCT:.1%} of equity "
            f"${equity:,.2f}). No NEW positions for the rest of the trading "
            f"day; closing/reducing existing positions stays allowed.",
            "block",
        ))
    return out


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
        # Intent is SIDE-AWARE (the pre-fix "any existing row → close"
        # classified BUY-to-ADD to a held long as a close, skipping every
        # opening gate — the last single-leg bypass of the convexity
        # retirement):
        #   SELL — CLOSE iff ANY open row exists (selling a held long, or
        #          SIDE_UNKNOWN failing open so a real close is never
        #          hard-blocked as SELL-to-open); otherwise OPEN (new
        #          short → R_short_option_open_blocked).
        #   BUY  — CLOSE only when the existing open row is a SHORT
        #          (side=SELL buy-back, e.g. a combo's short leg). BUY
        #          against an existing LONG (or SIDE_UNKNOWN, or no row)
        #          is an OPEN: an add-on increases exposure and must run
        #          the status gate, thesis freshness, R5d and sizing.
        existing_side = open_position_side_for(opt_sym)
        if side == "SELL":
            intent = "close" if existing_side is not None else "open"
        else:
            intent = "close" if existing_side == "SELL" else "open"
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
    legs: tuple[dict, ...] | None = None  # per-leg summary for combos (audit)

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
    option_quote: dict | None = None,
) -> GuardDecision:
    """Run the full order gate: parse → intent → thesis freshness →
    short-option-open hard block → R5d liquidity → R1-R7 sizing. DB reads
    plus, for OPENING option orders only, one live snapshot fetch (R5d);
    the caller supplies live equity/cash (each layer has its own broker
    transport). ``option_quote`` lets a caller that already holds a fresh
    snapshot row for the option pass it through instead of triggering a
    second fetch.
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

    # ---- R5d option liquidity (OPENING option orders only) ----
    # Closes are ALWAYS exempt: the guard must never trap an exit, and a
    # wide spread on the way out is a cost to pay, not a reason to hold.
    # SELL-to-open never reaches here (hard-blocked above), so in practice
    # this gates BUY-to-open.
    liq_vs: list[SizingViolation] = []
    if proposed.asset_type == "OPT" and proposed.intent == "open":
        if option_quote is not None:
            quote = option_quote
        else:
            try:
                quote = fetch_option_quote(symbol)
            except Exception:
                quote = None  # belt-and-suspenders; fetch already swallows
        liq_vs = check_option_liquidity(symbol, quote)

    # ---- R8 daily -2R circuit breaker (opens only; closes exempt) ----
    breaker_vs = check_daily_loss_breaker(equity, proposed.intent)

    # ---- sizing (R1-R7) ----
    opens = load_open_positions(sector_map)
    ctx = SizingContext(
        equity=equity,
        cash=cash,
        opens=tuple(opens),
        sector_lookup_available=bool(sector_map),
    )
    vs = liq_vs + breaker_vs + check(ctx, proposed)
    bs = blockers(vs)
    warns = tuple(f"{v.rule}: {v.message}" for v in vs if v.severity == "warn")

    if bs:
        # Distinct reason when ONLY liquidity / only the breaker blocked —
        # hook_audit_log greps for these; mixed violations keep the generic
        # sizing reason.
        liq_rules = {R5D, R5D_UNQUOTABLE}
        r8_rules = {R8, R8_STORE_BLIND}
        if all(v.rule in liq_rules for v in bs):
            reason = "option liquidity gate failed (R5d)"
        elif all(v.rule in r8_rules for v in bs):
            reason = "daily loss circuit breaker tripped (R8)"
        else:
            reason = "sizing rules violated"
        return GuardDecision(
            allowed=False,
            reason=reason,
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


# ---- multi-leg defined-risk combo path (area A) ----
#
# A SEPARATE entry point from evaluate_order. The single-leg
# is_opening_option_short hard block in evaluate_order is UNTOUCHED — a short
# leg is only ever legal inside evaluate_combo, where check_combo proves the
# long leg caps it (defined risk). Closing a spread still uses the single-leg
# close path (BUY-to-close short / SELL-to-close long, both intent=close,
# already allowed).


def build_proposed_combo(
    params: dict, sector_map: dict[str, str]
) -> ProposedCombo | str:
    """Build a ProposedCombo from place_paper_option_combo arguments.

    Returns a string error (caller blocks with it) on unparseable/mismatched
    leg symbols. Both legs must share an underlying; right/expiry/strike
    relationships are validated downstream by check_combo.
    """
    short_sym = str(params.get("short_leg_symbol", ""))
    long_sym = str(params.get("long_leg_symbol", ""))
    sp = parse_option_symbol(short_sym)
    lp = parse_option_symbol(long_sym)
    if not sp:
        return (f"unparseable short_leg_symbol {short_sym!r}; expected "
                "'US.<TICKER><YYMMDD><C|P><strike*1000>'")
    if not lp:
        return (f"unparseable long_leg_symbol {long_sym!r}; expected "
                "'US.<TICKER><YYMMDD><C|P><strike*1000>'")
    s_tkr, s_dte, s_right, s_strike = sp
    l_tkr, l_dte, l_right, l_strike = lp
    if s_tkr != l_tkr:
        return f"combo legs span different underlyings: {s_tkr} vs {l_tkr}"
    try:
        contracts = float(params.get("contracts", 0))
        short_price = float(params.get("short_price", 0))
        long_price = float(params.get("long_price", 0))
    except (TypeError, ValueError):
        return "combo contracts/short_price/long_price must be numeric"

    def _opt_float(key):
        val = params.get(key)
        return float(val) if val is not None else None

    short_leg = ComboLeg(
        option_symbol=short_sym, side="SELL", contracts=contracts,
        price=short_price, right=s_right, strike=s_strike,  # type: ignore[arg-type]
        dte=s_dte, delta=_opt_float("short_delta"),
    )
    long_leg = ComboLeg(
        option_symbol=long_sym, side="BUY", contracts=contracts,
        price=long_price, right=l_right, strike=l_strike,  # type: ignore[arg-type]
        dte=l_dte, delta=_opt_float("long_delta"),
    )
    return ProposedCombo(
        ticker=s_tkr,
        legs=(short_leg, long_leg),
        strategy_label=params.get("strategy_label"),
        sector=sector_map.get(s_tkr.upper()),
        intent="open",
    )


def _combo_leg_summary(combo: ProposedCombo) -> tuple[dict, ...]:
    return tuple(
        {"option_symbol": leg.option_symbol, "side": leg.side,
         "contracts": leg.contracts, "price": leg.price,
         "right": leg.right, "strike": leg.strike, "dte": leg.dte,
         "delta": leg.delta}
        for leg in combo.legs
    )


def evaluate_combo(
    params: dict,
    *,
    equity: float,
    cash: float | None,
    leg_quotes: dict[str, dict] | None = None,
) -> GuardDecision:
    """Gate one atomic two-leg defined-risk vertical: parse → thesis freshness
    on the underlying → check_combo (R5e + R1-R5 sized as ONE position) → R5d
    liquidity on BOTH legs. ``leg_quotes`` (``{option_symbol: snapshot}``) lets
    a caller that already holds fresh snapshots pass them through instead of
    triggering fetches (same spoofing-guard pattern as evaluate_order's
    option_quote).
    """
    short_sym = str(params.get("short_leg_symbol") or "")
    long_sym = str(params.get("long_leg_symbol") or "")
    symbol = f"{short_sym}+{long_sym}"
    sector_map = load_sector_map()
    combo = build_proposed_combo(params, sector_map)
    if isinstance(combo, str):
        return GuardDecision(allowed=False, reason=combo, symbol=symbol)

    legs_summary = _combo_leg_summary(combo)
    proposed_summary = {
        "ticker": combo.ticker, "asset_type": "OPT_COMBO",
        "strategy_label": combo.strategy_label, "sector": combo.sector,
        "contracts": combo.contracts, "net_credit": combo.net_credit,
        "width": combo.width, "max_loss": combo.max_loss, "intent": "open",
    }

    # One fresh thesis on the underlying covers the whole combo.
    ok, thesis_id = has_fresh_open_thesis(combo.ticker)
    if not ok:
        return GuardDecision(
            allowed=False,
            reason=(f"no open thesis for {combo.ticker} in the last "
                    f"{THESIS_FRESHNESS_MIN} minutes"),
            symbol=symbol, ticker=combo.ticker, intent="open",
            proposed=proposed_summary, legs=legs_summary,
        )

    opens = load_open_positions(sector_map)
    ctx = SizingContext(
        equity=equity, cash=cash, opens=tuple(opens),
        sector_lookup_available=bool(sector_map),
    )
    # A combo only ever OPENS, so the R8 daily -2R breaker always applies.
    vs: list[SizingViolation] = list(check_daily_loss_breaker(equity, "open"))
    vs += check_combo(ctx, combo)

    # R5d liquidity on BOTH legs (each must independently pass). A combo only
    # opens, so liquidity always applies; the protective long wing gets the
    # same gate (a thin wing you can't price is a spread you skip).
    # The same snapshots are echoed into proposed["leg_quotes"] (role-keyed
    # {short: {bid, ask}, long: {bid, ask}}) so place_paper_option_combo can
    # pass the ENTRY touch out to the fill-capture hook — _sync_combo_entry
    # clamps the journaled net credit to "fill no better than the touch"
    # (M1-0.2 cost honesty) against exactly these quotes.
    entry_leg_quotes: dict[str, dict] = {}
    for leg in combo.legs:
        if leg_quotes is not None:
            q = leg_quotes.get(leg.option_symbol)
        else:
            try:
                q = fetch_option_quote(leg.option_symbol)
            except Exception:
                q = None
        if isinstance(q, dict):
            role = "short" if leg.side == "SELL" else "long"
            entry_leg_quotes[role] = {
                "bid": _pos_float(q.get("bid_price"), q.get("bid")),
                "ask": _pos_float(q.get("ask_price"), q.get("ask")),
            }
        vs += check_option_liquidity(leg.option_symbol, q)
    proposed_summary["leg_quotes"] = entry_leg_quotes

    bs = blockers(vs)
    warns = tuple(f"{v.rule}: {v.message}" for v in vs if v.severity == "warn")

    if bs:
        liq_rules = {R5D, R5D_UNQUOTABLE}
        r8_rules = {R8, R8_STORE_BLIND}
        if all(v.rule in liq_rules for v in bs):
            reason = "option liquidity gate failed (R5d)"
        elif all(v.rule in r8_rules for v in bs):
            reason = "daily loss circuit breaker tripped (R8)"
        elif any(v.rule == R5E for v in bs):
            reason = "combo defined-risk gate failed (R5e)"
        else:
            reason = "sizing rules violated"
        return GuardDecision(
            allowed=False, reason=reason, symbol=symbol, ticker=combo.ticker,
            intent="open", thesis_id=thesis_id, violations=tuple(vs),
            warns=warns, equity=equity, open_count=len(opens),
            proposed=proposed_summary, legs=legs_summary,
        )

    return GuardDecision(
        allowed=True, reason="ok", symbol=symbol, ticker=combo.ticker,
        intent="open", thesis_id=thesis_id, violations=tuple(vs), warns=warns,
        equity=equity, open_count=len(opens),
        proposed=proposed_summary, legs=legs_summary,
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
