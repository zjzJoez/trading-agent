"""Daily EOD option-chain snapshot → Postgres option_chain_snapshots.

Moomoo's free tier has no historical option quotes — once a session ends,
the chain the agent saw is gone. This job runs after US close (systemd
timer, 21:15 UTC weekdays, before the 21:30 eod_review) and persists a
near-the-money slice of every chain the system could plausibly trade:
the CORE_UNDERLYINGS index set (SPY/QQQ/IWM — sleeve 1 needs those chains
daily, with a wider ±15%-of-spot strike window) UNION today's watchlist
UNION the underlyings of OPEN journal trades UNION the underlyings of
today's SHADOW_ONLY proposals (the retired-strategy
counterfactual book must have its contracts priced). That makes
the strategies actually traded backtestable someday — "what would the
other strike/expiry have cost" stops being unanswerable.

Per underlying: spot quote → expiries in the 7-90 DTE band (max 4, spread
across the band) → ~12 strikes each side of spot per expiry → batch quote
for bid/ask/greeks/iv. One underlying failing (halted name, missing chain)
must not kill the run; rows commit per-underlying so a mid-run crash keeps
everything captured so far.

Idempotent: UNIQUE (snapshot_date, code) + ON CONFLICT DO UPDATE, so a
manual re-run after a partial failure refreshes rows instead of duplicating.

Usage:
    python -m trading_agent.jobs.cache_option_chains [--dry-run] \
        [--underlyings AAPL,QQQ]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone

from trading_agent import events
from trading_agent.order_guard import ticker_from_any_symbol
from trading_agent.store import postgres

log = logging.getLogger(__name__)

# DTE band to snapshot. Wider than the R5 trade window (14-60) on purpose:
# the snapshot must also price the expiries the trader *considered* (chain
# prefetch shows 7-45) and the tail a position rolls into near 60 DTE.
DTE_MIN, DTE_MAX = 7, 90
MAX_EXPIRIES_PER_UNDERLYING = 4
STRIKES_EACH_SIDE = 12

# Sleeve 1 (credit_vertical_index_30_45, docs/REVIVAL_PLAN_2026-07-20.md M1-0)
# trades SPY/QQQ/IWM verticals — these indexes must be in EVERY nightly
# snapshot regardless of what the rotating watchlist holds (SPY had 4 days
# of history, QQQ 1, IWM 4 when this gap was found). The DTE band above
# already spans the 30-45 sleeve window and the ~60 DTE roll tail; strikes,
# however, need ±15% of spot (a 30-delta short leg + further-OTM long leg
# sit well outside the nearest-12 band on a $600 index), so the core set
# gets a percent-of-spot strike window instead of nearest-N-only.
CORE_UNDERLYINGS = ("SPY", "QQQ", "IWM")
CORE_STRIKE_PCT = 0.15

# moomoo's get_market_snapshot caps codes per request (400 on the free
# tier); the core strike window can push one expiry past that, so batch.
_QUOTE_BATCH_MAX = 300

_INSERT_SQL = """
    INSERT INTO option_chain_snapshots
        (snapshot_date, underlying, spot, code, expiry, strike, option_type,
         bid, ask, last, volume, open_interest, iv, delta, gamma, vega, theta)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (snapshot_date, code) DO UPDATE SET
        spot = EXCLUDED.spot,
        bid = EXCLUDED.bid,
        ask = EXCLUDED.ask,
        last = EXCLUDED.last,
        volume = EXCLUDED.volume,
        open_interest = EXCLUDED.open_interest,
        iv = EXCLUDED.iv,
        delta = EXCLUDED.delta,
        gamma = EXCLUDED.gamma,
        vega = EXCLUDED.vega,
        theta = EXCLUDED.theta,
        captured_at = NOW()
"""


def _market_fns():
    """Lazy import of the moomoo quote helpers — keeps this module importable
    (and testable) without the broker SDK until a real run needs OpenD."""
    from trading_agent.mcp_servers.moomoo.server import (
        get_option_chain,
        get_quote,
        list_option_expiries,
    )
    return get_quote, list_option_expiries, get_option_chain


def _f(v) -> float | None:
    """Tolerant numeric: None for unparseable AND for NaN. Moomoo's snapshot
    DataFrame pads missing greeks/iv with NaN floats, and psycopg refuses
    NaN for NUMERIC columns — one NaN row must not abort the batch."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _watchlist_underlyings() -> list[str]:
    """Same source priority as the premarket job (WATCHLIST_TICKERS env →
    Postgres watchlist_members via refresh_dynamic_watchlist → static
    fallback) — reuse its resolver so the two can never drift."""
    from trading_agent.jobs.premarket_watchlist import _get_watchlist
    return _get_watchlist()


def _open_trade_underlyings() -> list[str]:
    """Underlyings of OPEN Postgres journal trades. A position can outlive
    its watchlist membership (rotating-tier eviction), and an open position
    is exactly the chain we most need priced for post-mortems.

    None from the store means UNKNOWN (PG unreachable) — for a cache job
    that's safe to treat as empty: we lose one day of chain history for
    those names, we don't make a trading decision."""
    rows = postgres.get_open_journal_trades()
    if rows is None:
        log.warning("[cache_option_chains] journal_trades unreadable — "
                    "caching watchlist underlyings only")
        return []
    out: list[str] = []
    for r in rows:
        t = ticker_from_any_symbol(r.get("symbol") or "")
        if t:
            out.append(t.upper())
    return out


def _shadow_underlyings() -> list[str]:
    """Underlyings of TODAY's SHADOW_ONLY shadow_proposals rows — the
    retired-strategy counterfactual book (convexity_long_premium,
    docs/REVIVAL_PLAN_2026-07-20.md). A shadow proposal's underlying can be
    absent from both the watchlist and the open-trade set (nothing was ever
    opened), and counterfactual replay needs the proposed contract itself
    priced in tonight's snapshot, not a neighbour's. Best-effort like
    _open_trade_underlyings: an unreachable store means one lost day of
    coverage, not a trading decision."""
    out: list[str] = []
    try:
        with postgres.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT ticker, symbol FROM shadow_proposals "
                "WHERE final_action = 'SHADOW_ONLY' "
                "  AND proposal_ts::date = CURRENT_DATE"
            )
            for r in cur.fetchall() or []:
                if isinstance(r, dict):
                    tick, sym = r.get("ticker"), r.get("symbol")
                else:
                    tick, sym = r[0], r[1]
                t = ((tick or "").strip().upper()
                     or (ticker_from_any_symbol(sym or "") or "").upper())
                if t:
                    out.append(t)
    except Exception as e:  # noqa: BLE001 — cache job, never fatal
        log.warning("[cache_option_chains] shadow_proposals unreadable — "
                    "skipping shadow-book underlyings: %s", e)
        return []
    return out


def build_underlying_set(override: str | None = None) -> list[str]:
    """Sorted, deduped underlying tickers for today's snapshot.

    CORE_UNDERLYINGS are always present in the sourced set — sleeve 1 needs
    SPY/QQQ/IWM chain history every day even when the rotating watchlist,
    the journal, and the shadow book contain none of them. An explicit
    --underlyings override stays a pure override (a targeted manual re-run
    must be able to snapshot exactly the names asked for).
    """
    if override:
        return sorted({t.strip().upper() for t in override.split(",") if t.strip()})
    return sorted(set(CORE_UNDERLYINGS)
                  | set(_watchlist_underlyings())
                  | set(_open_trade_underlyings())
                  | set(_shadow_underlyings()))


def _pick_expiries(expiry_rows: list[dict], today: date) -> list[str]:
    """In-window expiries, at most MAX_EXPIRIES_PER_UNDERLYING, spread evenly
    across the DTE band (nearest + farthest always kept) rather than the
    first N — four weeklies in a row would leave the 30-90 DTE region,
    where swing entries actually live, completely unpriced."""
    in_window: list[str] = []
    for row in expiry_rows:
        st = str(row.get("strike_time") or "")[:10]
        try:
            dte = (date.fromisoformat(st) - today).days
        except ValueError:
            continue
        if DTE_MIN <= dte <= DTE_MAX:
            in_window.append(st)
    in_window.sort()
    n = len(in_window)
    if n <= MAX_EXPIRIES_PER_UNDERLYING:
        return in_window
    k = MAX_EXPIRIES_PER_UNDERLYING
    idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return [in_window[i] for i in idx]


def snapshot_underlying(underlying: str, today: date) -> list[tuple]:
    """Fetch one underlying's chain slice; return rows shaped for _INSERT_SQL.

    Raises on total failure (no spot, expiry list unreachable) so the caller
    counts it as a failed underlying; per-expiry hiccups are logged and
    skipped — a partial snapshot beats none.
    """
    get_quote, list_option_expiries, get_option_chain = _market_fns()
    code = f"US.{underlying}"

    spot_rows = get_quote([code]).get("rows") or []
    spot = _f(spot_rows[0].get("last_price")) if spot_rows else None
    if not spot or spot <= 0:
        raise RuntimeError(f"no spot price for {code}")

    expiries = _pick_expiries(list_option_expiries(code).get("rows") or [], today)

    rows: list[tuple] = []
    for exp in expiries:
        try:
            chain = get_option_chain(code, exp).get("rows") or []
        except Exception as e:  # noqa: BLE001 — one bad expiry must not sink the name
            log.warning("[cache_option_chains] %s %s chain fetch failed: %s",
                        underlying, exp, e)
            continue
        chain = [c for c in chain if c.get("code") and _f(c.get("strike_price"))]
        # Nearest STRIKES_EACH_SIDE strikes per side — the band that trades.
        # Core (sleeve-1 index) names instead take every strike within
        # ±CORE_STRIKE_PCT of spot, floored at the nearest-N band so a
        # coarse chain can never leave a core name with LESS than default.
        is_core = underlying in CORE_UNDERLYINGS
        band = CORE_STRIKE_PCT * spot
        by_code: dict[str, dict] = {}
        keep: list[str] = []
        for side in ("CALL", "PUT"):
            side_rows = [c for c in chain if (c.get("option_type") or c.get("type")) == side]
            side_rows.sort(key=lambda c: abs(float(c["strike_price"]) - spot))
            selected = side_rows[:STRIKES_EACH_SIDE]
            if is_core:
                # side_rows is distance-sorted, so the pct window is a
                # prefix; take the longer of the two selections.
                in_band = [c for c in side_rows
                           if abs(float(c["strike_price"]) - spot) <= band]
                if len(in_band) > len(selected):
                    selected = in_band
            for c in selected:
                keep.append(c["code"])
                by_code[c["code"]] = c
        if not keep:
            continue
        try:
            keep.sort()
            quotes: list[dict] = []
            for i in range(0, len(keep), _QUOTE_BATCH_MAX):
                quotes.extend(
                    get_quote(keep[i:i + _QUOTE_BATCH_MAX]).get("rows") or [])
        except Exception as e:  # noqa: BLE001
            log.warning("[cache_option_chains] %s %s option quotes failed: %s",
                        underlying, exp, e)
            continue
        for q in quotes:
            qcode = q.get("code") or ""
            ref = by_code.get(qcode, {})
            # Quote rows carry option_* fields; fall back to the chain row
            # (strike/type are contract facts, always present there).
            strike = _f(q.get("option_strike_price")) or _f(ref.get("strike_price"))
            opt_type = q.get("option_type") or ref.get("option_type") or ref.get("type")
            iv_raw = _f(q.get("option_implied_volatility"))
            rows.append((
                today, underlying, spot, qcode,
                date.fromisoformat(exp), strike, opt_type,
                _f(q.get("bid_price") or q.get("bid")),
                _f(q.get("ask_price") or q.get("ask")),
                _f(q.get("last_price")),
                _f(q.get("volume")),
                _f(q.get("option_open_interest")),
                # Broker reports IV in percent; store the fraction (matches
                # the trade_nodes prefetch convention and the column comment).
                (iv_raw / 100.0) if iv_raw is not None else None,
                _f(q.get("option_delta")),
                _f(q.get("option_gamma")),
                _f(q.get("option_vega")),
                _f(q.get("option_theta")),
            ))
    return rows


def _insert_rows(rows: list[tuple]) -> None:
    """One cursor() context per underlying = one commit per underlying, so a
    crash on underlying N keeps the N-1 already snapped."""
    with postgres.cursor() as cur:
        for r in rows:
            cur.execute(_INSERT_SQL, r)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Cache EOD option-chain snapshots into Postgres")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and report what would be written; no DB writes")
    ap.add_argument("--underlyings", default=None,
                    help="comma-separated override, e.g. AAPL,QQQ (skips "
                         "watchlist+journal sourcing)")
    args = ap.parse_args(argv)

    # The timer fires 21:15 UTC; the US trading date equals the UTC date at
    # that hour year-round (ET = UTC-4/-5), so no tz machinery needed.
    today = datetime.now(timezone.utc).date()
    run_id = events.new_run_id("chaincache")

    underlyings = build_underlying_set(args.underlyings)
    per_underlying: dict[str, int] = {}
    failures: dict[str, str] = {}
    total_rows = 0

    try:
        for u in underlyings:
            try:
                rows = snapshot_underlying(u, today)
                if not args.dry_run and rows:
                    _insert_rows(rows)
                per_underlying[u] = len(rows)
                total_rows += len(rows)
                log.info("[cache_option_chains] %s: %d contracts", u, len(rows))
            except Exception as e:  # noqa: BLE001 — per-underlying isolation
                failures[u] = repr(e)
                log.warning("[cache_option_chains] %s failed: %s", u, e)
    finally:
        # The moomoo SDK's quote context starts NON-daemon callback threads;
        # without an explicit shutdown this systemd oneshot hangs at exit and
        # gets killed/failed by the unit every night. Same pattern as the
        # PreToolUse hook's cleanup — only if the module actually loaded.
        srv = sys.modules.get("trading_agent.mcp_servers.moomoo.server")
        if srv is not None:
            try:
                srv.shutdown()
            except Exception as e:  # noqa: BLE001 — never mask the run result
                log.warning("[cache_option_chains] moomoo shutdown failed: %s", e)

    all_failed = bool(underlyings) and len(failures) == len(underlyings)
    if not args.dry_run:
        # One audit row per run. Every underlying failing means OpenD itself
        # is down/unreachable — that's a sev-2 (operator should look), a few
        # individual misses are a warn.
        severity = (events.SEV_ERROR if all_failed
                    else events.SEV_WARN if failures else events.SEV_INFO)
        events.emit(
            run_id=run_id, trigger="chain_cache", agent="cache_option_chains",
            event_type="chain_cache_complete",
            payload={
                "snapshot_date": today.isoformat(),
                "underlyings": len(underlyings),
                "rows": total_rows,
                "per_underlying": per_underlying,
                "failures": failures,
                "all_failed": all_failed,
            },
            severity=severity,
        )

    summary = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "job": "cache_option_chains",
        "run_id": run_id,
        "dry_run": args.dry_run,
        "snapshot_date": today.isoformat(),
        "underlyings": underlyings,
        "rows": total_rows,
        "per_underlying": per_underlying,
        "failures": failures,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 1 if all_failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
