"""Phase 2.7.5 — Intraday MAE / MFE updater.

Called from ``intraday_monitor_graph`` after ``refresh_quotes_and_greeks``.
Walks every OPEN journal_trades row, computes the current realized-R at the
latest mid, and updates the running (mae_so_far, mfe_so_far) extremes.

Quote source: same Moomoo OpenD path the rest of the system uses (so this
module degrades gracefully when OpenD is unavailable — see
``mcp_servers.moomoo.server.get_quote``).

Closing semantics: ``outcome.compute_outcome`` reads the persisted
``mae_so_far`` / ``mfe_so_far`` and copies them into
``trade_outcome_features.mae`` / ``.mfe`` at close time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)

# When a trade has no explicit stop, fall back to the sizing.py implicit
# fraction so MAE / MFE are still meaningful in R-units.
IMPLICIT_STOP_FRAC = 0.05


@dataclass
class ExcursionUpdate:
    trade_id: int
    symbol: str
    realized_R_now: float
    mae_so_far: float
    mfe_so_far: float


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


_STALE_QUOTE_MAX_AGE_S = 300.0  # 5 minutes — drop fallback quotes older than this


def _quote_age_seconds(row: dict) -> float | None:
    """Best-effort age of a moomoo quote row.  Tries common timestamp fields;
    returns None when no timestamp is available (caller may then accept the
    quote conservatively or skip it)."""
    for key in ("update_time", "data_time", "ts", "timestamp"):
        v = row.get(key)
        if not v:
            continue
        try:
            from datetime import datetime, timezone
            if isinstance(v, (int, float)):
                ts = datetime.fromtimestamp(float(v), tz=timezone.utc)
            else:
                # Most moomoo timestamps are 'YYYY-MM-DD HH:MM:SS' local;
                # fall back to ISO parsing.
                s = str(v).replace("Z", "+00:00")
                ts = datetime.fromisoformat(s) if "T" in s else datetime.fromisoformat(
                    s.replace(" ", "T")
                )
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - ts).total_seconds()
        except (ValueError, TypeError):
            continue
    return None


_QUOTE_FETCH_TIMEOUT_S = 5.0


def _fetch_quote_with_timeout(symbol: str, timeout_s: float = _QUOTE_FETCH_TIMEOUT_S) -> dict | None:
    """Wrap get_quote in a thread + timeout.  moomoo's get_market_snapshot
    can hang on option contracts that aren't subscribed (no auto-subscribe
    on the free OpenD path); a hard timeout is the only safe defense.
    """
    import concurrent.futures
    try:
        from trading_agent.mcp_servers.moomoo.server import get_quote
    except Exception as e:
        log.warning("excursion: get_quote import failed: %s", e)
        return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        # NOTE: get_quote signature is `symbols: list[str]`, not `str`.
        future = ex.submit(get_quote, [symbol])
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            log.warning("excursion: get_quote(%s) timed out after %.1fs", symbol, timeout_s)
            return None
        except Exception as e:
            log.warning("excursion: get_quote(%s) failed: %s", symbol, e)
            return None


def _get_latest_mid(symbol: str) -> float | None:
    """Best-effort latest mid price.  Returns None when:
      * quote layer is down or hangs (5s timeout)
      * no rows
      * neither bid/ask nor last_price is positive
      * we fell back to last_price AND the quote age is > 5 minutes
        (avoids using stale prints for expired-or-halted contracts)
    """
    q = _fetch_quote_with_timeout(symbol)
    if q is None:
        return None
    rows = (q or {}).get("rows") or []
    if not rows:
        return None
    row = rows[0]
    bid = _safe_float(row.get("bid_price") or row.get("bid"))
    ask = _safe_float(row.get("ask_price") or row.get("ask"))
    if bid and ask and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    last = _safe_float(row.get("last_price") or row.get("cur_price") or row.get("price"))
    if last is None or last <= 0:
        return None
    age = _quote_age_seconds(row)
    if age is not None and age > _STALE_QUOTE_MAX_AGE_S:
        log.warning(
            "_get_latest_mid(%s) dropping stale fallback quote: age=%.0fs",
            symbol, age,
        )
        return None
    return last


def _R_per_unit(entry: float, stop: float | None) -> float:
    if stop is not None and abs(entry - stop) > 1e-9:
        return abs(entry - stop)
    return max(1e-9, abs(entry) * IMPLICIT_STOP_FRAC)


_TOTAL_UPDATE_BUDGET_S = 60.0  # whole node must finish well inside the 15-min timer


def update_excursions_once(now: datetime | None = None) -> list[ExcursionUpdate]:
    """Walk all OPEN journal_trades rows and update mae/mfe.

    Returns the list of updates applied (for logging in agent_events).
    Never raises — DB or quote failures degrade to skipping that row.

    Total wall-clock budget: 60 s.  Once exceeded, remaining open positions
    are skipped this tick — the next intraday timer fire (15 min later) will
    pick them back up.  This bound exists because moomoo OpenD's
    get_market_snapshot can hang on un-subscribed option contracts; without
    a hard ceiling here, a portfolio of N positions that all hang would
    prevent the rest of intraday_monitor from running.
    """
    import time
    out: list[ExcursionUpdate] = []
    now = now or datetime.now(timezone.utc)
    deadline = time.monotonic() + _TOTAL_UPDATE_BUDGET_S
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT id, symbol, entry_price, stop, mae_so_far, mfe_so_far,
                       broker_fill_json->'combo' AS combo
                FROM journal_trades
                WHERE outcome = 'OPEN'
                """,
            )
            rows = cur.fetchall()
    except Exception as e:
        log.warning("update_excursions_once: db read failed (%s)", e)
        return out

    from trading_agent.combo import combo_meta

    for tid, symbol, entry, stop, mae, mfe, combo_raw in rows:
        if time.monotonic() > deadline:
            log.warning(
                "update_excursions_once: budget %.0fs exhausted; %d positions skipped",
                _TOTAL_UPDATE_BUDGET_S, len(rows) - len(out),
            )
            break
        entry_f = _safe_float(entry)
        if entry_f is None:
            continue
        meta = combo_meta(combo_raw)
        if meta is not None:
            # Combo row: entry_price is the NET CREDIT across both legs, so
            # the mark must be net-of-legs too (short mid − long mid) or the
            # mae/mfe fields silently change units (M1-0.1: excursions stay
            # in units of the net credit). Either leg unquotable, or a
            # non-positive spread mid (crossed junk) → skip this tick.
            short_mid = _get_latest_mid(meta.short_leg)
            long_mid = _get_latest_mid(meta.long_leg)
            if short_mid is None or long_mid is None:
                continue
            mid = short_mid - long_mid
            if mid <= 0:
                continue
        else:
            mid = _get_latest_mid(str(symbol))
        if mid is None:
            continue
        R_unit = _R_per_unit(entry_f, _safe_float(stop))
        if meta is not None:
            # Combo rows are SHORT the spread: entry is the net CREDIT and a
            # FALLING spread value is favorable. LONG polarity here would
            # record every winning combo as a large negative realized_R
            # (MAE/MFE semantically swapped for all combo learning data).
            realized_R = (entry_f - mid) / R_unit
        else:
            realized_R = (mid - entry_f) / R_unit
        prev_mae = float(mae) if mae is not None else realized_R
        prev_mfe = float(mfe) if mfe is not None else realized_R
        new_mae = min(prev_mae, realized_R)
        new_mfe = max(prev_mfe, realized_R)
        try:
            with cursor() as cur:
                cur.execute(
                    """
                    UPDATE journal_trades
                    SET mae_so_far = %s, mfe_so_far = %s,
                        last_quote_at = %s, last_quote_price = %s
                    WHERE id = %s
                    """,
                    (new_mae, new_mfe, now, mid, int(tid)),
                )
        except Exception as e:
            log.warning("update_excursions_once: write failed for id=%s: %s", tid, e)
            continue
        out.append(ExcursionUpdate(
            trade_id=int(tid),
            symbol=str(symbol),
            realized_R_now=realized_R,
            mae_so_far=new_mae,
            mfe_so_far=new_mfe,
        ))
    return out


def read_extremes(trade_id: int) -> tuple[float | None, float | None]:
    """Pull (mae_so_far, mfe_so_far) for a closed trade.  Used by
    outcome.compute_outcome at close time."""
    try:
        with cursor() as cur:
            cur.execute(
                "SELECT mae_so_far, mfe_so_far FROM journal_trades WHERE id = %s",
                (trade_id,),
            )
            row = cur.fetchone()
    except Exception as e:
        log.warning("read_extremes(%s) failed: %s", trade_id, e)
        return (None, None)
    if row is None:
        return (None, None)
    mae, mfe = row
    return (
        float(mae) if mae is not None else None,
        float(mfe) if mfe is not None else None,
    )


__all__ = [
    "ExcursionUpdate",
    "IMPLICIT_STOP_FRAC",
    "read_extremes",
    "update_excursions_once",
]
