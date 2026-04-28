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


def _get_latest_mid(symbol: str) -> float | None:
    """Best-effort latest mid price.  Returns None if quote layer is down."""
    try:
        from trading_agent.mcp_servers.moomoo.server import get_quote
        q = get_quote(symbol)
    except Exception as e:
        log.warning("get_quote(%s) failed: %s", symbol, e)
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
    return last


def _R_per_unit(entry: float, stop: float | None) -> float:
    if stop is not None and abs(entry - stop) > 1e-9:
        return abs(entry - stop)
    return max(1e-9, abs(entry) * IMPLICIT_STOP_FRAC)


def update_excursions_once(now: datetime | None = None) -> list[ExcursionUpdate]:
    """Walk all OPEN journal_trades rows and update mae/mfe.

    Returns the list of updates applied (for logging in agent_events).
    Never raises — DB or quote failures degrade to skipping that row.
    """
    out: list[ExcursionUpdate] = []
    now = now or datetime.now(timezone.utc)
    try:
        with cursor() as cur:
            cur.execute(
                """
                SELECT id, symbol, entry_price, stop, mae_so_far, mfe_so_far
                FROM journal_trades
                WHERE outcome = 'OPEN'
                """,
            )
            rows = cur.fetchall()
    except Exception as e:
        log.warning("update_excursions_once: db read failed (%s)", e)
        return out

    for tid, symbol, entry, stop, mae, mfe in rows:
        entry_f = _safe_float(entry)
        if entry_f is None:
            continue
        mid = _get_latest_mid(str(symbol))
        if mid is None:
            continue
        R_unit = _R_per_unit(entry_f, _safe_float(stop))
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
