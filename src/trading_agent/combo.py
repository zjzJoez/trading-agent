"""Canonical combo (defined-risk vertical) detection — ONE module, reused
everywhere (M1-0.1 combo-close lane).

A defined-risk vertical is journaled as ONE row keyed by the short leg's
OCC symbol with ``side='SELL'`` and ``entry_price=net_credit``. The legs
themselves live in a canonical JSON payload:

  * SQLite: the latest ``market_snapshots`` payload with ``combo: true``
    (written by hooks/posttool_fill_capture's combo branch).
  * Postgres: ``journal_trades.broker_fill_json->'combo'`` (same shape,
    written by the same hook's dual-write).

Shape (both stores):
    {"combo": true, "short_leg": ..., "long_leg": ..., "net_credit": ...,
     "width": ..., "max_loss": ..., "contracts": ..., "broker_order_ids": [...]}

Fail-closed contract: a malformed payload parses to ``None`` — the row then
degrades to "unknown" (HOLD / alert), NEVER to single-leg management, which
would orphan the protective long wing.

Import-light by design (stdlib + lazy trading_agent.db / store.postgres
imports inside functions) so order_guard can use it without slowing the
PreToolUse hook's cold start.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComboMeta:
    """Parsed canonical combo payload for one journaled vertical.

    ``short_bid/short_ask/long_bid/long_ask`` are the per-leg ENTRY touch
    captured from the R5d guard snapshots at placement time (payload key
    ``leg_quotes``); None when the payload predates M1-0.2 or a side had no
    quote. ``_sync_combo_entry`` uses them to clamp the journaled net
    credit to "fill no better than the touch"."""

    short_leg: str
    long_leg: str
    net_credit: float
    width: float | None
    max_loss: float | None
    contracts: float
    broker_order_ids: tuple[str, ...]
    short_bid: float | None = None
    short_ask: float | None = None
    long_bid: float | None = None
    long_ask: float | None = None


def _opt_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def combo_meta(payload: Any) -> ComboMeta | None:
    """Parse a canonical combo payload dict (or JSON string) → ComboMeta.

    Returns None unless ``combo`` is truthy AND short_leg AND long_leg AND
    net_credit all parse. Malformed ⇒ None (fail-closed: the row degrades
    to "unknown", never to single-leg management).
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return None
    if not isinstance(payload, dict) or not payload.get("combo"):
        return None
    short_leg = str(payload.get("short_leg") or "").strip()
    long_leg = str(payload.get("long_leg") or "").strip()
    net_credit = _opt_float(payload.get("net_credit"))
    if not short_leg or not long_leg or net_credit is None:
        return None
    raw_ids = payload.get("broker_order_ids")
    if isinstance(raw_ids, (list, tuple)):
        order_ids = tuple(str(x) for x in raw_ids if x is not None)
    else:
        order_ids = ()

    def _leg_quote(role: str, side: str) -> float | None:
        lq = payload.get("leg_quotes")
        entry = lq.get(role) if isinstance(lq, dict) else None
        v = _opt_float(entry.get(side)) if isinstance(entry, dict) else None
        # 0/negative means "no quote on that side" — same convention as
        # order_guard._pos_float.
        return v if v is not None and v > 0 else None

    return ComboMeta(
        short_leg=short_leg,
        long_leg=long_leg,
        net_credit=net_credit,
        width=_opt_float(payload.get("width")),
        max_loss=_opt_float(payload.get("max_loss")),
        contracts=_opt_float(payload.get("contracts")) or 0.0,
        broker_order_ids=order_ids,
        short_bid=_leg_quote("short", "bid"),
        short_ask=_leg_quote("short", "ask"),
        long_bid=_leg_quote("long", "bid"),
        long_ask=_leg_quote("long", "ask"),
    )


def sqlite_combo_payloads(trade_ids: list[int]) -> dict[int, dict]:
    """Latest market_snapshots payload with ``combo: true`` per trade_id.

    Canonical lift of jobs/reconcile_order_guard's private
    ``_combo_snapshots_sqlite`` (that job re-imports this now). Best-effort:
    unreadable payloads are skipped — the row then reconciles as a plain
    single-leg and any drift surfaces as a finding (alert, never mask).
    """
    if not trade_ids:
        return {}
    from trading_agent.db import connection
    placeholders = ",".join("?" for _ in trade_ids)
    with connection() as conn:
        rows = conn.execute(
            "SELECT trade_id, payload FROM market_snapshots "
            f"WHERE trade_id IN ({placeholders}) ORDER BY id",
            trade_ids,
        ).fetchall()
    out: dict[int, dict] = {}
    for r in rows:
        try:
            payload = json.loads(r["payload"])
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("combo"):
            out[int(r["trade_id"])] = payload  # ORDER BY id → latest wins
    return out


def open_combo_long_legs() -> set[str]:
    """Long-leg symbols of every OPEN combo row, both stores, best-effort.

    A SELL on one of these symbols is the vertical's wing being CLOSED, not
    a new short being opened — order_guard.has_existing_open_for consults
    this so guard-mediated long-leg closes classify as intent='close'.

    Errors ⇒ that store contributes nothing (callers decide fail direction).
    """
    out: set[str] = set()
    # SQLite: OPEN trades × latest combo snapshot payloads.
    try:
        from trading_agent.db import connection
        with connection() as conn:
            ids = [int(r[0]) for r in conn.execute(
                "SELECT id FROM trades WHERE outcome = 'OPEN'").fetchall()]
        for payload in sqlite_combo_payloads(ids).values():
            meta = combo_meta(payload)
            if meta is not None:
                out.add(meta.long_leg)
    except Exception as e:
        log.debug("open_combo_long_legs: sqlite read failed: %s", e)
    # Postgres: broker_fill_json->'combo'->>'long_leg' on OPEN rows.
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                "SELECT broker_fill_json->'combo' FROM journal_trades "
                "WHERE outcome = 'OPEN' AND broker_fill_json ? 'combo'",
            )
            for (payload,) in cur.fetchall():
                meta = combo_meta(payload)
                if meta is not None:
                    out.add(meta.long_leg)
    except Exception as e:
        log.debug("open_combo_long_legs: postgres read failed: %s", e)
    return out


__all__ = [
    "ComboMeta",
    "combo_meta",
    "open_combo_long_legs",
    "sqlite_combo_payloads",
]
