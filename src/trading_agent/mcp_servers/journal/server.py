"""journal-mcp: MCP server for the trade journal + self-learning loop.

This is the core of the self-learning loop. Every trade has a thesis; every
closed trade feeds a post-mortem; every post-mortem emits lessons that are
embedded and retrieved on the next similar setup.

Tools fall into three layers:
  - write: record_thesis, record_fill, record_virtual_fill, append_note,
    close_thesis
  - read: list_open_theses, get_open_positions_with_thesis
  - retrieval: search_past_trades (vector + lexical), search_notes,
    generate_post_mortem_prompt

Design invariants:
- All writes go through the shared db.connect() so sqlite-vec is loaded
  uniformly. Embeddings are 384-d MiniLM vectors, inserted into notes_vec
  via struct.pack (sqlite-vec's FLOAT[384] wire format).
- Embedder is loaded lazily once per process — first call pays the model
  download + warm-up cost (~15s cold, ~2s warm).
- search_past_trades does a hybrid retrieval: vector recall via notes_vec
  plus a cheap LIKE on trades.strategy_label/symbol. Results are merged.
- record_thesis is a plain INSERT — each call creates a new row. Callers
  that need idempotency must dedupe via list_open_theses first.
"""
from __future__ import annotations

import json
import logging
import struct
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from trading_agent.config import CONFIG
from trading_agent.db import connection
from trading_agent.strategy_specs import spec_for_label

log = logging.getLogger(__name__)

# Lazy imports for heavyweight deps. SentenceTransformer load is ~500MB,
# so we only import when the first embedding is needed.
_embedder = None
_embedder_lock = threading.Lock()


mcp = FastMCP("journal-mcp")


# --------- helpers ---------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _embedder_singleton():
    global _embedder
    if _embedder is not None:
        return _embedder
    with _embedder_lock:
        if _embedder is not None:
            return _embedder
        from sentence_transformers import SentenceTransformer  # deferred import
        _embedder = SentenceTransformer(CONFIG.embedding_model)
        return _embedder


def _embed(text: str) -> bytes:
    """Encode text → 384-d float32 → sqlite-vec wire bytes."""
    model = _embedder_singleton()
    vec = model.encode([text], normalize_embeddings=True)[0]
    if len(vec) != 384:
        raise RuntimeError(
            f"Embedding dim {len(vec)} != 384; vec0 table won't accept it."
        )
    # sqlite-vec reads FLOAT[N] as packed little-endian float32.
    return struct.pack(f"{len(vec)}f", *[float(x) for x in vec])


def _rowdicts(rows) -> list[dict]:
    return [dict(r) for r in rows]


# --------- writes ---------

@mcp.tool()
def record_thesis(
    ticker: str,
    direction: Literal[
        "LONG", "SHORT", "LONG_CALL", "LONG_PUT", "SHORT_CALL", "SHORT_PUT",
        "VERTICAL_CALL_DEBIT", "VERTICAL_PUT_DEBIT",
    ],
    thesis_text: str,
    invalidation: str,
    timeframe: str | None = None,
    expected_return_pct: float | None = None,
    max_loss_pct: float | None = None,
) -> dict:
    """Record a trade thesis. A thesis must exist before any order is placed —
    the PreToolUse hook enforces this gate.

    `ticker` is the bare symbol (AAPL), not the Moomoo-format code (US.AAPL);
    the hook normalizes when comparing against place_order inputs.
    `invalidation` is the condition that would make you close / flip.
    Concrete triggers (price levels, earnings beats/misses, macro prints)
    are strongly preferred — a vague invalidation is a useless one.

    Returns the thesis_id to pass into place_paper_order.
    """
    with connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO theses (created_at, ticker, direction, thesis_text,
                invalidation, timeframe, expected_return_pct, max_loss_pct, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')
            """,
            (_now(), ticker.upper(), direction, thesis_text, invalidation,
             timeframe, expected_return_pct, max_loss_pct),
        )
        thesis_id = cur.lastrowid
        conn.commit()
    return {"thesis_id": thesis_id, "ticker": ticker.upper(), "status": "open"}


@mcp.tool()
def close_thesis(thesis_id: int, status: Literal["triggered", "void"], note: str = "") -> dict:
    """Mark a thesis as triggered (invalidation hit) or void (cancelled before action).

    Use this when your thesis is no longer live but you haven't yet closed
    the corresponding position — it prevents the hook from letting new
    sibling orders through under a stale thesis.
    """
    with connection() as conn:
        conn.execute(
            "UPDATE theses SET status = ? WHERE id = ?",
            (status, thesis_id),
        )
        if note:
            conn.execute(
                "INSERT INTO notes (created_at, topic, text, tags, source) "
                "VALUES (?, ?, ?, ?, 'manual')",
                (_now(), f"thesis-{thesis_id}-close", note, "close_thesis"),
            )
        conn.commit()
    return {"thesis_id": thesis_id, "status": status}


@mcp.tool()
def mark_thesis_broken(thesis_id: int, reason: str = "") -> dict:
    """Flag a thesis as broken so the deterministic intraday executor exits the
    position on its next tick (P0c in exits.hard_executor).

    Use this when the thesis is invalidated by news / a fundamental change but
    the price stop hasn't been hit — invalidation discipline says exit anyway.
    The executor reads journal_theses.status from POSTGRES, so this dual-writes:
    SQLite (the journal source of truth, picked up by db_sync) AND a best-effort
    direct Postgres update by id so the flag is visible within one tick rather
    than waiting on the next sync (db_sync dedups theses on a key that includes
    status, so a status change may not propagate as an in-place update).
    """
    with connection() as conn:
        conn.execute(
            "UPDATE theses SET status = 'thesis_broken' WHERE id = ?",
            (thesis_id,),
        )
        if reason:
            conn.execute(
                "INSERT INTO notes (created_at, topic, text, tags, source) "
                "VALUES (?, ?, ?, ?, 'manual')",
                (_now(), f"thesis-{thesis_id}-broken", reason, "mark_thesis_broken"),
            )
        conn.commit()

    pg_synced = False
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            # Only flag a still-live thesis; never resurrect a closed one.
            cur.execute(
                "UPDATE journal_theses SET status = 'thesis_broken' "
                "WHERE id = %s AND status IN ('open', 'triggered')",
                (thesis_id,),
            )
        pg_synced = True
    except Exception as e:  # noqa: BLE001 — best-effort; db_sync is the backstop
        log.warning("mark_thesis_broken: direct Postgres update failed (%s); "
                    "relying on db_sync", e)

    return {"thesis_id": thesis_id, "status": "thesis_broken", "pg_synced": pg_synced}


# Valid trades.provenance values. Aggregates and retrieval treat them
# differently: only 'agent' rows are evidence of the system's own edge.
#   agent            — autonomous pipeline or operator /enter paper fill
#   virtual          — broker rejected the paper order; journaled at mid
#   virtual_backfill — manual copy of an operator REAL trade (possibly scaled)
#   real_mirror      — jobs/mirror_real_fills.py shadow row of a real fill
#   dry_run          — exercise of the pipeline, cancelled before fill
PROVENANCES = ("agent", "virtual", "virtual_backfill", "real_mirror", "dry_run")


def _insert_fill(
    broker_order_id: str,
    symbol: str,
    asset_type: str,
    side: str,
    qty: float,
    fill_price: float,
    thesis_id: int | None,
    strategy_label: str | None,
    stop: float | None,
    target: float | None,
    reasoning: str | None,
    provenance: str = "agent",
    fees: float | None = None,
) -> dict:
    """Internal: insert one row into `trades`. Used by both record_fill and
    record_virtual_fill so neither re-enters an @mcp.tool() wrapper."""
    if provenance not in PROVENANCES:
        provenance = "agent"
    opened_at = _now()
    with connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO trades (thesis_id, symbol, asset_type, strategy_label,
                side, qty, entry_price, stop, target, opened_at, reasoning,
                outcome, broker_order_id, is_paper, provenance, fees)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, 1, ?, ?)
            """,
            (thesis_id, symbol, asset_type, strategy_label, side, qty,
             fill_price, stop, target, opened_at, reasoning, broker_order_id,
             provenance, fees),
        )
        trade_id = cur.lastrowid
    return {
        "trade_id": trade_id,
        "thesis_id": thesis_id,
        "broker_order_id": broker_order_id,
        "is_paper": True,
        "provenance": provenance,
    }


@mcp.tool()
def record_fill(
    broker_order_id: str,
    symbol: str,
    asset_type: Literal["STK", "OPT"],
    side: Literal["BUY", "SELL"],
    qty: float,
    fill_price: float,
    thesis_id: int | None = None,
    strategy_label: str | None = None,
    commission: float | None = None,
    stop: float | None = None,
    target: float | None = None,
    reasoning: str | None = None,
    provenance: str = "agent",
) -> dict:
    """Record a broker fill in the trades table. Usually called by
    posttool_fill_capture; exposed here for manual corrections and the
    virtual-fill fallback.

    `commission` is the entry-side commission+fees in USD and is persisted
    into trades.fees. When omitted, the execution-cost model's per-side
    estimate is charged — paper P&L must never be silently friction-free.
    `provenance` tags the row's origin (see PROVENANCES); only 'agent' rows
    count toward the system's own expectancy statistics.
    """
    if commission is None:
        from trading_agent.execution_costs import fees_per_side
        commission = fees_per_side(qty, asset_type)
    return _insert_fill(
        broker_order_id=broker_order_id,
        symbol=symbol,
        asset_type=asset_type,
        side=side,
        qty=qty,
        fill_price=fill_price,
        thesis_id=thesis_id,
        strategy_label=strategy_label,
        stop=stop,
        target=target,
        reasoning=reasoning,
        provenance=provenance,
        fees=float(commission),
    )


@mcp.tool()
def record_virtual_fill(
    option_symbol: str,
    side: Literal["BUY", "SELL"],
    contracts: int,
    mid_price: float,
    thesis_id: int,
    strategy_label: str | None = None,
    reasoning: str | None = None,
    provenance: str = "virtual",
) -> dict:
    """Record a virtual options fill when MoomooOpenD rejects paper option
    orders. Stored in trades with `broker_order_id='VIRTUAL-<uuid>'` and
    `is_paper=1`, provenance='virtual' (pass 'virtual_backfill' when copying
    an operator REAL trade so aggregates can exclude it). A virtual fill at
    mid never paid the spread, so half a quoted spread plus per-side fees is
    charged into trades.fees up front — friction-free virtual P&L is exactly
    the accounting fiction Phase 0 removes."""
    import uuid

    from trading_agent.execution_costs import fees_per_side, half_spread_cost
    virtual_id = f"VIRTUAL-{uuid.uuid4().hex[:12]}"
    entry_friction = round(
        fees_per_side(contracts, "OPT")
        + half_spread_cost(mid_price, contracts, "OPT"), 4)
    return _insert_fill(
        broker_order_id=virtual_id,
        symbol=option_symbol,
        asset_type="OPT",
        side=side,
        qty=contracts,
        fill_price=mid_price,
        thesis_id=thesis_id,
        strategy_label=strategy_label,
        stop=None,
        target=None,
        reasoning=reasoning,
        provenance=provenance,
        fees=entry_friction,
    )


@mcp.tool()
def record_virtual_stock_fill(
    ticker: str,
    side: Literal["BUY", "SELL"],
    qty: int,
    price: float,
    thesis_id: int,
    strategy_label: str | None = None,
    reasoning: str | None = None,
) -> dict:
    """Record a virtual stock/ETF fill — used to mirror real-account stock
    positions into the paper journal for shadow tracking (e.g. logging an
    AMZN holding so the post-mortem pipeline can reason about it without
    actually placing a paper order).

    Mirror of `record_virtual_fill` but for the stock leg of the asset
    universe: `ticker` is the bare symbol ("AMZN", not "US.AMZN"), `qty` is
    shares, `price` is the fill price. Stored in trades with
    `broker_order_id='VIRTUAL-<uuid>'`, `asset_type='STK'`, `is_paper=1`.
    `qty` is stored unsigned — `side` carries the direction (matching the
    existing convention of `record_fill` / `record_virtual_fill`).
    """
    import uuid

    from trading_agent.execution_costs import fees_per_side, half_spread_cost
    virtual_id = f"VIRTUAL-{uuid.uuid4().hex[:12]}"
    entry_friction = round(
        fees_per_side(qty, "STK") + half_spread_cost(price, qty, "STK"), 4)
    return _insert_fill(
        broker_order_id=virtual_id,
        symbol=ticker.upper(),
        asset_type="STK",
        side=side,
        qty=qty,
        fill_price=price,
        thesis_id=thesis_id,
        strategy_label=strategy_label,
        stop=None,
        target=None,
        reasoning=reasoning,
        provenance="virtual",
        fees=entry_friction,
    )


# |pnl - pnl_recomputed| beyond max(this, 1% of recomputed) flags a mismatch.
_PNL_MISMATCH_TOLERANCE_USD = 1.0


@mcp.tool()
def close_trade(
    trade_id: int,
    exit_price: float,
    outcome: Literal["WIN", "LOSS", "SCRATCH"],
    closed_at: str | None = None,
    pnl: float | None = None,
    exit_fees: float | None = None,
) -> dict:
    """Mark a trade as closed.

    `pnl` can be passed explicitly (preferred when it folds in multi-leg
    economics) — but it is no longer trusted blindly: the journal recomputes
    gross P&L from the stored legs ((exit-entry)*qty*mult, sign-flipped for
    SELL-to-open), persists it as `pnl_recomputed`, and sets `pnl_mismatch=1`
    when the caller's figure disagrees beyond tolerance (trade id 8 recorded
    -$136 against leg arithmetic of -$268 and nobody noticed). When `pnl` is
    omitted it is computed here: gross from legs minus total fees.

    `exit_fees` (USD) is added to trades.fees; when omitted, the
    execution-cost model's per-side estimate is charged.
    """
    closed_at = closed_at or _now()
    thesis_id = None
    thesis_closed = False
    with connection() as conn:
        row = conn.execute(
            "SELECT entry_price, qty, asset_type, side, fees, outcome, "
            "broker_order_id, pnl, pnl_recomputed, pnl_mismatch "
            "FROM trades WHERE id = ?",
            (trade_id,),
        ).fetchone()
        if row is None:
            return {"trade_id": trade_id, "error": "trade not found"}
        # Idempotency guard: a second close (MCP retry, double-fire) must not
        # accumulate phantom exit fees or rewrite pnl — exactly the ledger
        # corruption Phase 0 removes.
        if row["outcome"] != "OPEN":
            return {
                "trade_id": trade_id,
                "error": "already closed",
                "outcome": row["outcome"],
                "pnl": row["pnl"],
                "pnl_recomputed": row["pnl_recomputed"],
                "pnl_mismatch": bool(row["pnl_mismatch"]),
                "fees": row["fees"],
            }
        entry_price = float(row["entry_price"] or 0.0)
        qty = float(row["qty"] or 0.0)
        asset_type = str(row["asset_type"] or "STK")
        side = str(row["side"] or "BUY").upper()
        mult = 100.0 if asset_type == "OPT" else 1.0

        if exit_fees is None:
            from trading_agent.execution_costs import fees_per_side, half_spread_cost
            exit_fees = fees_per_side(qty, asset_type)
            # A virtual position closes at a MARK, not a dealt price — the
            # exit never paid the spread, so charge half a quoted spread on
            # top of fees (the entry side was charged at record time).
            if str(row["broker_order_id"] or "").startswith("VIRTUAL-"):
                exit_fees = round(
                    exit_fees + half_spread_cost(exit_price, qty, asset_type), 4)
        total_fees = round(float(row["fees"] or 0.0) + float(exit_fees), 4)

        # Gross from the stored legs. side is the OPENING side: a SELL-to-open
        # (short premium) profits when exit < entry.
        if side == "SELL":
            gross = (entry_price - float(exit_price)) * qty * mult
        else:
            gross = (float(exit_price) - entry_price) * qty * mult
        pnl_recomputed = round(gross - total_fees, 2)

        if pnl is None:
            pnl = pnl_recomputed
        # Tolerance covers rounding and a caller reporting gross (pre-fee)
        # P&L — the signal we're after is id-8-class errors ($100+), not a
        # missing fee line.
        tolerance = max(_PNL_MISMATCH_TOLERANCE_USD, total_fees,
                        abs(pnl_recomputed) * 0.01)
        # 1e-6 epsilon: a caller reporting exact gross differs from the net
        # figure by exactly total_fees — float noise must not flag that.
        mismatch = 1 if abs(float(pnl) - pnl_recomputed) > tolerance + 1e-6 else 0

        conn.execute(
            "UPDATE trades SET exit_price = ?, outcome = ?, closed_at = ?, pnl = ?, "
            "fees = ?, pnl_recomputed = ?, pnl_mismatch = ? "
            "WHERE id = ? AND outcome = 'OPEN'",
            (exit_price, outcome, closed_at, pnl, total_fees,
             pnl_recomputed, mismatch, trade_id),
        )
        # Thesis lifecycle: once a position fully closes its thesis is no longer
        # live, so it must not linger as a stale 'open' thesis. When this trade's
        # thesis has no remaining OPEN trades, transition it out of 'open'.
        # (Previously close_trade only touched the trade row — every closed
        # position left its thesis 'open' forever, which is why a pile of stale
        # theses accumulated even though their trades had all closed.)
        row = conn.execute(
            "SELECT thesis_id FROM trades WHERE id = ?", (trade_id,)
        ).fetchone()
        thesis_id = row[0] if row else None
        if thesis_id is not None:
            remaining_open = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE thesis_id = ? AND outcome = 'OPEN'",
                (thesis_id,),
            ).fetchone()[0]
            if remaining_open == 0:
                cur = conn.execute(
                    "UPDATE theses SET status = 'triggered' "
                    "WHERE id = ? AND status = 'open'",
                    (thesis_id,),
                )
                thesis_closed = cur.rowcount > 0
        conn.commit()
    return {
        "trade_id": trade_id,
        "outcome": outcome,
        "closed_at": closed_at,
        "thesis_id": thesis_id,
        "thesis_closed": thesis_closed,
        "pnl": pnl,
        "pnl_recomputed": pnl_recomputed,
        "pnl_mismatch": bool(mismatch),
        "fees": total_fees,
    }


@mcp.tool()
def append_note(
    topic: str,
    text: str,
    tags: str = "",
    source: Literal["post_mortem", "manual", "research"] = "manual",
) -> dict:
    """Add a note to the journal. The text is automatically embedded with
    MiniLM and inserted into `notes_vec` for later retrieval.

    `tags` is a comma-separated string ("lesson,AAPL,earnings_IV_crush").
    Lessons from post-mortems should use source='post_mortem' so future
    research queries can filter by origin.
    """
    embedding = _embed(text)
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO notes (created_at, topic, text, tags, source) "
            "VALUES (?, ?, ?, ?, ?)",
            (_now(), topic, text, tags, source),
        )
        note_id = cur.lastrowid
        conn.execute(
            "INSERT INTO notes_vec (note_id, embedding) VALUES (?, ?)",
            (note_id, embedding),
        )
        conn.commit()
    return {"note_id": note_id, "source": source}


# --------- reads ---------

@mcp.tool()
def list_open_theses(ticker: str | None = None, within_minutes: int = 60) -> dict:
    """Theses with status='open'. Filters to the last `within_minutes` minutes
    by default — the PreToolUse hook uses a tighter window (10min) but the
    LLM may want broader context."""
    params: list[Any] = []
    sql = "SELECT * FROM theses WHERE status = 'open'"
    if within_minutes > 0:
        sql += " AND created_at >= datetime('now', ?)"
        params.append(f"-{within_minutes} minutes")
    if ticker:
        sql += " AND ticker = ?"
        params.append(ticker.upper())
    sql += " ORDER BY created_at DESC"
    with connection() as conn:
        rows = _rowdicts(conn.execute(sql, params).fetchall())
    return {"count": len(rows), "rows": rows}


@mcp.tool()
def get_open_positions_with_thesis() -> dict:
    """Join trades (outcome=OPEN) with their theses. The `/eod-review` slash
    command lives on top of this — flag positions whose invalidation has
    already been hit by end-of-day."""
    sql = """
        SELECT
            t.id AS trade_id, t.symbol, t.asset_type, t.side, t.qty,
            t.entry_price, t.stop, t.target, t.opened_at, t.strategy_label,
            th.id AS thesis_id, th.ticker, th.direction, th.thesis_text,
            th.invalidation, th.timeframe, th.expected_return_pct, th.max_loss_pct,
            th.status AS thesis_status
        FROM trades t
        LEFT JOIN theses th ON th.id = t.thesis_id
        WHERE t.outcome = 'OPEN'
        ORDER BY t.opened_at DESC
    """
    with connection() as conn:
        rows = _rowdicts(conn.execute(sql).fetchall())
    return {"count": len(rows), "rows": rows}


# --------- retrieval ---------

@mcp.tool()
def search_past_trades(query: str, k: int = 5) -> dict:
    """Hybrid retrieval over the journal.

    1. Vector: encode `query` with MiniLM, KNN over `notes_vec`, join back
       to `notes` for text.
    2. Lexical: LIKE match on `trades.strategy_label` and `trades.symbol`
       (upper-cased query tokens); attach any thesis + post-mortem notes.

    Returns both blocks separately so the caller can render them
    differently. `k` caps each side independently.
    """
    emb = _embed(query)

    with connection() as conn:
        # Vector neighbors in notes_vec → pull note rows.
        vec_rows = _rowdicts(conn.execute(
            """
            SELECT n.id AS note_id, n.created_at, n.topic, n.text, n.tags, n.source,
                   v.distance
            FROM notes_vec v
            JOIN notes n ON n.id = v.note_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (emb, k),
        ).fetchall())

        # Lexical match on trades (symbol + strategy_label). Contaminated
        # provenance is excluded: virtual_backfill rows are scaled copies of
        # operator real trades (the XLE 1→5x amplified loss) and dry_run rows
        # never traded — retrieving them as "past trades" poisons the memory
        # the synthesizer learns from. agent + real_mirror + virtual remain,
        # tagged so the caller can weigh them.
        q_upper = query.upper()
        like = f"%{q_upper}%"
        lex_rows = _rowdicts(conn.execute(
            """
            SELECT id AS trade_id, symbol, asset_type, strategy_label, side,
                   qty, entry_price, exit_price, outcome, pnl, opened_at, closed_at,
                   COALESCE(provenance, 'agent') AS provenance
            FROM trades
            WHERE (UPPER(symbol) LIKE ? OR UPPER(COALESCE(strategy_label,'')) LIKE ?)
              AND COALESCE(provenance, 'agent') NOT IN ('virtual_backfill', 'dry_run')
            ORDER BY opened_at DESC
            LIMIT ?
            """,
            (like, like, k),
        ).fetchall())

    return {
        "query": query,
        "notes_semantic": vec_rows,
        "trades_lexical": lex_rows,
    }


@mcp.tool()
def search_notes(query: str, source: str | None = None, k: int = 10) -> dict:
    """Vector-only search over `notes`. Filter by `source` when you want
    only post-mortem lessons (`source='post_mortem'`)."""
    emb = _embed(query)
    sql = """
        SELECT n.id AS note_id, n.created_at, n.topic, n.text, n.tags, n.source,
               v.distance
        FROM notes_vec v
        JOIN notes n ON n.id = v.note_id
        WHERE v.embedding MATCH ? AND k = ?
    """
    params: list[Any] = [emb, k]
    if source:
        sql += " AND n.source = ?"
        params.append(source)
    sql += " ORDER BY v.distance"
    with connection() as conn:
        rows = _rowdicts(conn.execute(sql, params).fetchall())
    return {"query": query, "count": len(rows), "rows": rows}


def _realized_r(t: dict, pnl: float) -> float | None:
    """One closed trade's pnl in R units: pnl / (|entry − stop| × qty × mult).

    None when the trade carries no stop (legacy rows) — we refuse to invent
    a risk denominator, the same posture as sizing's implicit-stop WARN.
    """
    entry, stop, qty = t.get("entry_price"), t.get("stop"), t.get("qty")
    if entry is None or stop is None or not qty:
        return None
    risk_per_unit = abs(float(entry) - float(stop))
    if risk_per_unit < 1e-9:
        return None
    mult = 100.0 if t.get("asset_type") == "OPT" else 1.0
    return pnl / (risk_per_unit * float(qty) * mult)


@mcp.tool()
def generate_post_mortem_prompt(since_date: str) -> dict:
    """Return a structured prompt payload for the weekly post-mortem.

    `since_date` is ISO YYYY-MM-DD. Collects all trades closed at/after
    that date, their theses, and any snapshots. The post-mortem skill uses
    this payload to grade each strategy_label against its declared
    StrategySpec (expectancy, not raw win rate) and produce 3–8 lessons —
    which are then written back via `append_note`.
    """
    with connection() as conn:
        trades = _rowdicts(conn.execute(
            """
            SELECT
                t.id AS trade_id, t.symbol, t.asset_type, t.strategy_label, t.side,
                t.qty, t.entry_price, t.exit_price, t.stop, t.target, t.pnl,
                t.pnl_recomputed, t.pnl_mismatch, t.fees,
                COALESCE(t.provenance, 'agent') AS provenance,
                t.outcome, t.opened_at, t.closed_at, t.broker_order_id,
                th.id AS thesis_id, th.ticker, th.direction, th.thesis_text,
                th.invalidation, th.timeframe, th.expected_return_pct, th.max_loss_pct,
                th.status AS thesis_status
            FROM trades t
            LEFT JOIN theses th ON th.id = t.thesis_id
            WHERE t.closed_at IS NOT NULL AND t.closed_at >= ?
            ORDER BY t.closed_at
            """,
            (since_date,),
        ).fetchall())

    # Aggregates are split by provenance: only 'agent' rows are evidence of
    # the SYSTEM's edge. Before this split, 5x-scaled copies of operator real
    # losses (virtual_backfill) sat in the same by_strategy buckets as agent
    # trades, so the post-mortem "learned" from losses the agent never took.
    by_strategy: dict[str, dict[str, Any]] = {}
    by_strategy_shadow: dict[str, dict[str, Any]] = {}
    # Per-agent-strategy raw series for the spec_comparison block: dollar
    # pnl (for avg win / avg loss) and realized R (for mean R).
    realized_pnls: dict[str, list[float]] = {}
    realized_rs: dict[str, list[float]] = {}
    agent_trades = 0
    for t in trades:
        s = t["strategy_label"] or "(none)"
        # 'virtual' fills are the agent's OWN decisions (broker rejected the
        # paper order) — they count as system evidence. Only copies of
        # operator trades and dry runs are quarantined to the shadow bucket.
        is_agent = t["provenance"] in ("agent", "virtual")
        bucket = by_strategy if is_agent else by_strategy_shadow
        agent_trades += 1 if is_agent else 0
        b = bucket.setdefault(s, {"n": 0, "wins": 0, "losses": 0, "scratches": 0, "pnl": 0.0})
        b["n"] += 1
        if t["outcome"] == "WIN":
            b["wins"] += 1
        elif t["outcome"] == "LOSS":
            b["losses"] += 1
        elif t["outcome"] == "SCRATCH":
            b["scratches"] += 1
        # Prefer the leg-arithmetic figure over self-reported pnl when they
        # disagree — the recomputed value is the auditable one.
        pnl_val = t["pnl_recomputed"] if t["pnl_mismatch"] else t["pnl"]
        pnl_used = float(pnl_val or t["pnl"] or 0.0)
        b["pnl"] += pnl_used
        if is_agent:
            realized_pnls.setdefault(s, []).append(pnl_used)
            r = _realized_r(t, pnl_used)
            if r is not None:
                realized_rs.setdefault(s, []).append(r)

    # Declared-vs-realized (Phase 2 item 9): grade each agent strategy
    # bucket against its declared StrategySpec so the post-mortem judges
    # SHAPE (WR envelope + payoff ratio + expectancy) instead of raw win
    # rate. Legacy labels with no governing spec are skipped — we don't
    # grade a trade against a spec it never declared.
    spec_comparison: dict[str, Any] = {}
    for s, b in by_strategy.items():
        spec = spec_for_label(s)
        if spec is None or spec.expectancy_profile is None:
            continue
        n = b["n"]
        wr = (b["wins"] / n) if n else 0.0
        rs = realized_rs.get(s, [])
        pnls = realized_pnls.get(s, [])
        win_pnls = [p for p in pnls if p > 0]
        loss_pnls = [abs(p) for p in pnls if p < 0]
        lo, hi = spec.expectancy_profile.expected_wr_range
        # verdict_hint grades the WR envelope only — expectancy verdicts
        # need LB95(mean R) at n >= min_trades_for_eval, which the
        # moratorium instruction below leaves to the LLM + promote gates.
        if n < spec.min_trades_for_eval:
            verdict = "INSUFFICIENT_N"
        elif lo <= wr <= hi:
            verdict = "WITHIN_DECLARED"
        else:
            verdict = "OUTSIDE_DECLARED"
        spec_comparison[s] = {
            "spec": spec.name,
            "spec_status": spec.status,
            "declared": {
                "expected_wr_range": [lo, hi],
                "breakeven_wr_net": round(
                    spec.expectancy_profile.breakeven_wr_net, 4),
                "min_risk_reward": spec.min_risk_reward,
                "min_trades_for_eval": spec.min_trades_for_eval,
                "falsification": spec.falsification,
            },
            "realized": {
                "n": n,
                "win_rate": round(wr, 4),
                "mean_r": (round(sum(rs) / len(rs), 4) if rs else None),
                "n_with_r": len(rs),  # rows with a stop → R computable
                "avg_win": (round(sum(win_pnls) / len(win_pnls), 2)
                            if win_pnls else None),
                "avg_loss": (round(sum(loss_pnls) / len(loss_pnls), 2)
                             if loss_pnls else None),
            },
            "verdict_hint": verdict,
        }

    return {
        "since_date": since_date,
        "closed_trade_count": len(trades),
        "agent_trade_count": agent_trades,
        "by_strategy": by_strategy,
        "by_strategy_shadow": by_strategy_shadow,
        "spec_comparison": spec_comparison,
        "trades": trades,
        "instructions": (
            "Analyze the closed trades above. by_strategy covers ONLY "
            "provenance='agent' rows — the system's own record; "
            "by_strategy_shadow holds mirrored/backfilled operator trades for "
            "context and must NOT be merged into the system's statistics. "
            "spec_comparison grades each agent strategy bucket against its "
            "DECLARED StrategySpec (strategy_specs.py is canonical): compare "
            "realized win rate / mean R / avg win / avg loss against the "
            "declared envelope and explicitly flag any strategy with "
            "verdict_hint=OUTSIDE_DECLARED — outside the declared envelope "
            "in EITHER direction needs a lesson. Flag as moratorium "
            "candidates ONLY strategies whose realized mean R is credibly "
            "negative (LB95(mean R) < 0 with n >= spec.min_trades_for_eval, "
            "or net-negative pnl with n >= 10); win rate alone is NEVER a "
            "moratorium reason — low-WR/high-payoff is the declared shape of "
            "the convexity track. Rows with pnl_mismatch=1 have self-reported "
            "pnl contradicting leg arithmetic — trust pnl_recomputed. Check "
            "thesis_text vs outcome — was the thesis right but sizing wrong, "
            "or was the thesis wrong to begin with? Produce 3–8 concrete "
            "lessons (each 1–2 sentences, with tags like "
            "'lesson,<symbol>,<label>') and call append_note for each with "
            "source='post_mortem'."
        ),
    }


# --------- entry point ---------

def main() -> None:
    print(
        f"[journal-mcp] db={CONFIG.db_path} "
        f"embedder={CONFIG.embedding_model} (loaded on first use)",
        file=sys.stderr,
    )
    mcp.run()


if __name__ == "__main__":
    main()
