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
import struct
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from trading_agent.config import CONFIG
from trading_agent.db import connection

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
) -> dict:
    """Internal: insert one row into `trades`. Used by both record_fill and
    record_virtual_fill so neither re-enters an @mcp.tool() wrapper."""
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
             fill_price, stop, target, opened_at, reasoning, broker_order_id),
        )
        trade_id = cur.lastrowid
    return {
        "trade_id": trade_id,
        "thesis_id": thesis_id,
        "broker_order_id": broker_order_id,
        "is_paper": True,
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
    commission: float = 0.0,
    stop: float | None = None,
    target: float | None = None,
    reasoning: str | None = None,
) -> dict:
    """Record a broker fill in the trades table. Usually called by
    posttool_fill_capture; exposed here for manual corrections and the
    virtual-fill fallback. `commission` is accepted for future use but not
    persisted yet (MVP paper accounts on Moomoo don't emit commissions)."""
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
) -> dict:
    """Record a virtual options fill when MoomooOpenD rejects paper option
    orders. Stored in trades with `broker_order_id='VIRTUAL-<uuid>'` and
    `is_paper=1`. The journal/post-mortem pipeline treats virtual fills
    identically to broker fills — the only difference is that cancelling
    and modifying are no-ops (there's no broker order to cancel)."""
    import uuid
    virtual_id = f"VIRTUAL-{uuid.uuid4().hex[:12]}"
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
    virtual_id = f"VIRTUAL-{uuid.uuid4().hex[:12]}"
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
    )


@mcp.tool()
def close_trade(
    trade_id: int,
    exit_price: float,
    outcome: Literal["WIN", "LOSS", "SCRATCH"],
    closed_at: str | None = None,
    pnl: float | None = None,
) -> dict:
    """Mark a trade as closed. `pnl` can be passed explicitly (preferred,
    since it folds in commissions and multi-leg economics); otherwise the
    caller should pre-compute and pass it.
    """
    closed_at = closed_at or _now()
    with connection() as conn:
        conn.execute(
            "UPDATE trades SET exit_price = ?, outcome = ?, closed_at = ?, pnl = ? "
            "WHERE id = ?",
            (exit_price, outcome, closed_at, pnl, trade_id),
        )
        conn.commit()
    return {"trade_id": trade_id, "outcome": outcome, "closed_at": closed_at}


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

        # Lexical match on trades (symbol + strategy_label).
        q_upper = query.upper()
        like = f"%{q_upper}%"
        lex_rows = _rowdicts(conn.execute(
            """
            SELECT id AS trade_id, symbol, asset_type, strategy_label, side,
                   qty, entry_price, exit_price, outcome, pnl, opened_at, closed_at
            FROM trades
            WHERE UPPER(symbol) LIKE ? OR UPPER(COALESCE(strategy_label,'')) LIKE ?
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


@mcp.tool()
def generate_post_mortem_prompt(since_date: str) -> dict:
    """Return a structured prompt payload for the weekly post-mortem.

    `since_date` is ISO YYYY-MM-DD. Collects all trades closed at/after
    that date, their theses, and any snapshots. The post-mortem skill uses
    this payload to compute win rates per strategy_label and produce 3–8
    lessons — which are then written back via `append_note`.
    """
    with connection() as conn:
        trades = _rowdicts(conn.execute(
            """
            SELECT
                t.id AS trade_id, t.symbol, t.asset_type, t.strategy_label, t.side,
                t.qty, t.entry_price, t.exit_price, t.stop, t.target, t.pnl,
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

    # Quick aggregates so the skill doesn't have to recompute.
    by_strategy: dict[str, dict[str, Any]] = {}
    for t in trades:
        s = t["strategy_label"] or "(none)"
        b = by_strategy.setdefault(s, {"n": 0, "wins": 0, "losses": 0, "scratches": 0, "pnl": 0.0})
        b["n"] += 1
        if t["outcome"] == "WIN":
            b["wins"] += 1
        elif t["outcome"] == "LOSS":
            b["losses"] += 1
        elif t["outcome"] == "SCRATCH":
            b["scratches"] += 1
        b["pnl"] += float(t["pnl"] or 0.0)

    return {
        "since_date": since_date,
        "closed_trade_count": len(trades),
        "by_strategy": by_strategy,
        "trades": trades,
        "instructions": (
            "Analyze the closed trades above. For each strategy_label, compute "
            "win rate and avg pnl; flag strategies with <40% win rate or net-negative "
            "pnl as candidates for moratorium. Check thesis_text vs outcome — was the "
            "thesis right but sizing wrong, or was the thesis wrong to begin with? "
            "Produce 3–8 concrete lessons (each 1–2 sentences, with tags like "
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
