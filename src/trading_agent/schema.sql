-- trading-agent Phase-1 schema
-- SQLite with sqlite-vec extension (vec0 virtual table for embeddings).

CREATE TABLE IF NOT EXISTS theses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,              -- LONG/SHORT/LONG_CALL/LONG_PUT/...
    thesis_text TEXT NOT NULL,
    invalidation TEXT NOT NULL,
    timeframe TEXT,
    expected_return_pct REAL,
    max_loss_pct REAL,
    status TEXT NOT NULL DEFAULT 'open'   -- open/triggered/void
);
CREATE INDEX IF NOT EXISTS idx_theses_ticker_time ON theses(ticker, created_at);
CREATE INDEX IF NOT EXISTS idx_theses_status ON theses(status);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thesis_id INTEGER REFERENCES theses(id),
    symbol TEXT NOT NULL,                 -- US.AAPL or US.AAPL250117C00200000
    asset_type TEXT NOT NULL,             -- STK/OPT
    strategy_label TEXT,
    side TEXT,                            -- BUY/SELL
    qty REAL,
    entry_price REAL,
    exit_price REAL,
    stop REAL,
    target REAL,
    opened_at TEXT,
    closed_at TEXT,
    reasoning TEXT,
    pnl REAL,
    outcome TEXT NOT NULL DEFAULT 'OPEN', -- OPEN/WIN/LOSS/SCRATCH
    broker_order_id TEXT,
    is_paper INTEGER NOT NULL DEFAULT 1,
    -- Phase 0 measurement integrity (2026-06): see db._ensure_trade_columns
    -- for the matching ALTERs applied to pre-existing databases.
    provenance TEXT NOT NULL DEFAULT 'agent',  -- agent/virtual/virtual_backfill/real_mirror/dry_run
    fees REAL,                            -- cumulative round-trip commissions+fees (USD)
    pnl_recomputed REAL,                  -- (exit-entry)*qty*mult from stored legs; audits self-reported pnl
    pnl_mismatch INTEGER DEFAULT 0,       -- 1 when |pnl - pnl_recomputed| exceeds tolerance
    exit_order_id TEXT,                   -- broker order id of an in-flight close (pending-exit state machine)
    exit_state_json TEXT                  -- pending-close state: action/reason/requested price/attempts
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_opened ON trades(opened_at);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_label);
CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades(outcome);
CREATE INDEX IF NOT EXISTS idx_trades_thesis ON trades(thesis_id);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER REFERENCES trades(id),
    taken_at TEXT NOT NULL,
    payload JSON NOT NULL                 -- {quote, chain_excerpt, greeks, iv, vix, spy}
);
CREATE INDEX IF NOT EXISTS idx_snapshots_trade ON market_snapshots(trade_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_time ON market_snapshots(taken_at);

CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    topic TEXT,
    text TEXT NOT NULL,
    tags TEXT,                            -- comma-separated, e.g. "lesson,AAPL,earnings_IV_crush"
    source TEXT                           -- post_mortem/manual/research
);
CREATE INDEX IF NOT EXISTS idx_notes_tags ON notes(tags);
CREATE INDEX IF NOT EXISTS idx_notes_source ON notes(source);
CREATE INDEX IF NOT EXISTS idx_notes_created ON notes(created_at);

-- vec0 virtual table lives in separate statements (require sqlite-vec loaded).
-- Embedding dim = 384 (sentence-transformers all-MiniLM-L6-v2).
CREATE VIRTUAL TABLE IF NOT EXISTS notes_vec USING vec0(
    note_id INTEGER PRIMARY KEY,
    embedding FLOAT[384]
);

CREATE TABLE IF NOT EXISTS post_mortems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    summary TEXT,
    lessons_json JSON,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_post_mortems_period ON post_mortems(period_end);

CREATE TABLE IF NOT EXISTS hook_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    hook_name TEXT NOT NULL,
    tool_name TEXT,
    decision TEXT NOT NULL,               -- allow/block
    reason TEXT,
    payload JSON
);
CREATE INDEX IF NOT EXISTS idx_audit_created ON hook_audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_decision ON hook_audit_log(decision);
