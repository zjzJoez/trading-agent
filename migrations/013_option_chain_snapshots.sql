-- Migration 013 — Daily EOD option-chain snapshots
--
-- Captures one near-the-money chain slice per underlying per trading day:
-- watchlist names plus the underlyings of any OPEN journal trade. Written
-- by jobs/cache_option_chains.py (systemd timer, 21:15 UTC weekdays —
-- after US close, before the 21:30 eod_review).
--
-- WHY: the system trades short-dated options, but moomoo's free tier has
-- no historical option quotes — once a session ends, the chain the agent
-- actually saw is gone forever. Without this table every strategy the
-- agent runs is permanently un-backtestable: we can replay decisions from
-- agent_events/shadow_proposals, but not price what the alternatives
-- would have cost. A daily EOD snapshot is the cheapest possible record
-- that makes "what if we'd picked the other strike/expiry" answerable
-- later.
--
-- Granularity: ~12 strikes each side of spot, up to 4 expiries in the
-- 7-90 DTE band per underlying — the band the system trades (R5 windows
-- sit inside it). Not the full chain: deep wings cost quote-quota and
-- never get traded.
--
-- UNIQUE (snapshot_date, code) makes re-runs idempotent: the job INSERTs
-- with ON CONFLICT DO UPDATE, so a manual re-run after a partial failure
-- refreshes rows instead of duplicating them.

CREATE TABLE IF NOT EXISTS option_chain_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    snapshot_date   DATE NOT NULL,                  -- US trading date being snapped
    underlying      TEXT NOT NULL,                  -- bare ticker, e.g. AAPL
    spot            NUMERIC,                        -- underlying last_price at capture
    code            TEXT NOT NULL,                  -- full moomoo option code, e.g. US.AAPL260717C200000
    expiry          DATE,
    strike          NUMERIC,
    option_type     TEXT,                           -- CALL | PUT
    bid             NUMERIC,
    ask             NUMERIC,
    last            NUMERIC,
    volume          NUMERIC,
    open_interest   NUMERIC,
    iv              NUMERIC,                        -- implied vol as a FRACTION (0.285 = 28.5%);
                                                    -- broker reports percent, the job divides by 100
                                                    -- (same convention as trade_nodes' chain prefetch)
    delta           NUMERIC,
    gamma           NUMERIC,
    vega            NUMERIC,
    theta           NUMERIC,
    captured_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (snapshot_date, code)
);
CREATE INDEX IF NOT EXISTS idx_chain_snap_underlying_date
    ON option_chain_snapshots(underlying, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_chain_snap_date
    ON option_chain_snapshots(snapshot_date);

INSERT INTO schema_migrations (version, notes)
VALUES ('013_option_chain_snapshots',
        'Daily EOD option-chain cache — makes traded strategies backtestable')
ON CONFLICT (version) DO NOTHING;
