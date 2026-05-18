-- Migration 010 — DB-backed two-tier watchlist
--
-- Replaces the hand-maintained DEFAULT_WATCHLIST in jobs/premarket_watchlist.py.
-- The watchlist is now state, not code, so it can be:
--   1. Refreshed daily from yfinance screeners (day_gainers / day_losers /
--      most_actives) and the user's moomoo CUSTOM groups (MGGA7, 芯片, etc.)
--   2. Auto-evicted when a rotating ticker hasn't generated a proposal or
--      meaningful scout score in N days
--   3. Audited — every add/evict is logged with its source so we can see
--      "which channel did each trade originate from"
--
-- TWO-TIER STRUCTURE:
--   tier='core'     ~10-15 tickers — Mag-7 + broad indices. Never auto-evicted.
--                                    Seeded from moomoo MGGA7 + ['SPY','QQQ','IWM'].
--   tier='rotating' ~20-30 tickers — Auto-managed daily. Sources:
--                                      • yfinance day_gainers / day_losers / most_actives
--                                      • moomoo 芯片 / 商品 / 指数 group changes
--                                      • manual_add via CLI (--ticker flag)
--
-- LIQUIDITY FLOOR for any auto-add:
--   30d avg volume ≥ 1,000,000
--   market_cap ≥ $2,000,000,000
--   US-listed common stock (no OTC, no derivatives)
-- These are enforced in scripts/refresh_dynamic_watchlist.py, NOT here.
-- The DB stores whatever the refresh script accepted.
--
-- EVICTION (rotating only — core never evicts) — all conditions must hold:
--   tier = 'rotating'
--   added_at < NOW() - INTERVAL '14 days'
--   (last_scout_score IS NULL OR last_scout_score < 0.4)
--   (last_proposed_at IS NULL OR last_proposed_at < NOW() - INTERVAL '30 days')
--   ticker NOT IN (currently held positions)  -- enforced by refresh script
--
-- CAP: total membership hard-capped at 40 by refresh script. If above cap
-- after the add-pass, lowest-priority rotating members (lowest add_score
-- in metadata) are pruned.

CREATE TABLE IF NOT EXISTS watchlist_members (
    ticker            TEXT PRIMARY KEY,                -- bare ticker, e.g. 'NVDA' (no 'US.' prefix)
    tier              TEXT NOT NULL CHECK (tier IN ('core', 'rotating')),
    source            TEXT NOT NULL,                   -- 'static_seed' | 'moomoo:<group>' | 'screener:<scrId>' | 'user'
    added_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    added_by_run_id   TEXT,                            -- run_id of the refresh job that added this row
    last_seen_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),   -- last refresh that re-confirmed this ticker from its source
    last_scout_score  REAL,                            -- most recent scout score (from rank_candidates)
    last_scout_ts     TIMESTAMPTZ,
    last_proposed_at  TIMESTAMPTZ,                     -- most recent shadow_proposals.proposal_ts for this ticker
    eligible          BOOLEAN NOT NULL DEFAULT TRUE,   -- if FALSE, refresh keeps the row but premarket scan excludes it (manual hide)
    metadata          JSONB NOT NULL DEFAULT '{}'::jsonb  -- snapshot of mover stats: market_cap, avg_volume_30d, gap_pct, etc.
);

CREATE INDEX IF NOT EXISTS idx_watchlist_members_tier ON watchlist_members(tier);
CREATE INDEX IF NOT EXISTS idx_watchlist_members_last_seen ON watchlist_members(last_seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_watchlist_members_last_proposed ON watchlist_members(last_proposed_at DESC NULLS LAST);

-- Audit table — every add / evict / manual change leaves a row here.
-- Lets us answer "which channel surfaced the ticker we ended up trading?"
CREATE TABLE IF NOT EXISTS watchlist_audit (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ticker      TEXT NOT NULL,
    action      TEXT NOT NULL CHECK (action IN ('ADD', 'EVICT', 'TIER_CHANGE', 'DISABLE', 'ENABLE', 'TOUCH')),
    tier_before TEXT,
    tier_after  TEXT,
    source      TEXT,                                  -- channel that triggered the action
    run_id      TEXT,                                  -- refresh job run_id
    reason      TEXT,                                  -- human-readable
    metadata    JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_watchlist_audit_ts ON watchlist_audit(ts DESC);
CREATE INDEX IF NOT EXISTS idx_watchlist_audit_ticker ON watchlist_audit(ticker, ts DESC);

-- SEED — initial core tier. These mirror the existing static Mag-7 + broad
-- indices set. Rotating tier members are added on the first refresh run.
-- ON CONFLICT DO NOTHING — re-running this migration is a no-op.
INSERT INTO watchlist_members (ticker, tier, source) VALUES
    ('SPY',  'core', 'static_seed'),
    ('QQQ',  'core', 'static_seed'),
    ('IWM',  'core', 'static_seed'),
    ('AAPL', 'core', 'static_seed'),
    ('MSFT', 'core', 'static_seed'),
    ('NVDA', 'core', 'static_seed'),
    ('GOOGL','core', 'static_seed'),
    ('META', 'core', 'static_seed'),
    ('TSLA', 'core', 'static_seed'),
    ('AMZN', 'core', 'static_seed')
ON CONFLICT (ticker) DO NOTHING;

INSERT INTO schema_migrations (version, applied_at)
VALUES ('010', NOW())
ON CONFLICT (version) DO NOTHING;
