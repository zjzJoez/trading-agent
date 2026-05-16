-- Migration 007 — paper-benchmark measurement windows
--
-- Lets us commit to a fixed evaluation window (T0 → T0+N) and accumulate
-- daily marks of (portfolio NAV, SPY close, 60/40 SPY/TLT baseline NAV) so
-- that at the end of the window we can compute Sharpe / max drawdown /
-- win rate vs the passive baselines with a clear pre-registered start.
--
-- Why a separate table (not agent_events): agent_events is high-churn —
-- premarket scans, intraday ticks, EOD reviews all dump rows there. The
-- benchmark window must be cleanly queryable years from now, even after
-- agent_events has been partitioned/pruned.
--
-- A row in benchmark_windows is the "manifesto": this is the period we
-- claim measures the system. There should typically be at most one row
-- with status='active' at a time.

CREATE TABLE IF NOT EXISTS benchmark_windows (
    id                          BIGSERIAL PRIMARY KEY,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    t0_ts                       TIMESTAMPTZ NOT NULL,
    -- Captured at T0 so the per-mark series doesn't need to back-fill these:
    t0_spy_close                NUMERIC(12, 4),
    t0_tlt_close                NUMERIC(12, 4),
    t0_portfolio_nav            NUMERIC(14, 2),
    t0_baseline_60_40_nav       NUMERIC(14, 2),       -- 60% SPY + 40% TLT @ T0, no rebal
    -- How the 60/40 baseline is constructed — recorded here so we can audit
    -- "did we change the construction mid-window".  Allowed values:
    --   'fixed_units' = at T0 buy SPY units = 0.6 * NAV / SPY_T0, TLT units
    --                   = 0.4 * NAV / TLT_T0, hold forever. No rebal.
    --   'daily_rebal' = each day rebalance to 60/40. Adds friction model.
    baseline_construction       TEXT NOT NULL DEFAULT 'fixed_units',
    status                      TEXT NOT NULL DEFAULT 'active',  -- active | closed | aborted
    note                        TEXT,
    closed_at                   TIMESTAMPTZ
);

-- One window can have many daily marks. The unique constraint stops a cron
-- from inserting two rows for the same trading day if it fires twice.
CREATE TABLE IF NOT EXISTS benchmark_marks (
    id                          BIGSERIAL PRIMARY KEY,
    window_id                   BIGINT NOT NULL REFERENCES benchmark_windows(id) ON DELETE CASCADE,
    ts                          TIMESTAMPTZ NOT NULL,
    mark_date                   DATE NOT NULL,
    spy_close                   NUMERIC(12, 4),
    tlt_close                   NUMERIC(12, 4),
    portfolio_nav               NUMERIC(14, 2),
    baseline_60_40_nav          NUMERIC(14, 2),
    -- Cumulative returns from T0 — denormalized for query speed
    cum_strategy_ret            NUMERIC(8, 6),
    cum_spy_ret                 NUMERIC(8, 6),
    cum_baseline_ret            NUMERIC(8, 6),
    notes                       JSONB DEFAULT '{}'::jsonb,
    UNIQUE (window_id, mark_date)
);
CREATE INDEX IF NOT EXISTS idx_benchmark_marks_window ON benchmark_marks(window_id, mark_date DESC);

INSERT INTO schema_migrations (version, applied_at)
VALUES ('007', NOW())
ON CONFLICT (version) DO NOTHING;
