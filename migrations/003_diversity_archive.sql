-- Phase 2.6.5 — Diversity Archive (QuantEvolve cells)
-- Run via: ta-pg-migrate (which loads .env.postgres for the DSN)
-- Idempotent: every CREATE uses IF NOT EXISTS.
--
-- Adds one table — `param_archive_cells` — that stores per-cell running
-- aggregates of realized trade outcomes, partitioned by:
--
--     cell_key = "<regime>|<strategy>|<dte_bucket>|<iv_bucket>"
--
-- Each (cell × param_version) gets one row. Aggregates update atomically
-- on each closed-trade enrichment. The `is_champion` flag elects one
-- best-scoring param_version per cell (atomic via a single transaction
-- in `learning/archive.update_champion`).
--
-- Composite score (per cell): `sharpe − λ · max_drawdown_R`, λ = 0.5.
-- IR is intentionally NOT included at cell granularity — too noisy for
-- the small per-cell sample sizes the archive sees in early operation.

BEGIN;

CREATE TABLE IF NOT EXISTS param_archive_cells (
    id                  BIGSERIAL PRIMARY KEY,
    cell_key            TEXT NOT NULL,
    cell_regime         TEXT NOT NULL,
    cell_strategy       TEXT NOT NULL,
    cell_dte_bucket     TEXT NOT NULL,
    cell_iv_bucket      TEXT NOT NULL,
    param_version_id    INTEGER NOT NULL REFERENCES param_versions(id),

    -- Running aggregates (Welford-style — no need to re-read history)
    n_trades            INTEGER NOT NULL DEFAULT 0,
    cum_R               NUMERIC(14,6) NOT NULL DEFAULT 0,
    sum_R_sq            NUMERIC(14,6) NOT NULL DEFAULT 0,
    peak_cum_R          NUMERIC(14,6) NOT NULL DEFAULT 0,
    max_drawdown_R      NUMERIC(14,6) NOT NULL DEFAULT 0,

    -- Derived
    composite_score     NUMERIC(14,6),
    is_champion         BOOLEAN NOT NULL DEFAULT FALSE,

    -- Bookkeeping
    first_trade_at      TIMESTAMPTZ,
    last_trade_at       TIMESTAMPTZ,
    last_updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (cell_key, param_version_id)
);

-- Champion lookup must be O(1) and atomic.
CREATE INDEX IF NOT EXISTS idx_archive_cell_champion
    ON param_archive_cells (cell_key) WHERE is_champion;

-- Reverse lookup: which cells does this param_version live in?
CREATE INDEX IF NOT EXISTS idx_archive_cell_param
    ON param_archive_cells (param_version_id);

-- Diagnostics: scan an entire regime / strategy in cell-key prefix order.
CREATE INDEX IF NOT EXISTS idx_archive_cell_regime
    ON param_archive_cells (cell_regime, cell_strategy);

INSERT INTO schema_migrations (version, notes)
VALUES ('003_diversity_archive',
        'Phase 2.6.5 — QuantEvolve diversity archive: per-cell running aggregates + champion-per-cell election')
ON CONFLICT (version) DO NOTHING;

COMMIT;
