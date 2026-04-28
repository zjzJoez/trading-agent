-- Phase 2.7.5 — Intraday MAE / MFE for open positions
-- Run via: ta-pg-migrate (which loads .env.postgres for the DSN)
-- Idempotent: ADD COLUMN IF NOT EXISTS.
--
-- Adds 4 columns to journal_trades so the intraday_monitor graph can update
-- a running adverse / favorable excursion on every refresh.  At close time,
-- the values are copied into trade_outcome_features.{mae,mfe} by
-- outcome.compute_outcome.
--
-- Convention (long-call paper-only — MVP scope):
--    mae_so_far  = MIN(realized_R_at_quote)   (most-negative)
--    mfe_so_far  = MAX(realized_R_at_quote)   (most-positive)
-- where realized_R_at_quote = (mid - entry) / |entry - stop|.
-- For trades without a stop, the implicit-stop fraction from sizing.py
-- (currently 5%) is substituted.

BEGIN;

ALTER TABLE journal_trades
    ADD COLUMN IF NOT EXISTS mae_so_far       NUMERIC(14,6),
    ADD COLUMN IF NOT EXISTS mfe_so_far       NUMERIC(14,6),
    ADD COLUMN IF NOT EXISTS last_quote_at    TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_quote_price NUMERIC(14,6);

INSERT INTO schema_migrations (version, notes)
VALUES ('004_intraday_extremes',
        'Phase 2.7.5 — running mae_so_far / mfe_so_far on journal_trades; updated by intraday_monitor refresh, copied to trade_outcome_features at close')
ON CONFLICT (version) DO NOTHING;

COMMIT;
