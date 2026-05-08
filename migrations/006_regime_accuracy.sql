-- Migration 003 — Regime accuracy labels (Phase 2.5+)
--
-- Adds two columns to regime_states so update_regime_accuracy_labels
-- (eod_review_graph) can persist the daily directional accuracy of each
-- regime classification.
--
-- accuracy_label:       'correct' | 'incorrect' | 'neutral' | 'error'
-- spy_next_day_return:  next-day SPY return fraction (e.g. 0.0123 = +1.23%)
--
-- Safe to re-run: ADD COLUMN IF NOT EXISTS is idempotent on Postgres 9.6+.

ALTER TABLE regime_states
    ADD COLUMN IF NOT EXISTS accuracy_label        TEXT,
    ADD COLUMN IF NOT EXISTS spy_next_day_return   NUMERIC(8, 6);

CREATE INDEX IF NOT EXISTS idx_regime_states_accuracy
    ON regime_states(accuracy_label, created_at DESC)
    WHERE accuracy_label IS NOT NULL;

-- Record this migration
INSERT INTO schema_migrations (version, applied_at)
VALUES ('006', NOW())
ON CONFLICT (version) DO NOTHING;
