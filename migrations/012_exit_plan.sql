-- Migration 012 — exit_plan on journal_trades
--
-- Stores the full deterministic exit plan baked at entry time:
--   {hard_stop, hard_target, scale_out_ladder, trail_stop, dte_rules,
--    regime_rules, time_in_trade_max_days}
--
-- Rationale (see docs/position_management.md):
--   * Exit decisions are now deterministic code, not LLM judgment.
--   * Plan must be set at entry — the executor never re-derives it.
--   * JSONB so the plan can evolve (new rule families) without migrations.
--
-- Backward compat: existing OPEN rows get NULL exit_plan; the executor
-- falls back to journal_trades.stop + journal_trades.target as a minimal
-- plan (no scale-out, no trail, default DTE rules).

ALTER TABLE journal_trades
    ADD COLUMN IF NOT EXISTS exit_plan JSONB;

ALTER TABLE journal_trades
    ADD COLUMN IF NOT EXISTS scale_rungs_taken INT NOT NULL DEFAULT 0;

COMMENT ON COLUMN journal_trades.exit_plan IS
    'Deterministic exit plan set at entry; see ExitPlan pydantic schema '
    'and docs/position_management.md. NULL for legacy rows.';
COMMENT ON COLUMN journal_trades.scale_rungs_taken IS
    'Number of scale-out ladder rungs already executed on this trade. '
    'Incremented by route_exit_or_hold after a successful partial close. '
    'Read by hard_exit_decision so the same rung does not fire repeatedly.';

INSERT INTO schema_migrations (version, notes)
VALUES ('012_exit_plan',
        'Phase 3 — deterministic exit plan column on journal_trades')
ON CONFLICT (version) DO NOTHING;
