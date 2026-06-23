-- Migration 014 — regime partial-downsize guard column + thesis_broken status
--
-- area C (exits) adds two deterministic exit branches to hard_exit_decision:
--
--   P0b  regime partial downsize — when the live regime label is in the plan's
--        downsize_50_on_labels, trim the position by 50%. This must fire AT
--        MOST ONCE per (trade, regime-label) episode, otherwise every 15-min
--        tick in a sustained degrade regime would keep halving the residual
--        until the position is gone. The guard is a dedicated column recording
--        the label the trade was last downsized at; the branch fires only when
--        regime_downsized_at_label != current label. (Scale-rung counters are
--        keyed to ladder index and can't express this, hence a new column.)
--
--   P0c  event rule — exit when the trade's thesis is flagged 'thesis_broken'.
--        journal_theses.status has no CHECK constraint, so 'thesis_broken' is
--        accepted without DDL; this migration only documents the new value in
--        the column comment. The flag is set out-of-band (journal-mcp
--        mark_thesis_broken, or a future premarket news node); the intraday
--        executor only READS it (no LLM on the hot path).

ALTER TABLE journal_trades
    ADD COLUMN IF NOT EXISTS regime_downsized_at_label TEXT;

COMMENT ON COLUMN journal_trades.regime_downsized_at_label IS
    'Regime label this OPEN trade was last partially downsized at (P0b in '
    'exits.hard_executor). NULL = never downsized. Prevents re-trimming every '
    'tick within one degrade-regime episode.';

COMMENT ON COLUMN journal_theses.status IS
    'open | triggered | invalidated | closed | thesis_broken. thesis_broken is '
    'set out-of-band (journal-mcp mark_thesis_broken / premarket news node) and '
    'fires the deterministic P0c event exit in exits.hard_executor.';

INSERT INTO schema_migrations (version, notes)
VALUES ('014_regime_downsize',
        'regime_downsized_at_label guard column + thesis_broken status (area C exits)')
ON CONFLICT (version) DO NOTHING;
