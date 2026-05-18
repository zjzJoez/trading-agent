-- Migration 011 — Universe snapshot on shadow proposals
--
-- WHY THIS MATTERS:
-- The dynamic watchlist (migration 010) means the LLM's selection universe
-- changes daily. When we later analyze shadow_proposals for LLM directional
-- skill ("did the LLM pick winners?"), we MUST be able to normalize by the
-- universe it picked from on each day. Without this column we'd be
-- conflating "is the LLM good at picking?" with "was today's universe good?".
--
-- The snapshot is captured at trader-synthesizer time (insert_shadow_proposal)
-- and stores:
--   * the full list of tickers that scout saw that day
--   * each ticker's tier (core vs rotating)
--   * the channel that surfaced it (screener / moomoo group / static)
--   * optionally each ticker's pre-scout liquidity stats
-- This lets a later analysis run something like:
--   "for proposals where universe size was 30-40 and target was rotating
--    tier, what's the LLM's 5d directional accuracy?"
--
-- JSONB shape (illustrative, not enforced):
--   {
--     "captured_at": "2026-05-19T13:30:00Z",
--     "universe_size": 38,
--     "tickers": [
--       {"t":"NVDA","tier":"core","source":"static_seed"},
--       {"t":"PLTR","tier":"rotating","source":"screener:most_actives"},
--       ...
--     ]
--   }

ALTER TABLE shadow_proposals
    ADD COLUMN IF NOT EXISTS universe_snapshot JSONB;

-- Partial index — only on rows where snapshot is captured (post-migration).
-- Useful for analysis queries like "all proposals where universe was small".
CREATE INDEX IF NOT EXISTS idx_shadow_proposals_universe
    ON shadow_proposals ((universe_snapshot->>'universe_size'))
    WHERE universe_snapshot IS NOT NULL;

INSERT INTO schema_migrations (version, applied_at)
VALUES ('011', NOW())
ON CONFLICT (version) DO NOTHING;
