-- Migration 008 — Shadow Proposal Log
--
-- Captures EVERY trader-synthesizer proposal, regardless of whether the
-- deterministic gate or LLM risk council subsequently blocked it. This
-- gives us a high-frequency record of what the LLM-driven decision
-- pipeline WOULD HAVE DONE in absence of risk guardrails.
--
-- The point: real `journal_trades` rows accumulate at ~1-3/week (gate +
-- council filter heavily). That's too sparse for any kind of statistical
-- alpha test. Shadow proposals accumulate at the rate the system runs
-- premarket_scan + intraday triggers — ~5-15 per day — so a 6-8-week
-- forward window can produce hundreds of decision points to compare
-- against actual market outcomes.
--
-- WHY THIS IS DIFFERENT FROM journal_theses:
--   - journal_theses only contains theses that became executed trades
--   - shadow_proposals contains EVERY proposal, including VETOED/DEFERRED
--   - This separation is intentional: shadow_proposals is for evaluating
--     "would the LLM have been right?", journal_theses is for tracking
--     real positions
--
-- Forward outcome is filled in by a daily cron (`evaluate_shadow_proposals.py`)
-- once enough trading days have elapsed past `proposal_ts`.

CREATE TABLE IF NOT EXISTS shadow_proposals (
    id                          BIGSERIAL PRIMARY KEY,
    proposal_ts                 TIMESTAMPTZ NOT NULL,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id                      TEXT NOT NULL,                  -- ties back to agent_events
    trigger                     TEXT NOT NULL,                  -- premarket_scan | intraday_5min | etc.
    -- LLM proposal (verbatim from trader-synthesizer)
    ticker                      TEXT NOT NULL,
    symbol                      TEXT,                           -- option code if applicable
    asset_type                  TEXT,                           -- OPT | STOCK
    direction                   TEXT,                           -- LONG | LONG_CALL | LONG_PUT | etc.
    qty_request                 NUMERIC,                        -- pre-sizing
    entry_price                 NUMERIC,
    stop                        NUMERIC,
    target                      NUMERIC,
    strategy_label              TEXT,
    expected_return_pct         NUMERIC,
    max_loss_pct                NUMERIC,
    proposal_json               JSONB NOT NULL,                 -- full TradeProposal blob
    -- Regime context at the moment of proposal (already in DB, just link)
    regime_state_id             BIGINT REFERENCES regime_states(id),
    regime_label                TEXT,
    regime_confidence           NUMERIC,
    -- Deterministic gate (regime_execution_gate)
    gate_decision               TEXT,                           -- ALLOW | DOWNSIZE | VETO | DEFER
    gate_size_factor            NUMERIC,                        -- 1.0 = unchanged, 0.5 = half size, 0 = blocked
    gate_block_reason           TEXT,
    -- Deterministic risk (R1-R6 + guardrails)
    deterministic_decision      TEXT,                           -- APPROVE | DOWNSIZE | VETO | DEFER
    deterministic_factor        NUMERIC,
    deterministic_reasons       JSONB,
    -- LLM risk council (if invoked)
    council_invoked             BOOLEAN NOT NULL DEFAULT FALSE,
    council_decision            TEXT,                           -- APPROVE | DOWNSIZE | VETO | DEFER
    council_size_factor         NUMERIC,
    council_reasons             JSONB,
    -- Final outcome
    final_action                TEXT NOT NULL DEFAULT 'pending', -- EXECUTED | VETOED_BY_GATE | VETOED_BY_RISK | VETOED_BY_COUNCIL | DEFERRED | DECLINED_BY_TRADER
    journal_thesis_id           BIGINT REFERENCES journal_theses(id),  -- if EXECUTED, points to the real thesis
    -- Forward outcome (filled by cron after N days)
    fwd_evaluated_at            TIMESTAMPTZ,
    fwd_1d_underlying_ret       NUMERIC,                        -- underlying SPY/ticker simple return
    fwd_5d_underlying_ret       NUMERIC,
    fwd_20d_underlying_ret      NUMERIC,
    fwd_directional_correct_5d  BOOLEAN,                        -- did LLM's direction match fwd_5d sign?
    fwd_directional_correct_20d BOOLEAN,
    fwd_eval_notes              JSONB DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_shadow_proposals_ts ON shadow_proposals(proposal_ts DESC);
CREATE INDEX IF NOT EXISTS idx_shadow_proposals_ticker ON shadow_proposals(ticker, proposal_ts DESC);
CREATE INDEX IF NOT EXISTS idx_shadow_proposals_pending_fwd
    ON shadow_proposals(proposal_ts) WHERE fwd_evaluated_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_shadow_proposals_run ON shadow_proposals(run_id);

INSERT INTO schema_migrations (version, applied_at)
VALUES ('008', NOW())
ON CONFLICT (version) DO NOTHING;
