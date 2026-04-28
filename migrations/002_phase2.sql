-- Phase 2 v3 schema migration — Postgres flavor
-- Run via: ta-pg-migrate (which loads .env.postgres for the DSN)
-- Idempotent: every CREATE uses IF NOT EXISTS; every ALTER uses ADD COLUMN IF NOT EXISTS.

BEGIN;

-- =========================================================================
-- agent_events — append-only domain audit log
--
-- This is the source-of-truth for "what happened in the autonomous loop"
-- across regime / scout / research / risk / executor / learning. LangGraph
-- checkpointer covers state recovery, but this table covers human / replay
-- inspection of decision rationale.
-- =========================================================================
CREATE TABLE IF NOT EXISTS agent_events (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id          TEXT NOT NULL,
    trigger         TEXT NOT NULL,            -- premarket_scan | candidate_entry | intraday_monitor | eod_review | weekly_learning | healthcheck
    agent           TEXT NOT NULL,            -- e.g. regime_classifier, risk_arbiter, scout, trader_synth
    event_type      TEXT NOT NULL,            -- start, finish, llm_call, schema_violation, guardrail_breach, decision, etc.
    severity        SMALLINT NOT NULL DEFAULT 0,  -- 0=info, 1=warn, 2=error, 3=critical
    payload         JSONB NOT NULL DEFAULT '{}'::jsonb,
    cost_usd        NUMERIC(10, 4)            -- nullable; only set on llm_call rows
);
CREATE INDEX IF NOT EXISTS idx_agent_events_ts        ON agent_events(ts);
CREATE INDEX IF NOT EXISTS idx_agent_events_run       ON agent_events(run_id);
CREATE INDEX IF NOT EXISTS idx_agent_events_agent     ON agent_events(agent, ts);
CREATE INDEX IF NOT EXISTS idx_agent_events_severity  ON agent_events(severity, ts) WHERE severity > 0;

-- =========================================================================
-- regime_* — Regime Detection Agent state (§6)
-- =========================================================================
CREATE TABLE IF NOT EXISTS regime_model_versions (
    id                  SERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_type          TEXT NOT NULL,        -- 'hmm_gaussian_v1'
    feature_set_version TEXT NOT NULL,        -- e.g. 'v1.0' — increments when feature engineering changes
    train_start         DATE,
    train_end           DATE,
    params_json         JSONB NOT NULL,       -- HMM transition matrix, emission means/covars, calibration mapping
    metrics_json        JSONB,                -- training/validation metrics
    status              TEXT NOT NULL DEFAULT 'active'   -- active | retired | shadow
);
CREATE INDEX IF NOT EXISTS idx_regime_model_status ON regime_model_versions(status);

CREATE TABLE IF NOT EXISTS regime_feature_snapshots (
    id                  BIGSERIAL PRIMARY KEY,
    as_of               TIMESTAMPTZ NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_freshness    JSONB NOT NULL,       -- per-feature freshness in seconds
    features            JSONB NOT NULL,       -- numerical feature vector (vix, breadth_pct, spy_rv_20, etc.)
    data_quality        JSONB NOT NULL        -- {missing: [...], stale: [...], outlier: [...], degradation_level: 0|1|2}
);
CREATE INDEX IF NOT EXISTS idx_regime_features_asof ON regime_feature_snapshots(as_of DESC);

CREATE TABLE IF NOT EXISTS regime_states (
    id                      BIGSERIAL PRIMARY KEY,
    as_of                   TIMESTAMPTZ NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version_id        INTEGER REFERENCES regime_model_versions(id),
    feature_snapshot_id     BIGINT REFERENCES regime_feature_snapshots(id),
    label                   TEXT NOT NULL,    -- BULL_TREND | RANGE_LOW_VOL | VOLATILE_TRANSITION | BEAR_TREND | CRISIS
    prior_label             TEXT,
    pending_label           TEXT,
    confidence              NUMERIC(5, 4) NOT NULL,
    probabilities           JSONB NOT NULL,   -- {BULL_TREND: 0.65, RANGE_LOW_VOL: 0.22, ...}
    degradation_level       SMALLINT NOT NULL DEFAULT 0,
    crisis_flags            JSONB NOT NULL DEFAULT '[]'::jsonb,
    gate                    JSONB NOT NULL    -- {size_multiplier, allow_long_calls, allow_long_puts, ...}
);
CREATE INDEX IF NOT EXISTS idx_regime_states_asof  ON regime_states(as_of DESC);
CREATE INDEX IF NOT EXISTS idx_regime_states_label ON regime_states(label, as_of DESC);

CREATE TABLE IF NOT EXISTS regime_llm_reviews (
    id                  BIGSERIAL PRIMARY KEY,
    regime_state_id     BIGINT REFERENCES regime_states(id) ON DELETE CASCADE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trigger_reason      TEXT NOT NULL,        -- low_confidence | label_changed | data_warning | crisis_overlay_disagreement
    prompt_hash         TEXT NOT NULL,
    response            JSONB NOT NULL,       -- {review_label, confidence_adjustment, risk_notes, must_defer}
    applied_action      TEXT NOT NULL,        -- CONFIRM | DOWNGRADE_TO_TRANSITION | DOWNGRADE_TO_CRISIS | DEFER
    cost_usd            NUMERIC(10, 4)
);

-- =========================================================================
-- risk_* — Active Risk Agent state (§7)
-- =========================================================================
CREATE TABLE IF NOT EXISTS risk_snapshots (
    id                      BIGSERIAL PRIMARY KEY,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    regime_state_id         BIGINT REFERENCES regime_states(id),
    equity                  NUMERIC(14, 2) NOT NULL,
    open_positions          JSONB NOT NULL,      -- list of {symbol, qty, entry, stop, mark, delta, ...}
    correlations            JSONB NOT NULL,      -- pairwise corr matrix, 90-day window
    factor_exposures        JSONB NOT NULL,      -- {tech, energy, rates_sensitive, defensive, commodity}
    aggregate_greeks        JSONB NOT NULL,      -- {net_delta, gamma_2sigma, vega_10vol, theta}
    heat                    JSONB NOT NULL,      -- {portfolio_heat_pct, per_underlying}
    liquidity               JSONB NOT NULL,      -- {bid_ask_spread_avg, oi, fallback_to_mark_count}
    data_quality            JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_risk_snapshots_created ON risk_snapshots(created_at DESC);

CREATE TABLE IF NOT EXISTS risk_decisions (
    id                      BIGSERIAL PRIMARY KEY,
    risk_decision_id        TEXT NOT NULL UNIQUE,    -- ULID; referenced by orders + audit
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at              TIMESTAMPTZ NOT NULL,    -- decisions auto-expire (5 min default)
    proposal_id             TEXT NOT NULL,
    thesis_id               INTEGER,
    regime_state_id         BIGINT REFERENCES regime_states(id),
    risk_snapshot_id        BIGINT REFERENCES risk_snapshots(id),
    decision                TEXT NOT NULL,           -- APPROVE | DOWNSIZE | VETO | DEFER
    proposed                JSONB NOT NULL,
    approved                JSONB NOT NULL,          -- {qty, max_entry_price, stop, target}
    hard_violations         JSONB NOT NULL DEFAULT '[]'::jsonb,
    llm_reviews             JSONB,                   -- {conservative, opportunity, arbiter}
    reasons                 JSONB NOT NULL DEFAULT '[]'::jsonb,
    used_for_order_id       TEXT
);
CREATE INDEX IF NOT EXISTS idx_risk_decisions_proposal ON risk_decisions(proposal_id);
CREATE INDEX IF NOT EXISTS idx_risk_decisions_decision ON risk_decisions(decision, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_decisions_expires  ON risk_decisions(expires_at);

CREATE TABLE IF NOT EXISTS portfolio_marks (
    id              BIGSERIAL PRIMARY KEY,
    as_of           TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    equity          NUMERIC(14, 2) NOT NULL,
    cash            NUMERIC(14, 2),
    positions       JSONB NOT NULL,
    greeks          JSONB NOT NULL,
    exposures       JSONB NOT NULL,
    pnl             JSONB NOT NULL    -- {day, week, mtd, ytd}
);
CREATE INDEX IF NOT EXISTS idx_portfolio_marks_asof ON portfolio_marks(as_of DESC);

-- =========================================================================
-- param_versions + learning_* — Online Learning Loop (§8)
-- =========================================================================
CREATE TABLE IF NOT EXISTS param_versions (
    id                      SERIAL PRIMARY KEY,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by              TEXT NOT NULL,        -- seed | deterministic_optimizer | llm_critic | manual
    parent_version_id       INTEGER REFERENCES param_versions(id),
    status                  TEXT NOT NULL,        -- active | shadow | canary | promoted | rejected | retired | rolled_back
    scope                   TEXT NOT NULL,        -- global | regime | strategy
    scope_key               TEXT,
    traffic_pct             NUMERIC(5, 2) NOT NULL DEFAULT 0,
    params                  JSONB NOT NULL,
    bounds                  JSONB NOT NULL,
    rationale               TEXT,
    replay_metrics          JSONB,
    promoted_at             TIMESTAMPTZ,
    retired_at              TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_param_versions_status ON param_versions(status);
CREATE INDEX IF NOT EXISTS idx_param_versions_scope  ON param_versions(scope, scope_key);

CREATE TABLE IF NOT EXISTS learning_experiments (
    id                  BIGSERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    param_version_id    INTEGER REFERENCES param_versions(id),
    experiment_type     TEXT NOT NULL,            -- shadow_replay | canary | promotion
    start_at            TIMESTAMPTZ,
    end_at              TIMESTAMPTZ,
    status              TEXT NOT NULL,            -- running | promoted | rejected | rolled_back
    assignment_rule     TEXT NOT NULL,
    metrics             JSONB,
    stop_reason         TEXT
);

CREATE TABLE IF NOT EXISTS learning_assignments (
    id                              BIGSERIAL PRIMARY KEY,
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    proposal_id                     TEXT NOT NULL,
    ticker                          TEXT NOT NULL,
    strategy_label                  TEXT NOT NULL,
    baseline_param_version_id       INTEGER REFERENCES param_versions(id),
    assigned_param_version_id       INTEGER REFERENCES param_versions(id),
    assignment_bucket               INTEGER NOT NULL,    -- stable_hash(...) % 100
    reason                          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_learning_assignments_proposal ON learning_assignments(proposal_id);
CREATE INDEX IF NOT EXISTS idx_learning_assignments_param    ON learning_assignments(assigned_param_version_id);

CREATE TABLE IF NOT EXISTS trade_outcome_features (
    id                          BIGSERIAL PRIMARY KEY,
    trade_id                    INTEGER NOT NULL,        -- references journal trades.id (sqlite mirror)
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    entry_regime_state_id       BIGINT REFERENCES regime_states(id),
    exit_regime_state_id        BIGINT REFERENCES regime_states(id),
    risk_snapshot_id            BIGINT REFERENCES risk_snapshots(id),
    param_version_id            INTEGER REFERENCES param_versions(id),
    mae                         NUMERIC(10, 4),          -- max adverse excursion
    mfe                         NUMERIC(10, 4),          -- max favorable excursion
    realized_r                  NUMERIC(10, 4),
    holding_days                NUMERIC(8, 2),
    slippage_bps                NUMERIC(8, 2),
    option_iv_change            NUMERIC(10, 4),
    feature                     JSONB NOT NULL,
    label                       JSONB NOT NULL           -- {win: bool, w_hit, ...}
);
CREATE INDEX IF NOT EXISTS idx_trade_outcomes_trade   ON trade_outcome_features(trade_id);
CREATE INDEX IF NOT EXISTS idx_trade_outcomes_param   ON trade_outcome_features(param_version_id);
CREATE INDEX IF NOT EXISTS idx_trade_outcomes_regime  ON trade_outcome_features(entry_regime_state_id);

CREATE TABLE IF NOT EXISTS learning_events (
    id                  BIGSERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type          TEXT NOT NULL,        -- shadow_proposed | replay_passed | replay_rejected | canary_started | canary_disabled | promoted | rolled_back
    param_version_id    INTEGER REFERENCES param_versions(id),
    payload             JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_learning_events_param ON learning_events(param_version_id, created_at);

CREATE TABLE IF NOT EXISTS params_history (
    id                          BIGSERIAL PRIMARY KEY,
    archived_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    promoted_param_version_id   INTEGER REFERENCES param_versions(id),
    snapshot                    JSONB NOT NULL    -- full snapshot of all active params at the time of this promote
);

-- =========================================================================
-- journal_* — mirror of Phase 1 SQLite trader.db tables on Postgres so the
-- LangGraph orchestrator can query everything in one place. Bidirectional
-- sync handled by a separate job (Phase 2.1).
-- =========================================================================
CREATE TABLE IF NOT EXISTS journal_theses (
    id                  INTEGER PRIMARY KEY,    -- mirrors sqlite primary key (no SERIAL)
    created_at          TIMESTAMPTZ NOT NULL,
    ticker              TEXT NOT NULL,
    direction           TEXT NOT NULL,
    thesis_text         TEXT NOT NULL,
    invalidation        TEXT,
    timeframe           TEXT,
    return_pct          NUMERIC(8, 2),
    loss_pct            NUMERIC(8, 2),
    status              TEXT NOT NULL DEFAULT 'open',
    regime_state_id     BIGINT REFERENCES regime_states(id),
    synced_from_sqlite_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_journal_theses_ticker ON journal_theses(ticker, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_journal_theses_status ON journal_theses(status);

CREATE TABLE IF NOT EXISTS journal_trades (
    id                          INTEGER PRIMARY KEY,
    thesis_id                   INTEGER REFERENCES journal_theses(id),
    symbol                      TEXT NOT NULL,
    asset_type                  TEXT NOT NULL,
    side                        TEXT NOT NULL,
    qty                         NUMERIC(14, 4) NOT NULL,
    entry_price                 NUMERIC(14, 4) NOT NULL,
    exit_price                  NUMERIC(14, 4),
    stop                        NUMERIC(14, 4),
    target                      NUMERIC(14, 4),
    outcome                     TEXT NOT NULL DEFAULT 'OPEN',
    broker_order_id             TEXT,
    is_paper                    BOOLEAN NOT NULL DEFAULT TRUE,
    opened_at                   TIMESTAMPTZ NOT NULL,
    closed_at                   TIMESTAMPTZ,
    -- Phase 2 v3 additions
    proposal_id                 TEXT,
    risk_decision_id            TEXT,
    params_version_id           INTEGER REFERENCES param_versions(id),
    entry_regime_state_id       BIGINT REFERENCES regime_states(id),
    exit_regime_state_id        BIGINT REFERENCES regime_states(id),
    close_reason                TEXT,
    broker_fill_json            JSONB,
    synced_from_sqlite_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_journal_trades_symbol  ON journal_trades(symbol, opened_at DESC);
CREATE INDEX IF NOT EXISTS idx_journal_trades_outcome ON journal_trades(outcome);
CREATE INDEX IF NOT EXISTS idx_journal_trades_thesis  ON journal_trades(thesis_id);
CREATE INDEX IF NOT EXISTS idx_journal_trades_proposal ON journal_trades(proposal_id);

-- =========================================================================
-- migration metadata
-- =========================================================================
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes       TEXT
);

INSERT INTO schema_migrations (version, notes)
VALUES ('002_phase2',
        'Phase 2 v3 — agent_events + regime + risk + learning + journal mirror tables')
ON CONFLICT (version) DO NOTHING;

COMMIT;
