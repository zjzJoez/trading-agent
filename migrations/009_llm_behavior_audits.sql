-- Migration 009 — LLM Behavior Audit Log
--
-- Every LLM call through `OAuthLLMRouter.call()` runs a small deterministic
-- audit on the returned (parsed) output. Violations are logged here.
--
-- Audit types:
--   parse_failure         — router returned parsed=None (LLM produced invalid JSON
--                           or hit transport error). Logged as info — caller may
--                           fall back to stub.
--   constraint_violation  — output parsed OK but violates a hard role constraint
--                           (e.g. risk_arbiter tried to upgrade VETO→APPROVE;
--                           trader_synthesizer emitted stop on wrong side of entry).
--                           Severity error.
--   role_compliance       — soft check (output missing expected content keywords).
--                           Severity warn.
--
-- The point: catch silent role drift. The LLM might pass the JSON schema and
-- still emit garbage (empty thesis, all-zero numbers, etc.). The hard
-- guardrails in risk/llm.py already SILENTLY OVERRIDE arbiter upgrades —
-- this audit log records HOW OFTEN the LLM tried to violate them, which
-- is the metric for "is the LLM drifting from role spec".

CREATE TABLE IF NOT EXISTS llm_behavior_audits (
    id                  BIGSERIAL PRIMARY KEY,
    audit_ts            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id              TEXT,                       -- ties back to agent_events
    role                TEXT NOT NULL,              -- bull_researcher, risk_arbiter, etc.
    audit_type          TEXT NOT NULL,              -- parse_failure | constraint_violation | role_compliance
    severity            TEXT NOT NULL,              -- info | warn | error
    violation_detail    JSONB NOT NULL,             -- what specifically failed
    raw_output_excerpt  TEXT,                       -- first ~500 chars of parsed output (or raw if parse_failure)
    model_name          TEXT,                       -- which model produced this
    cost_usd            NUMERIC(10, 4)              -- cost of the call that produced the violation
);
CREATE INDEX IF NOT EXISTS idx_llm_audits_role_ts
    ON llm_behavior_audits(role, audit_ts DESC);
CREATE INDEX IF NOT EXISTS idx_llm_audits_violation
    ON llm_behavior_audits(audit_type, severity, audit_ts DESC);
CREATE INDEX IF NOT EXISTS idx_llm_audits_run
    ON llm_behavior_audits(run_id);

INSERT INTO schema_migrations (version, applied_at)
VALUES ('009', NOW())
ON CONFLICT (version) DO NOTHING;
