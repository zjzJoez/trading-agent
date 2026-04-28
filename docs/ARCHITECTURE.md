# Architecture

> Module-by-module reference for `trading-agent`. Pairs with `README.md` (the
> elevator pitch) — this file is the engineering deep-dive.

## Topology

A single Ubuntu box runs the entire Phase 2 stack. Phase 1 still runs Mac-side
for manual override sessions; the two share the `mcp_servers/`, `sizing.py`,
and `hooks/` modules so the safety floor is identical in both modes.

```
┌──────────────────────── Single EC2 box (Ubuntu 22.04+, 8 GB RAM, 50 GB EBS) ───────────────────────┐
│                                                                                                    │
│  systemd units (all enabled at boot):                                                              │
│    trading-agent-opend.service    — Moomoo OpenD on :11111 (auto-login from .env.opend)            │
│    trading-agent-halt.service     — FastAPI /halt + /resume on :8443                               │
│    cloudflared.service            — Cloudflare Tunnel → halt.<your-domain>                         │
│    postgresql.service             — Postgres 14, 17 tables                                         │
│                                                                                                    │
│  systemd timers (4):                                                                               │
│    healthcheck   *:05  (hourly, silent when OK)                                                    │
│    premarket     Mon-Fri 12:30 UTC                                                                 │
│    intraday      Mon-Fri 13:00–20:45 UTC every 15 min + 20:50 UTC                                  │
│    eod           Mon-Fri 21:30 UTC                                                                 │
│    All timers ExecCondition test ! -f data/halt.flag → instant kill on iOS Shortcut                │
│                                                                                                    │
│  Job runner: trading-agent-brain@<trigger>.service                                                 │
│      → orchestrator.py → LangGraph subgraph for that trigger                                       │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Postgres schema (17 tables)

Migration: `migrations/002_phase2.sql`. Grouped by purpose:

| Group | Tables |
| --- | --- |
| Audit | `agent_events` |
| LangGraph checkpointer | `langgraph_checkpoints`, `langgraph_writes` |
| Regime | `regime_model_versions`, `regime_feature_snapshots`, `regime_states`, `regime_llm_reviews` |
| Risk | `risk_snapshots`, `risk_decisions`, `portfolio_marks` |
| Online learning | `param_versions`, `learning_experiments`, `learning_assignments`, `trade_outcome_features`, `learning_events`, `params_history` |
| Journal | `journal_theses`, `journal_trades` |
| Migrations | `schema_migrations` |

Foreign keys form a single audit chain:

```
journal_trades.thesis_id            → journal_theses.id
journal_trades.entry_regime_state_id → regime_states.id
journal_trades.risk_decision_id     ↔ risk_decisions.risk_decision_id (text join)
journal_trades.params_version_id    → param_versions.id
risk_decisions.regime_state_id      → regime_states.id
risk_decisions.risk_snapshot_id     → risk_snapshots.id
risk_snapshots.regime_state_id      → regime_states.id
trade_outcome_features.trade_id     → journal_trades.id
trade_outcome_features.param_version_id → param_versions.id
```

A single `SELECT … FROM journal_trades JOIN risk_decisions … JOIN regime_states …`
reproduces every input the system saw at decision time.

## LangGraph subgraphs

Five subgraphs, all Postgres-checkpointed:

```
premarket_scan_graph        12:30 UTC — regime classification + watchlist scan
candidate_entry_graph       per ticker — research → propose → size → risk → execute → persist
intraday_monitor_graph      every 15 min — refresh quotes + Greeks → exit triggers
eod_review_graph            21:30 UTC — reconcile fills, mark-to-market, enrich outcomes, digest
healthcheck_graph           hourly — opend + postgres + ntfy probes
```

Conditional edges:
- `regime=CRISIS` → `candidate_entry_graph` short-circuits to scan-only.
- `risk=VETO/DEFER` → executor skipped, persist VETO/DEFER reason.
- `budget>80%` → DEFER_NO_BUDGET state.
- `data/halt.flag` exists → entry subgraph refuses (belt-and-braces; the timer ExecCondition already short-circuits).

### `candidate_entry_graph` node sequence

```
load_active_params              → resolves param_versions ACTIVE row, snapshot to state
load_latest_regime              → reads regime_states (most recent)
load_open_positions             → moomoo get_account_info + get_positions
research_ticker                 → 4 analysts in 2 sequential pairs (Sonnet+Haiku, Haiku+GPT-5.5)
researcher_debate               → bull (Sonnet) + bear (GPT-5.5), 1–2 rounds
retrieve_past_lessons           → Postgres RAG over journal_trades for this ticker
build_trade_proposal            → trader-synthesizer (Opus 4.7) → typed TradeProposal
create_or_refresh_thesis        → INSERT INTO journal_theses
deterministic_sizing            → R1–R6, downsizes qty
shadow_track                    → Phase 2.6 counterfactual replay against SHADOW param_versions
regime_execution_gate           → multiplies qty by regime size_multiplier
active_risk_snapshot            → portfolio aggregates (correlation, factor, Greeks, heat)
deterministic_risk_guardrails   → per-regime tiered caps
maybe_risk_llm_council          → 3-step (Conservative / Opportunity / Arbiter)
finalize_risk_decision          → writes risk_decisions row
route_risk_decision (cond)      → APPROVE → executor; VETO → persist_veto; DEFER → persist_defer
execute_paper_order             → moomoo place_paper_option_order
capture_fill                    → polls get_orders for FILLED_ALL / CANCELLED_ALL up to 10 s
persist_trade_event             → INSERT INTO journal_trades with full FK chain
ntfy_trade_event                → push to ntfy "trades" topic
```

## Module guide

### `regime/`

```
regime/features.py     build_feature_snapshot — 17 features from 19 macro tickers via moomoo L1
regime/classifier.py   3-layer: crisis_overlay → Gaussian HMM (4 states) → rule-based fallback
regime/llm_review.py   real_llm_review — Claude Sonnet 4.6, may CONFIRM / DOWNGRADE / DEFER only
regime/gates.py        regime_size_multiplier + gate_trade_for_regime
regime/persist.py      writers for regime_feature_snapshots / regime_states / regime_llm_reviews
```

5 labels: `BULL_TREND` (mult 1.00) · `RANGE_LOW_VOL` (0.75) · `VOLATILE_TRANSITION` (0.50) ·
`BEAR_TREND` (0.50) · `CRISIS` (0.00, hard floor).

The HMM is not yet trained on labeled history — the rule-based fallback is the
active path. HMM training is a Phase 2.6.5 / 2.7 follow-up.

### `risk/`

```
risk/portfolio.py      PortfolioSnapshot — aggregate Greeks · factor exposures · corr matrix · heat
risk/guardrails.py     per-regime tiered caps (§7.3 of the design plan); DOWNSIZE/VETO/DEFER classifier
risk/agent.py          decide(RiskInput) → RiskOutput; deterministic first, LLM council on borderline
risk/llm.py            real_council_review — Sonnet → GPT-5.5 → Opus arbiter
risk/persist.py        risk_snapshots + risk_decisions writers (with 30-min expires_at)
```

Decision taxonomy:

- **APPROVE** — proposal passes all caps; LLM council not needed.
- **DOWNSIZE** — proposed qty exceeds a soft cap; agent returns smaller approved_qty.
- **VETO** — hard cap breach (correlation, regime, R1/R5 absolute); no path forward.
- **DEFER** — data quality issue, missing snapshot, council disagreement; retry next tick.

A DOWNSIZE that drops to zero contracts is automatically promoted to VETO.

### `learning/`

```
learning/params.py     19 mutable keys × 6 families · FROZEN_HARD_CAPS · ParamResolver · seed_baseline
learning/shadow.py     replay_sizing_caps · replay_crisis_overlay · run_shadows · learning_assignments
learning/outcome.py    OutcomeMetrics (realized_R, holding_days, slippage_bps, …) + enrich_closed_trades
learning/replay.py     fetch_closed_trades → replay_one → composite score = SR + IR − λ·MDD (λ=0.5)
```

Mutable surface (every key has min/max/default in `PARAM_BOUNDS`):

| Family | Keys | Bounds-summary |
| --- | --- | --- |
| `sizing_aggression` | `r1_soft_cap_pct`, `r5_soft_notional_pct`, `r3_ticker_exposure_pct` | conservative ≤ value ≤ Phase-1 hard cap |
| `stop_distances` | `implicit_stop_frac`, `option_premium_stop_pct`, `default_stop_atr_mult` | 0.5× ≤ value ≤ 2.0× current |
| `entry_filters` | `min_setup_quality`, `max_iv_percentile_long_premium`, `min_dte` | published ranges per family |
| `regime_thresholds` | `crisis_vix_level`, `crisis_vix_delta_5d`, `crisis_spy_5d_drop_pct`, `regime_confidence_transition`, `regime_confidence_llm_review` | inside §6.2 envelope |
| `candidate_count` | `max_candidates_per_scout` | 1 ≤ value ≤ 12 |
| `regime_size_multipliers` | `size_mult_bull_trend`, `size_mult_range_low_vol`, `size_mult_volatile_transition`, `size_mult_bear_trend` | non-CRISIS only |

Frozen (cannot be mutated by any `param_versions` row):

```
MAX_SINGLE_RISK_PCT · MAX_OPTION_NOTIONAL_PCT · MAX_CONCURRENT_OPENS
MAX_TICKER_EXPOSURE_PCT · MAX_SAME_SECTOR_OPENS · OPTION_DELTA_MIN
OPTION_DELTA_MAX · EARNINGS_LOCK_DAYS · size_mult_crisis · enable_real · trd_env
```

### `llm/`

```
llm/oauth_router.py    subprocess wrapper for `claude -p` + `codex exec`; max-2-concurrent semaphore;
                       schema retry-once; auth detection + StubLLMRouter fallback
llm/roles.py           16 roles → channel + model + agent file; degrade table
llm/budget.py          WeeklyBudget governor (reads agent_events.cost_usd, soft trigger at 95%)
llm/schemas.py         16 Pydantic v2 models with strict extra=forbid
```

Channels: Sonnet 4.6 (5 roles) · Haiku 4.5 (4) · Opus 4.7 (2) · GPT-5.5 (4) · Stub (used only when OAuth missing).

### `graph/`

```
graph/state.py         TradingGraphState TypedDict + sub-states (RegimeState, TradeProposal, RiskDecision)
graph/builder.py       5 compiled subgraphs; conditional edges on risk decision + halt-flag
graph/checkpointer.py  PostgresSaver wrapper with thread_id construction
graph/nodes/
  stubs.py             original Phase-2.1 stubs; replaced by real nodes one phase at a time
  trade_nodes.py       Phase 2.5 — load_*, research_ticker, build_trade_proposal, deterministic_sizing,
                       execute_paper_order, capture_fill, persist_trade_event, ntfy_trade_event,
                       researcher_debate, retrieve_past_lessons, create_or_refresh_thesis
  regime_nodes.py      Phase 2.2 — wraps regime/* into LangGraph nodes
  risk_nodes.py        Phase 2.3 — active_risk_snapshot, deterministic_risk_guardrails,
                       maybe_risk_llm_council, finalize_risk_decision, route_risk_decision
  learning_nodes.py    Phase 2.6 — load_active_params_node, shadow_track_node
  eod_learning.py      Phase 2.6 — enrich_outcomes_node (in eod_review_graph)
```

Every node returns a dict with the keys it owns; LangGraph's reducer merges
concurrent writes per top-level key.

### `notify/`

```
notify/ntfy.py         ntfy.sh notifier; HTTP title is RFC-2047-base64-encoded if it contains
                       non-Latin-1 chars; falls back to ASCII-only title otherwise
```

Five topics under one random 64-bit prefix: `trades` · `risk` · `ops` · `digest` · `learning`.
The prefix IS the auth — anyone with the topic name can subscribe / publish, so it's
generated per-deployment and stored in `~/trading-agent/.env`.

### `hooks/`

Same Phase-1 hooks, kept intact:

```
hooks/reject_real_env.py        L3 defense — pattern match on tool input JSON
hooks/pretool_order_guard.py    thesis freshness (10 min) + R1–R6 numerical re-validation
hooks/posttool_fill_capture.py  inserts trades + market_snapshots after a fill
```

Phase 2 extends `pretool_order_guard.py` to also accept the autonomous-order
path: requires a valid `risk_decision_id` referencing a non-expired
`risk_decisions` row; otherwise exit 2.

## Operational drills (verified)

- Kill `trading-agent-brain` mid-run → systemd restart → Postgres checkpoint resume on next tick.
- Stop OpenD → ops alert on `ops` ntfy topic; new entries blocked; exits queue and execute on restart.
- Postgres unavailable → `events.emit` falls back to `~/agent_events.fallback.jsonl`; no broker writes; entry subgraph DEFERs.
- `/halt` from iOS Shortcut → entry subgraph refuses within 1 timer tick; exits unaffected.
- ntfy delivery fails → fallback to filesystem log; Discord mirror configured but unused by default.
- Codex weekly cap reached → Bear / Opportunity / Critic auto-degrade to Claude Sonnet via the degrade table; ntfy `learning` channel notified.

## Sources synthesized

| Source | Adopted as |
| --- | --- |
| **QuantBook L11** | Three-tier hierarchy (Meta → Experts → Risk-with-veto) |
| **QuantBook L12, L13** | Five-state regime taxonomy with `VOLATILE_TRANSITION` + `CRISIS` |
| **QuantBook L14** | LLMs analyze/explain/flag — never size, set thresholds, or execute |
| **QuantBook L15** | Risk veto authority + append-only audit + safe-mode envelope |
| **QuantBook L16** | Portfolio agent ≠ risk agent — correlation, factor exposure, hidden leverage |
| **QuantBook L17** | Online learning as controlled evolution: shadow → canary → promote |
| **TradingAgents (Tauric)** | Fixed LangGraph topology, typed state, `thread_id` checkpointing, debate rounds |
| **TradingGroup (arxiv 2508.17565)** | Self-reflection prefix on Trader prompt |
| **QuantEvolve (arxiv 2510.18569)** | Diversity archive (one candidate per cell) — *Phase 2.6.5 deferred* |
| **R&D-Agent-Quant (arxiv 2505.15155)** | Spec → Synthesis → Implementation → Validation → Analysis loop |
| **FinCon** | Dual-level risk control |
| **wshobson/agents** | Tier abstraction in env, not hardcoded model strings |
| **Shannon** | Production gateway pattern (lighter LangGraph + Postgres equivalent) |
| **open-multi-agent** | Token budget caps + loop detection |

Explicitly rejected: WASI sandbox · goal-first DAG for execution · FinRL-DS RL training ·
LLM as primary regime classifier · LangGraph SQLite checkpointer on EC2 · code-level strategy evolution.
