# Strategy specs

**`src/trading_agent/strategy_specs.py` is canonical — this doc mirrors it.**
If the two disagree, the code wins; fix the doc.

A strategy is admissible only if it is *falsifiable*: each spec declares its
expected shape (win-rate range, payoff target, friction-adjusted breakeven)
and the single observable result that kills it. Consumers:

- **sizing R7** — a strategy_label that maps to a spec is held to the spec's
  `min_risk_reward` (specs can only tighten the global 1.3 floor, never relax it).
- **journal post-mortem** — `generate_post_mortem_prompt` emits a
  `spec_comparison` block grading declared vs realized per strategy bucket.
- **moratorium language** — keyed on expectancy (`LB95(mean R) < 0` at
  `min_trades_for_eval`), never on win rate alone: low-WR/high-payoff is the
  *declared* shape of the convexity track.

Breakeven win rates are computed from `min_risk_reward` plus the execution-cost
model's friction (`execution_costs.py`) at typical premium — they are functions,
not constants, and move when `data/execution_costs.json` is recalibrated.

## Registry

| Spec | Status | Structure | Entry gates | Regimes | Expected WR | Breakeven WR (gross → net) | Falsification |
|---|---|---|---|---|---|---|---|
| `convexity_long_premium` | **shadow_only** | Single-leg long premium, debit-defined risk | DTE 21–45, \|delta\| 0.30–0.55, R:R ≥ 2.0, catalyst required in RANGE_LOW_VOL | BULL_TREND, RANGE_LOW_VOL | 30–45% (payoff target 2.5:1) | ~33% → ~39% | LB95(mean R) < 0 after 30 closed trades |
| `credit_put_spread_30_45` | **active** | Short put vertical, width-defined risk | DTE 30–45, short-leg \|delta\| 0.20–0.30, R:R ≈ 0.40 | BULL_TREND, RANGE_LOW_VOL | 70–80% (avg loss must stay < ~2.5× avg win net of friction) | ~71% → ~77% | LB95(mean R) < 0 after 30 closed trades, or realized WR < breakeven_wr_net |
| `credit_vertical_index_30_45` | **pending_prereqs** | Index (SPY/QQQ) credit vertical via `place_paper_option_combo`, managed 50%-PT / 21-DTE | DTE 30–45, short-leg \|delta\| 0.20–0.35, credit ≥ width/4, spread ≤ 5% of mid, news veto — **all placeholders pending M1-0.4 replay** | BULL_TREND, RANGE_LOW_VOL | — (managed-payoff profile from M1-0.4 replay, not expiry-binary formula) | — | Three-tier contract (plan M1-3): n=30 mechanical health check; n=60 kill-gate LB95(mean R) < −0.10R (entry-week block-bootstrap); promotion LB95 > 0 @ 97.5% one-sided |

### Status semantics (enforced)

- **active** — tradeable.
- **shadow_only** — retired from real fills (`convexity_long_premium`,
  operator-approved 2026-07-20, `docs/REVIVAL_PLAN_2026-07-20.md` sleeve 3).
  `build_trade_proposal` records in-band proposals to `shadow_proposals`
  (`final_action=SHADOW_ONLY`) plus a `shadow_proposal_recorded` agent event
  carrying the full proposal payload for option-level counterfactual replay;
  sizing refuses opens (`R_spec_status_not_tradeable`). Closes stay allowed.
- **pending_prereqs** — declared but blocked on prerequisites
  (`credit_vertical_index_30_45` until every M1-0 item is green); not
  tradeable through any consumer, no shadow book (`BLOCKED_SPEC_STATUS`).
- **blocked** — legacy name, same non-tradeable enforcement.

`credit_put_spread_30_45` is **active** (area A): a defined-risk vertical opens
via the `place_paper_option_combo` MCP tool, which proves the long leg caps the
short (R5e) and sizes the spread as ONE position off
`max_loss = (width − net_credit) × 100 × contracts`. The single-leg
SELL-to-open hard block (`R_short_option_open_blocked`) stays in force — only
provable verticals are unblocked, never a legged-in naked short. Known v1
limitations: portfolio heat over-counts the short leg at the broker-position
level (conservative), and a first-class combo-CLOSE path is a follow-up.

## Label mapping

`spec_for_label(label)`: labels starting `directional_` / `earnings_iv_drop` /
`pullback_` / `breakout_` → `convexity_long_premium`; `credit_put_spread*` →
`credit_put_spread_30_45`; `credit_vertical*` → `credit_vertical_index_30_45`;
exact spec names resolve directly; everything else → `None` (legacy, ungraded).
