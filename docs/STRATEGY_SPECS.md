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
| `convexity_long_premium` | **active** | Single-leg long premium, debit-defined risk | DTE 21–45, \|delta\| 0.30–0.55, R:R ≥ 2.0, catalyst required in RANGE_LOW_VOL | BULL_TREND, RANGE_LOW_VOL | 30–45% (payoff target 2.5:1) | ~33% → ~39% | LB95(mean R) < 0 after 30 closed trades |
| `credit_put_spread_30_45` | **blocked** | Short put vertical, width-defined risk | DTE 30–45, short-leg \|delta\| 0.20–0.30, R:R ≈ 0.40 | BULL_TREND, RANGE_LOW_VOL | 70–80% (avg loss must stay < ~2.5× avg win net of friction) | ~71% → ~77% | LB95(mean R) < 0 after 30 closed trades, or realized WR < breakeven_wr_net |

`credit_put_spread_30_45` is blocked because SELL-to-open is hard-blocked at
the order tools (`R_short_option_open_blocked`) and multi-leg combos have no
atomic sizing. It is declared now so the high-WR track has a falsifiable
target the day that infrastructure lands.

## Label mapping

`spec_for_label(label)`: labels starting `directional_` / `earnings_iv_drop` /
`pullback_` / `breakout_` → `convexity_long_premium`; exact spec names resolve
directly; everything else → `None` (legacy, ungraded).
