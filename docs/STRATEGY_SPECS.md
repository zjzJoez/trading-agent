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
model's friction (`execution_costs.py`) — they are functions, not constants, and
move when `data/execution_costs.json` is recalibrated.

**Friction-truth fix, 2026-07-25.** `execution_costs.friction_r` used to proxy
BOTH legs of a vertical by the net credit (leg-mid-sum ≈ 2 × credit), which is
exact only at a wing ratio `r = long/short = 1/3`. Real stored quotes
(`option_chain_snapshots`, IWM PUTS 30–37 DTE, short |delta| 0.20–0.35, each
width from 3 distinct (day, expiry) chains) measure `r` median `0.731` / p75
`0.734` on $5-wides (n=32 pairs) and `0.530` / p75 `0.536` on $10-wides (n=25),
i.e. leg-mid-sum/credit of **6.43 / 3.26 versus the assumed 2.0** — the model
understated the vertical spread bill by ~3× on $5-wides. Verticals now price
through `friction_r` off real per-leg marks (`combo_friction_r` on the live
combo path, which also charges the leg's LIVE bid/ask when the order guard has
it) or the MEASURED wing ratio.

The blind fallback is an **upper quantile, not a median**: the modeled bill
`cr(1+r)/(1−r)` has derivative `2cr/(1−r)² > 0`, so it is strictly increasing in
`r` and a median understates friction for every structure above it — the exact
opposite of fail-conservative. `DEFAULT_WING_RATIO` is therefore the largest
measured p75, ratio selection is width-aware (`by_width` p75 at the exact width,
else the max over narrower measured widths), and any ratio resolved from the
calibration file is floored at the smallest ratio ever measured on a real chain.

Consequence: `credit_put_spread_30_45`'s net breakeven moved from ~77% to
**~87%**, i.e. ABOVE its own declared 70–80% envelope. Nothing was retuned to
hide that — see the spec's comment and
`tests/test_strategy_specs.py::test_vertical_spec_breakeven_now_exceeds_its_declared_wr_envelope`.

Calibration figures are only published with **independence**: both the
per-underlying block and each `by_width` block need at least `--min-chains`
(default 3) distinct (day, expiry) chains plus a per-width pair floor, or they
are recorded under `by_width_refused` and never persisted. Pairs off one chain
share one vol surface — counting them as a sample is how a single snapshot gets
acted on as if it described the name.

## Registry

| Spec | Status | Structure | Entry gates | Regimes | Expected WR | Breakeven WR (gross → net) | Falsification |
|---|---|---|---|---|---|---|---|
| `convexity_long_premium` | **shadow_only** | Single-leg long premium, debit-defined risk | DTE 21–45, \|delta\| 0.30–0.55, R:R ≥ 2.0, catalyst required in RANGE_LOW_VOL | BULL_TREND, RANGE_LOW_VOL | 30–45% (payoff target 2.5:1) | ~33% → ~39% | LB95(mean R) < 0 after 30 closed trades |
| `credit_put_spread_30_45` | **active** | Short put vertical, width-defined risk | DTE 30–45, short-leg \|delta\| 0.20–0.30, R:R ≈ 0.40 | BULL_TREND, RANGE_LOW_VOL | 70–80% (avg loss must stay < ~2.5× avg win net of friction) | ~71% → **~87%** (honest per-leg friction ~0.22R; the declared envelope no longer covers it — operator decision pending) | LB95(mean R) < 0 after 30 closed trades, or realized WR < breakeven_wr_net |
| `credit_vertical_index_30_45` | **pending_prereqs** | Index (SPY/QQQ) credit vertical via `place_paper_option_combo`, managed 50%-PT / 21-DTE | DTE 30–45, short-leg \|delta\| 0.20–0.35, credit ≥ width/4 (**unresolved placeholder** — real IWM quotes pay a credit/width MEDIAN of 0.191 at $5 wide / 0.176 at $10, i.e. below the floor; deliberately NOT retuned), spread ≤ 5% of mid, news veto — **all placeholders pending M1-0.4 replay** | BULL_TREND, RANGE_LOW_VOL | — (managed-payoff profile from M1-0.4 replay, not expiry-binary formula) | — | Three-tier contract (plan M1-3): n=30 mechanical health check; n=60 kill-gate LB95(mean R) < −0.10R (entry-week block-bootstrap); promotion LB95 > 0 @ 97.5% one-sided |

### Status semantics (enforced)

- **active** — tradeable.
- **shadow_only** — retired from real fills (`convexity_long_premium`,
  operator-approved 2026-07-20, `docs/REVIVAL_PLAN_2026-07-20.md` sleeve 3).
  `build_trade_proposal` records in-band proposals to `shadow_proposals`
  (`final_action=SHADOW_ONLY`) plus a `shadow_proposal_recorded` agent event
  carrying the full proposal payload — including a best-effort proposal-time
  `shadow_quote` (bid/ask/last, hang-proof fetch) since the EOD chain snapshot
  may not sample the exact contract — for option-level counterfactual replay;
  the EOD `cache_option_chains` job also force-includes the day's SHADOW_ONLY
  underlyings. Sizing refuses opens (`R_spec_status_not_tradeable`). Closes
  stay allowed.
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

**Unmapped labels fail closed on structure** (`spec_trading_block(...,
asset_type=...)`): `strategy_label` is free-text LLM output, and off-prefix
labels have shipped in production (2026-07-08 CRNX
`momentum-continuation-ITM-call`). Because single-leg SELL-to-open is
hard-blocked, an unmapped-label (or label-less) **single-leg OPT open is
exactly the retired long-premium structure** — it is governed by the
`convexity_long_premium` status: sizing refuses the open
(`R_spec_status_not_tradeable`) and `build_trade_proposal` records it
`SHADOW_ONLY` (with `label_unmapped: true` in the event payload) so the
counterfactual book is not biased toward on-prefix proposals. Unmapped STOCK
opens and the combo path (own label families + R5e structural proof) are
unaffected; closes are always exempt.
