---
description: Weekly review of closed trades — stats by strategy, 3-5 lessons embedded into notes for future research queries.
---

# /weekly-post-mortem

Run on Friday after close (or Saturday morning). Review the last 7 days of closed paper trades, compute aggregate stats, extract 3-5 lessons, and persist them.

## Procedure

**1. Fetch the data.**

- Compute `since_date` = today − 7 days (YYYY-MM-DD).
- Call `mcp__journal-mcp__generate_post_mortem_prompt(since_date=<that>)`. Returns:
  - `closed_trade_count`
  - `by_strategy`: {label: {wins, losses, scratches, total_pnl, avg_pnl, count}}
  - `trades`: list of closed trades with their theses.

If `closed_trade_count == 0`, say so and exit — nothing to review.

**2. Compute stats.**

For each strategy_label:

- Win rate = wins / (wins + losses + scratches)
- Avg P&L per trade
- Profit factor = Σ winning_pnl / |Σ losing_pnl|  (∞ if no losses)
- Expectancy = avg_win × win_rate − avg_loss × (1 − win_rate)

For each trade, compute thesis-vs-outcome drift:

- `expected_return_pct` vs actual return (`pnl / (entry × qty)`)
- Flag trades where drift > 50%

**3. Draft 3-5 lessons.**

Follow the `post-mortem` skill's "good lesson" format — each lesson has pattern / cause / rule-for-next-time.

Prioritize lessons from:
- Biggest losers (every trade with pnl < -0.5% equity deserves at least a note)
- Repeated strategy failures (strategy with win-rate < 40% over ≥ 3 trades)
- Thesis-vs-outcome drift outliers
- Surprising wins — what made them work?

**4. Persist.**

For each lesson, call:

```
mcp__journal-mcp__append_note(
  topic="lesson-<strategy>-<short-phrase>",
  text="<pattern>. <cause>. <rule for next time>.",
  tags="lesson,<ticker>,<strategy_label>,<cause-tag>",
  source="post_mortem",
)
```

(Optional) Record a `post_mortems` table row if desired — for MVP, the `notes` with `source="post_mortem"` are sufficient.

**5. Output the brief.**

```
# Weekly post-mortem — week ending <YYYY-MM-DD>

## Aggregate
- Closed trades: N
- Net P&L: $+/-X
- Win rate: X.X%
- Profit factor: X.XX

## By strategy

| strategy_label | count | wins | losses | scratches | win rate | avg P&L | profit factor |
|----------------|-------|------|--------|-----------|----------|---------|---------------|
| pullback_to_200DMA | 3 | 2 | 1 | 0 | 67% | +$42 | 2.1 |
| ...

## Biggest losers (review in detail)
1. <trade_id>: US.TSLA BUY at 200 → 190, LOSS $-500. Thesis: "...". Invalidation: "...". Why it failed: <your judgment>.
2. ...

## Lessons filed
1. [lesson-pullback_to_200DMA-volume-confirmation] Pullbacks to 200DMA without a volume spike on the first green day have failed 3 of 4 times this month. Rule: require volume > 1.3× 20-day avg on the reclaim day.
2. ...

## Strategy allocation going forward
- Keep running: pullback_to_200DMA (profit factor > 2).
- Pause: earnings_long_call (2 losses to IV crush in a week). Switch to earnings_directional_debit_spread or skip earnings entirely.
- Experiment: <...>
```

## Constraints

- Every lesson MUST be persisted via `append_note`. A post-mortem that doesn't write to the DB is wasted effort — lessons need to surface on future `search_past_trades` calls.
- 3-5 lessons is the target. More than 8 is noise; fewer than 3 means not enough closed trades — say so.
- Do NOT invent stats — if a strategy has 1 trade, say "insufficient data" rather than extrapolating.
- Do NOT suggest real-money trading or leverage changes. MVP is paper-only.
