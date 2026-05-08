---
name: post-mortem
description: Weekly / monthly review of closed paper trades. Aggregates by strategy_label, extracts 3-8 lessons, writes them as embedded notes for future research queries.
---

# Post-mortem — the self-learning loop

This is the "long-term memory" side of the agent. Every Friday (or on demand via `/weekly-post-mortem`), run this skill to review closed trades, compute per-strategy stats, and emit **lessons** that will surface on future `search_past_trades` calls.

## Mechanism

1. `mcp__journal-mcp__generate_post_mortem_prompt(since_date)` returns:
   - `closed_trade_count`
   - `by_strategy`: {strategy_label: {wins, losses, scratches, total_pnl, avg_pnl, count}}
   - `trades`: list of closed trades joined with their theses (invalidation, expected return, actual outcome)
2. You produce a **summary** + **3-8 lessons** as the output of this skill.
3. For each lesson, call `mcp__journal-mcp__append_note`:
   - `topic`: "lesson-<ticker-or-strategy>-<short-phrase>"
   - `text`: the lesson itself (1-3 sentences, concrete, actionable)
   - `tags`: comma-separated, including `lesson` + involved tickers + strategy_labels + cause-tags
   - `source`: `"post_mortem"`
4. (Optional) record a row in `post_mortems` table summarizing the period: `period_start`, `period_end`, `summary`, `lessons_json` (JSON of the lessons).

The `notes_vec` virtual table auto-embeds the note text. Future `search_past_trades(query)` calls will find these lessons by semantic similarity.

## What makes a good lesson

Concrete, falsifiable, actionable. Each lesson should cite:

- **The pattern** (what happened, 1 sentence)
- **The cause** (why it happened, 1 sentence)
- **The rule for next time** (what to do / not do, 1 sentence)

### Good lesson
> "Holding long calls through AAPL earnings cost ~30% of the debit from IV crush on two separate trades. Both times the direction was right (+2-3% move on the print) but the vol collapse more than offset the delta gain. Next time: if holding through the print, use `earnings_directional_debit_spread` (caps profit but removes vega); otherwise close before the print."

Tags: `lesson,AAPL,earnings,earnings_long_call,earnings_iv_crush,vega`

### Bad lesson
> "Be more careful with earnings trades."

Too vague to retrieve or act on. Doesn't cite specific trades. Won't change behavior.

## Stats to compute (per `strategy_label`)

- **Win rate** = wins / (wins + losses + scratches)
- **Avg P&L per trade**
- **Profit factor** = sum(winning pnl) / |sum(losing pnl)|. > 1.5 is solid; > 2.0 is excellent.
- **Expectancy** = avg_win × win_rate − avg_loss × (1 − win_rate)
- **Thesis-vs-outcome drift**: for each trade, was the `expected_return_pct` within 25% of actual `pnl / (entry × qty)`? Drift > 50% across several trades suggests mis-calibrated expectations for that strategy.

## Thesis-quality review

Pick the 3 biggest losers of the period and re-read their `invalidation` field:

- Was the invalidation concrete (price / date / event)? If not, that's a lesson.
- Did invalidation trigger before you closed? If yes — lesson on exit discipline.
- Did you re-file the thesis before doubling down? If you doubled without a new thesis — lesson on averaging.

## Frequency

- **Weekly** (Friday post-close or Saturday morning): 7-day window, 3-5 lessons max. Quality over quantity.
- **Monthly** (first weekend of the month): 30-day window, deeper stats. 5-8 lessons, plus re-evaluate strategy allocation for the month ahead.
- **Ad-hoc after a large loss**: any trade > 1% drawdown gets its own mini post-mortem within 24h. One focused lesson.

## Seed lessons

When the journal is empty (Phase-1 first week), manually seed 3-5 well-known lessons before the first real trade:

1. Long single-leg calls through earnings on mega-caps (the IV crush lesson above).
2. First red candle back to 200DMA on heavy volume on strong names is a reliable buy setup in bull regimes.
3. Buying a breakout on the 3rd test of the range is higher-probability than buying the first breakout (which fades more often).
4. IV rank > 80 is a signal to sell vol, not buy it — but MVP doesn't sell vol, so treat it as "skip the trade".
5. When SPY closes below its 20DMA, reduce size on all new directional longs by half.

These go into `notes` with `source="post_mortem"`, `tags="lesson,seed,..."`. They become retrievable on the first real research query.
