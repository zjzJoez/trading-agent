---
name: risk-management
description: Invalidation discipline, exit rules, and portfolio heat — how to manage risk on open paper positions.
---

# Risk management

Position sizing prevents ruin. Risk management prevents death-by-a-thousand-cuts. Separate concerns.

## Invalidation discipline

Every thesis has an `invalidation` field. It is a **commitment**, not a suggestion:

- If invalidation triggers during market hours: close the position or scale down. Don't rationalize.
- If invalidation triggers overnight (gap-through): close at open unless a new thesis can be filed.
- If you want to hold past invalidation, you must file a NEW thesis with updated invalidation. Don't silently move the goalpost.

Use `close_thesis(thesis_id, "triggered", ...)` to mark an invalidated thesis even if you haven't yet exited the trade — future `/enter` calls for the same ticker won't see it as a live thesis.

## Exit taxonomy

Trades close for one of four reasons. Tag them correctly on `close_trade`:

- **WIN**: target hit or thesis fully played out, >25bps positive.
- **LOSS**: stop hit or invalidation triggered, net negative.
- **SCRATCH**: exit within ±25bps; usually thesis-change or time-stop.
- **OPEN**: still live.

Time-stop rule of thumb: if the thesis had a "2-4 weeks" timeframe and 4 weeks pass with no target hit and no invalidation, SCRATCH the position. Capital re-deployed > dead money.

## Portfolio heat

Before opening a new position, mentally sum current heat:

```
heat = Σ  max(0, unrealized_loss_to_stop) over open positions
       (= per-position R1 contribution if stops were hit simultaneously)
```

The order guard caps R1 per trade at 2.5% equity and R2 caps concurrent opens at 6 (2026-05-22 threshold raises; src/trading_agent/sizing.py is canonical). Worst-case heat ≈ 15% equity if every stop hits at once — and short option legs now contribute assignment-stress heat too. Live with that number — if it feels too high, keep open count down or widen position stops.

## The "don't touch it" list

Paper account or not, these reflexes kill real accounts:

1. **Averaging down into a loss without a new thesis.** Hook allows it if sizing passes, but it almost always compounds the mistake. File a new thesis first, explaining *why* the second entry is different from the first.
2. **Moving stops further away.** If the original stop is hit, that's the trade. Don't escape-hatch by widening.
3. **Revenge trading.** After a loss, the `/research` step (with `search_past_trades`) is mandatory, not optional.
4. **Selling premium on the way down.** Cash-secured puts (R5b, full collateral) and stop-protected naked calls (R5c) are allowed; SELL-to-open without those protections is hard-blocked at the order tools. Don't invent workarounds — every short leg charges assignment-stress heat.
5. **Holding single-leg long options through earnings.** IV crush is a known lesson (see earnings-play skill).

## What the hook does NOT catch

The PreToolUse hook protects *entry*. Exits are up to you:

- No auto-stop-loss on the broker side (Moomoo paper API is "fire a limit, manage manually").
- No trailing stop. Use `/eod-review` to flag positions that have run past targets without a plan.
- No correlation check beyond GICS sector bucket.

## When a position goes wrong

1. Is invalidation triggered? → close, `close_thesis(..., "triggered")`, then immediately append a `post_mortem` note with the cause.
2. Is invalidation *not* triggered but the position is underwater? → hold per plan. Check your journal — a "2-4 week" thesis that's 3 days old is not failing yet.
3. Have you re-read the original thesis today? If yes, and you still believe it, hold. If no, do it before acting.
