---
name: earnings-play
description: Framework for trading around earnings — detecting the earnings lock (R6), picking IV-aware structures, and recording post-earnings lessons.
---

# Earnings plays

Earnings is the best-understood edge source on single names (vol expansion / crush, gap mean-reversion, post-earnings drift) AND the most-common source of catastrophic losses (directional wrong + IV crush).

## The earnings lock (R6)

Within **2 trading days of next earnings** (`earnings_dte ∈ [0, 2]`), the PreToolUse hook rejects any order whose `strategy_label` does not start with `earnings_`. Rationale: normal directional trades get blown up by IV crush on the print.

`earnings_dte` is not looked up automatically — you must pass it into the order tool. If you're using `/research` or `/enter`, check the earnings calendar manually (via EDGAR 10-Q/8-K dates or a public calendar) and pass it explicitly when the print is imminent.

## Strategy labels for earnings setups

Use these verbatim — post-mortem aggregates on the exact label:

- `earnings_directional_debit_spread` — long call/put + short higher-strike call / lower-strike put at the same expiry. Neutralizes most of vega; caps profit but caps loss too. The safer pre-print bet when you have direction conviction.
- `earnings_iv_drop` — buy ATM/slightly-ITM call/put *after* the print once IV collapses, betting on mean-reversion of the gap.
- `earnings_long_call` / `earnings_long_put` — single-leg long premium through the print. **Highest risk**: IV crush typically eats 30-50% of the debit regardless of direction. Only attempt when IV is already low *before* the print and the expected move is larger than historical.
- `earnings_post_drift` — held **after** the print to capture post-earnings announcement drift (PEAD). 1-3 week holding period on beats with strong guide.

## Pre-earnings checklist

Before filing an earnings thesis:

1. **Consensus + whisper**: what are analysts expecting? Any recent guide update in 8-K? (Use `edgar-mcp.get_recent_filings_for_ticker`.)
2. **Last 4 reactions**: did the stock go up on beats? Down on misses? Or is the reaction uncorrelated with the surprise?
3. **Implied move**: straddle price ÷ stock price. If 8% is priced in and the historical average is 4%, options are expensive — prefer spreads or skip.
4. **IV rank**: if IV rank > 80 pre-print, long premium is almost always a net loss due to crush. Either spread it or stay out.
5. **Sector context**: did peers report? Did they beat/miss? Earnings cycles cluster.
6. **Insider activity**: `edgar-mcp.get_insider_transactions` — Form 4 within 30 days of the print is a weak but real signal.

## Post-earnings playbook

After the print, new thesis + new trade. Do NOT carry a pre-print position through unless the original thesis explicitly stated "through earnings" (rare and usually a mistake).

1. **Gap beat + guide up** → consider `earnings_post_drift` long for 1-3 weeks.
2. **Gap miss + guide cut** → consider short stock or long puts. (Short premium — CSPs etc. — is currently hard-blocked at the order tools until multi-leg combos get atomic sizing.)
3. **In-line + muted reaction** → no edge; move on.
4. **Big gap + IV still elevated** → `earnings_iv_drop` after the first 1-2 days once IV normalizes.

## Known lesson (seed this into notes early)

Long single-leg calls through earnings on mega-caps (AAPL/MSFT/NVDA) historically **lose 25-35% of the debit to IV crush alone** when the direction is correct by 2%. Either size the trade assuming you lose 30% on the vol move, or use a spread. This lesson should be appended to `notes` with `source="post_mortem"` and tags including `earnings_iv_crush` so it surfaces on semantic search next time.

## Logging earnings trades

When you record a thesis for an earnings play:

- `timeframe` = "through_earnings" or "post_earnings_1-3w"
- `invalidation` = concrete trigger. Examples:
  - "pre-print: close below $250 OR implied move prices in > 10%"
  - "post-drift: close below 5-day SMA OR 15% adverse move"
- `max_loss_pct` — for single-leg long premium, set this equal to the full debit (you can lose 100%).

After the trade closes, run `/weekly-post-mortem` or manually `append_note` with tags `lesson,earnings,<ticker>,<strategy_label>` so the next research cycle picks it up.
