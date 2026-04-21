---
description: End-of-day review — open positions vs theses, flag invalidations, check against today's close.
---

# /eod-review

Run after the US close. Walk every open paper position, re-read its thesis, check today's close against stop / invalidation / target, and flag anything stale.

## Procedure

**1. Pull the open book.**

- `mcp__journal-mcp__get_open_positions_with_thesis()` — returns every `trades.outcome='OPEN'` row joined with its thesis.
- `mcp__moomoo-mcp__get_positions()` — broker's view.
- Reconcile. If the journal says 3 opens and broker says 2, flag it — one trade is either not yet journaled or was closed without `close_trade()`.

**2. Current quotes.**

- Batch `mcp__moomoo-mcp__get_quote` on every open symbol (stocks + options). Build a per-position table.

**3. Per-position review.**

For each open position, compute and display:

| Field | Source |
|---|---|
| Symbol | journal |
| Side, qty | journal |
| Entry, stop, target | journal |
| Current price | quote |
| Open P&L $ | (current − entry) × qty (sign flipped for SHORT) |
| Open P&L % | relative to entry |
| Days open | today − opened_at |
| Thesis age | today − thesis.created_at |
| Invalidation | thesis.invalidation |
| Invalidation status | **🟢 clear / 🟡 approaching / 🔴 triggered** (your judgment) |

**Invalidation status rules (apply best-effort):**

- 🔴 **triggered**: current price closed past the `invalidation` level (if it's a price level), or the named event has happened (if it's a date/event). Example: invalidation "close below $250" and today's close is $249.50 → triggered.
- 🟡 **approaching**: within 25bps of a price-level invalidation, or within 2 trading days of a date-based trigger.
- 🟢 **clear**: otherwise.

If the invalidation text is vague, flag 🟡 "vague invalidation — consider re-filing" and move on.

**4. Summarize.**

```
# EOD review — <YYYY-MM-DD>

Opens: N (journal) / M (broker) — [mismatch if N != M]
Unrealized P&L: $+/-X (across all opens)

## Positions

| sym | side | qty | entry | stop | current | %chg | days | inv | status |
|-----|------|-----|-------|------|---------|------|------|-----|--------|
| US.AAPL | BUY | 10 | 266.00 | 258.00 | 271.50 | +2.1% | 4 | close < 250 | 🟢 |
| US.TSLA | BUY | 5 | 200.00 | 195.00 | 194.00 | -3.0% | 9 | close < 195 | 🔴 |
| ...

## Action items

- 🔴 US.TSLA: invalidation triggered at close. Recommend close next session. Run:
    mcp__journal-mcp__close_thesis(thesis_id=<id>, status='triggered', note='close below 195 at EOD')
    then close_trade(trade_id=<id>, exit_price=<market>, outcome='LOSS', pnl=<calc>)

- 🟡 US.GOOGL: approaching invalidation ($245 close, inv $244.50). Monitor tomorrow open.

- ⚠️  US.NVDA: 21 days open, thesis timeframe was "2 weeks". Consider SCRATCH if no further move.
```

**5. Macro tape context.**

One-line regime note: SPY close vs 20DMA, QQQ vs 20DMA, VIX level. This frames the next-day plan.

## Constraints

- `/eod-review` does NOT place any orders and does NOT close any positions. It surfaces action items only.
- If the user says "execute the action items," that's a separate turn — close the triggered positions one at a time with explicit confirmation.
- If a journaled open is missing from the broker's position list, the trade may have been closed externally — prompt the user to run `close_trade(...)` manually.
