---
name: trade-journal
description: Every paper order requires a thesis recorded in the last 10 minutes. Use journal-mcp before calling any moomoo order tool — the PreToolUse hook blocks otherwise.
---

# Trade journal — thesis-first discipline

The self-learning loop depends on every trade having a recorded reason. The PreToolUse hook enforces this: **no open thesis for the ticker in the last 10 minutes → order is blocked.**

MCP server: `journal-mcp`. DB tables: `theses`, `trades`, `notes`, `notes_vec`, `market_snapshots`, `post_mortems`.

## The required ordering (DO NOT SKIP)

```
1. mcp__journal-mcp__search_past_trades(query)     # context from similar past setups
2. mcp__journal-mcp__record_thesis(...)            # file thesis → get thesis_id
3. [sizing check — /size slash-command or sizing.py]
4. mcp__moomoo-mcp__place_paper_order(..., thesis_id=<id>)   # hook checks thesis + sizing
5. [posttool_fill_capture auto-writes trades + market_snapshots]
6. Later — close_trade(trade_id, exit_price, outcome, pnl)
```

If you place an order without step 2 the hook returns exit 2. If you file a thesis and then size+order 11 minutes later, thesis has expired (10-min freshness window) — re-file.

## Recording a thesis

Call `mcp__journal-mcp__record_thesis` with:

- `ticker`: bare symbol, UPPERCASE. "AAPL", not "US.AAPL".
- `direction`: one of LONG, SHORT, LONG_CALL, LONG_PUT, SHORT_CALL, SHORT_PUT, VERTICAL_CALL_DEBIT, VERTICAL_PUT_DEBIT.
- `thesis_text`: 2-5 sentences. Why does the trade work? What's the catalyst?
- `invalidation`: **concrete trigger** — a price level, a date, or a data release. Vague invalidations are useless. Good: "Close below 250 OR services guide cut in 10-Q". Bad: "if the trend reverses".
- `timeframe`: "2-4 weeks", "overnight", etc.
- `expected_return_pct` / `max_loss_pct`: rough plan.

## Recording a virtual fill (options fallback)

If `place_paper_option_order` returns `virtual_fill_suggested=True` (MoomooOpenD rejected the paper option leg), immediately call `mcp__journal-mcp__record_virtual_fill` with the **mid-price** of the contract at the time. The trade row is marked `broker_order_id=VIRTUAL-<uuid>` but is otherwise treated identically by post-mortem logic.

## Before any research / entry

Always run `mcp__journal-mcp__search_past_trades(query, k=5)` with:
- `query = ticker + " " + candidate strategy keywords`.

This pulls both semantic matches over `notes` (post-mortem lessons) and lexical matches over `trades.strategy_label` / `trades.symbol`. Ignoring past lessons is how you pay the same tuition twice.

## Closing a trade

`mcp__journal-mcp__close_trade(trade_id, exit_price, outcome, pnl)`:
- `outcome` ∈ {WIN, LOSS, SCRATCH}. SCRATCH for near-breakeven exits.
- `pnl` must be passed (the tool doesn't compute it — you know the commissions + multi-leg math better).

If invalidation triggers without you closing the position (e.g., gap-down through stop), also call `close_thesis(thesis_id, "triggered", note="<what happened>")` so future queries don't surface a stale thesis.

## Appending a lesson

`mcp__journal-mcp__append_note` with `source="post_mortem"` and `tags="lesson,<ticker>,<strategy_label>,<cause>"`. The text is auto-embedded into `notes_vec` and will surface on semantic search during future research.
