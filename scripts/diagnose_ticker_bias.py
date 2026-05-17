"""Diagnose where ticker-selection bias enters the trade pipeline.

If the system keeps trading SPY/QQQ when the watchlist also contains
NVDA/TSLA/etc., where in the pipeline is the bias introduced? Five
stages to inspect:

    1. WATCHLIST           — fixed list. SPY/QQQ here at all?
    2. SCOUT RANK          — does scout LLM rank SPY/QQQ above single names?
    3. PROPOSAL BUILT      — what ticker did trader_synthesizer pick?
    4. DECLINED            — why did the LLM decline single-name proposals?
    5. EXECUTED            — what's in journal_trades?

This script aggregates events at each stage and prints a per-ticker
funnel so you can see exactly where the bias is.

Usage:
    python -m scripts.diagnose_ticker_bias
    python -m scripts.diagnose_ticker_bias --days 30  # window
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from trading_agent.store.postgres import cursor

log = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser("diagnose_ticker_bias")
    p.add_argument("--days", type=int, default=60,
                   help="Look back this many days")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

    # ----- Stage 1: watchlist -----
    from trading_agent.jobs.premarket_watchlist import _get_watchlist
    watchlist = _get_watchlist()
    print("\n" + "=" * 72)
    print(f"STAGE 1 — WATCHLIST ({len(watchlist)} tickers)")
    print("=" * 72)
    for t in watchlist:
        print(f"  {t}")

    # ----- Stage 2: scout output (via agent_events) -----
    # Event name is `candidates_ranked` (emitted by rank_candidates node).
    # Payload has shape: { top: [ {ticker, score, reason}, ... ], top_ticker, n_candidates }
    with cursor() as cur:
        cur.execute(
            """
            SELECT ts, payload
            FROM agent_events
            WHERE event_type = 'candidates_ranked'
              AND ts >= %s
            ORDER BY ts DESC
            """,
            (cutoff,),
        )
        scan_rows = cur.fetchall()

    print(f"\n{'=' * 72}")
    print(f"STAGE 2 — SCOUT RANKINGS ({len(scan_rows)} candidates_ranked events in last {args.days}d)")
    print("=" * 72)
    scout_pick_first = Counter()  # ticker → times picked as top candidate
    scout_pick_topk = Counter()   # ticker → times appeared in top-N
    for ts, payload in scan_rows:
        candidates = (payload or {}).get("top", []) or \
                     (payload or {}).get("top_candidates", []) or \
                     (payload or {}).get("candidates", [])
        top_ticker = (payload or {}).get("top_ticker")
        if top_ticker:
            scout_pick_first[top_ticker.upper().replace("US.", "")] += 1
        for c in candidates:
            t = (c.get("ticker") if isinstance(c, dict) else c) or ""
            scout_pick_topk[t.upper().replace("US.", "")] += 1
    if scout_pick_first:
        print("  ticker | times-ranked-#1 | times-in-top-5")
        for t in watchlist + sorted(set(scout_pick_topk) - set(watchlist)):
            tu = t.upper().replace("US.", "")
            f = scout_pick_first.get(tu, 0)
            k = scout_pick_topk.get(tu, 0)
            if f or k:
                marker = " ← TOP" if f >= 2 else ""
                print(f"  {tu:8s} | {f:>3} | {k:>3}{marker}")
    else:
        print("  (no scout ranking events found — scout output not currently logged "
              "with parseable ticker list)")

    # ----- Stage 3: proposals built -----
    with cursor() as cur:
        cur.execute(
            """
            SELECT payload->>'ticker', payload->>'direction', payload->>'strategy', ts
            FROM agent_events
            WHERE event_type = 'proposal_built'
              AND ts >= %s
            ORDER BY ts DESC
            """,
            (cutoff,),
        )
        proposal_rows = cur.fetchall()

    proposal_counter = Counter()
    for ticker, direction, strategy, ts in proposal_rows:
        if ticker:
            proposal_counter[ticker.upper().replace("US.", "")] += 1

    print(f"\n{'=' * 72}")
    print(f"STAGE 3 — TRADER PROPOSALS BUILT ({len(proposal_rows)} in {args.days}d)")
    print("=" * 72)
    for t, n in proposal_counter.most_common():
        in_wl = "(in watchlist)" if t in {w.upper() for w in watchlist} else "(NOT IN WATCHLIST)"
        print(f"  {t:8s} | {n:>3} proposals  {in_wl}")

    # ----- Stage 4: declined proposals -----
    with cursor() as cur:
        cur.execute(
            """
            SELECT payload->>'ticker' AS t, payload->>'reason' AS reason
            FROM agent_events
            WHERE event_type = 'declined'
              AND ts >= %s
            ORDER BY ts DESC
            """,
            (cutoff,),
        )
        decline_rows = cur.fetchall()

    decline_counter = Counter()
    decline_reasons = defaultdict(list)
    for ticker, reason in decline_rows:
        if ticker:
            t = ticker.upper().replace("US.", "")
            decline_counter[t] += 1
            if reason:
                decline_reasons[t].append(reason[:140])

    print(f"\n{'=' * 72}")
    print(f"STAGE 4 — DECLINES ({len(decline_rows)} declined proposals in {args.days}d)")
    print("=" * 72)
    print("(LLM trader-synthesizer returned decline_to_trade=True)")
    for t, n in decline_counter.most_common():
        print(f"\n  {t:8s} | {n:>3} declines")
        for r in decline_reasons[t][:2]:
            print(f"    └─ {r}…")

    # ----- Stage 5: executed -----
    with cursor() as cur:
        cur.execute(
            """
            SELECT symbol, opened_at, side, qty, entry_price
            FROM journal_trades
            WHERE opened_at >= %s
            ORDER BY opened_at DESC
            """,
            (cutoff,),
        )
        trade_rows = cur.fetchall()

    print(f"\n{'=' * 72}")
    print(f"STAGE 5 — EXECUTED TRADES ({len(trade_rows)} in {args.days}d)")
    print("=" * 72)
    exec_counter = Counter()
    for symbol, opened_at, side, qty, entry_price in trade_rows:
        # symbol is like "US.SPY260605C715000" — extract underlying
        if symbol:
            parts = symbol.replace("US.", "").split(side or "")
            underlying = ""
            for ch in symbol.replace("US.", ""):
                if ch.isalpha():
                    underlying += ch
                else:
                    break
            exec_counter[underlying] += 1
            print(f"  {opened_at.date()} | {symbol} | qty={qty} | entry={entry_price}")
    if not trade_rows:
        print("  (no trades in window)")
    print()
    if exec_counter:
        print("  Underlying counts:")
        for t, n in exec_counter.most_common():
            print(f"    {t}: {n}")

    # ----- Funnel summary -----
    print(f"\n{'=' * 72}")
    print("FUNNEL: watchlist → proposed → declined → executed")
    print("=" * 72)
    all_tickers = sorted(set(watchlist) |
                         set(proposal_counter) |
                         set(decline_counter) |
                         set(exec_counter))
    print(f"  {'ticker':<8s} | {'in_wl':<5s} | {'prop':>4s} | {'decl':>4s} | {'exec':>4s}")
    print("  " + "-" * 50)
    for t in all_tickers:
        tu = t.upper()
        in_wl = "yes" if tu in {w.upper() for w in watchlist} else "no"
        prop = proposal_counter.get(tu, 0)
        decl = decline_counter.get(tu, 0)
        exec_n = exec_counter.get(tu, 0)
        if prop or decl or exec_n:
            print(f"  {tu:<8s} | {in_wl:<5s} | {prop:>4d} | {decl:>4d} | {exec_n:>4d}")

    # ----- Diagnosis hints -----
    print("\n" + "=" * 72)
    print("DIAGNOSIS HINTS")
    print("=" * 72)
    if proposal_counter:
        top_proposed = proposal_counter.most_common(1)[0]
        if top_proposed[0] in {"SPY", "QQQ", "IWM"} and top_proposed[1] > sum(proposal_counter.values()) * 0.5:
            print(f"  ⚠ Top proposed ticker is {top_proposed[0]} ({top_proposed[1]}/{sum(proposal_counter.values())}, "
                  f">{50}% of all proposals).")
            print(f"    Bias enters at scout/trader, BEFORE risk gates.")
            print(f"    Fix path: scout prompt + watchlist universe, NOT risk gates.")
    if decline_counter:
        top_declined = decline_counter.most_common(1)[0]
        print(f"  Top declined ticker: {top_declined[0]} ({top_declined[1]}/{sum(decline_counter.values())} declines).")
        if top_declined[0] in {"SPY", "QQQ", "IWM"}:
            print(f"    LLM trader keeps RECEIVING {top_declined[0]} and rejecting it — "
                  f"scout is feeding it the wrong universe.")
    if not proposal_counter:
        print("  No data in window — extend with --days or wait for more activity.")


if __name__ == "__main__":
    main()
