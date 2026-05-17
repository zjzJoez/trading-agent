"""Overnight EDGAR scan — 02:00 UTC on EC2, before US pre-market.

Walks a configured watchlist, pulls recent 8-K/10-Q/4/13-F filings via
edgar-mcp, and writes one `notes` row per ticker with a summary of what's
new. Notes carry tags for `search_past_trades` to find them the next
morning when `/research TICKER` fires.

This is a "boring" job by design: no LLM calls, just SEC fetch + cache
+ DB write. Keeps the EC2 infra minimal and predictable.

Environment:
  DB_PATH            — path to trader.ec2.db
  WATCHLIST_TICKERS  — comma-separated, default AAPL,MSFT,NVDA,GOOGL,META,TSLA,AMZN,SPY,QQQ,IWM
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from trading_agent.config import CONFIG, ensure_dirs
from trading_agent.db import connection, migrate

# Keep aligned with src/trading_agent/jobs/premarket_watchlist.py
# (see that file for rationale on watchlist expansion 2026-05-17).
# SPY/QQQ/IWM excluded here since they don't file EDGAR docs (ETFs).
DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AMZN",
    "AVGO", "AMD", "ORCL", "ADBE", "CRM", "NFLX",
    "PLTR", "SMCI", "RDDT",
    "COIN", "MSTR",
    "UBER", "ABNB", "DUOL", "HOOD", "SHOP", "SOFI",
    "LLY", "MRNA",
    "XOM", "OXY",
    "MARA", "RIOT",
]

INTERESTING_FORMS = {"8-K", "10-Q", "10-K", "4", "13F-HR", "13D", "13G"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_watchlist() -> list[str]:
    env = os.environ.get("WATCHLIST_TICKERS", "").strip()
    if env:
        return [t.strip().upper() for t in env.split(",") if t.strip()]
    return DEFAULT_WATCHLIST


async def scan_ticker(edgar_mod, ticker: str) -> dict:
    """Pull last-48h filings for a ticker. edgar-mcp's get_recent_filings_for_ticker
    already caches and rate-limits, so calling it in a loop is fine."""
    call = edgar_mod.get_recent_filings_for_ticker(ticker=ticker, limit=10)
    # The edgar MCP tool may be async (httpx-backed) or sync; handle both.
    import inspect
    result = await call if inspect.isawaitable(call) else call
    filings = result.get("filings", result.get("rows", []))
    recent = []
    today = datetime.now(timezone.utc).date()
    for f in filings:
        form = f.get("form") or f.get("formType") or ""
        date_str = f.get("filingDate") or f.get("filing_date") or ""
        if not date_str:
            continue
        try:
            fdate = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        days_old = (today - fdate).days
        if days_old <= 2 and form in INTERESTING_FORMS:
            recent.append({"form": form, "date": date_str, "accession": f.get("accessionNumber")})
    return {"ticker": ticker, "recent_filings": recent}


def write_note(ticker: str, scan: dict) -> int | None:
    recent = scan.get("recent_filings", [])
    if not recent:
        return None
    forms = ",".join(sorted({f["form"] for f in recent}))
    summary = (
        f"Overnight scan {_now()[:10]} — {ticker} "
        f"{len(recent)} new filing(s): "
        + "; ".join(f"{f['form']} ({f['date']})" for f in recent)
    )
    # Use append_note WITHOUT embedding here — we're running on a box that
    # may not have sentence-transformers installed. Plain insert; Mac-side
    # post-sync job will backfill embeddings if desired.
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO notes (created_at, topic, text, tags, source) "
            "VALUES (?, ?, ?, ?, ?)",
            (_now(), f"overnight-scan-{ticker}", summary,
             f"overnight_scan,{ticker},{forms}", "research"),
        )
        conn.commit()
        return cur.lastrowid


def main() -> int:
    ensure_dirs()
    migrate(CONFIG.db_path)
    # Import edgar server as a module (not via MCP); job runs in-process so
    # we skip the stdio overhead.
    from trading_agent.mcp_servers.edgar import server as E  # noqa

    watchlist = _get_watchlist()
    summary = {
        "ts": _now(),
        "job": "overnight_edgar_scan",
        "watchlist": watchlist,
        "results": [],
        "errors": [],
    }
    # scan_ticker auto-detects awaitable vs sync, so just always run it through asyncio.
    async def _run_all():
        out = []
        errs = []
        skipped = []
        for t in watchlist:
            try:
                r = await scan_ticker(E, t)
                if r["recent_filings"]:
                    r["note_id"] = write_note(t, r)
                out.append(r)
            except ValueError as e:
                # Unknown ticker (mostly ETFs not in SEC company_tickers.json)
                # is expected and should not fail the job. Log + skip.
                if "not in SEC company_tickers" in str(e):
                    skipped.append({"ticker": t, "reason": "not in SEC ticker map"})
                else:
                    errs.append({"ticker": t, "error": repr(e),
                                 "tb": traceback.format_exc(limit=3)})
            except Exception as e:
                errs.append({"ticker": t, "error": repr(e),
                              "tb": traceback.format_exc(limit=3)})
        return out, errs, skipped

    results, errors, skipped = asyncio.run(_run_all())
    summary["results"] = results
    summary["errors"] = errors
    summary["skipped"] = skipped

    print(json.dumps(summary, default=str, indent=2))
    return 0 if not summary["errors"] else 1


def _sync_scan(edgar_mod, ticker: str) -> list[dict]:
    """Synchronous wrapper mirroring scan_ticker() for when the tool is sync."""
    result = edgar_mod.get_recent_filings_for_ticker(ticker=ticker, limit=10)
    filings = result.get("filings", result.get("rows", []))
    today = datetime.now(timezone.utc).date()
    recent = []
    for f in filings:
        form = f.get("form") or f.get("formType") or ""
        date_str = f.get("filingDate") or f.get("filing_date") or ""
        if not date_str:
            continue
        try:
            fdate = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        if (today - fdate).days <= 2 and form in INTERESTING_FORMS:
            recent.append({"form": form, "date": date_str, "accession": f.get("accessionNumber")})
    return recent


if __name__ == "__main__":
    sys.exit(main())
