"""Premarket watchlist prep — 12:00 UTC (08:00 ET) on EC2.

Assembles a daily `premarket_brief` note per-ticker capturing yesterday's
filings (from overnight_edgar_scan), any Form 4 insider transactions in the
trailing 30 days, and any notes tagged `lesson,<ticker>` — so when the user
runs `/research TICKER` after open, the context is already staged.

Design: pure DB reads + one EDGAR call per ticker for insider transactions.
No OpenD access (EC2 doesn't have it). All output is written to trader.ec2.db.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_agent.config import CONFIG, ensure_dirs
from trading_agent.db import connection, migrate

DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AMZN",
    "SPY", "QQQ", "IWM",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_watchlist() -> list[str]:
    env = os.environ.get("WATCHLIST_TICKERS", "").strip()
    if env:
        return [t.strip().upper() for t in env.split(",") if t.strip()]
    return DEFAULT_WATCHLIST


def fetch_recent_overnight(ticker: str) -> list[dict]:
    """Last 48h of overnight_scan notes for this ticker."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    with connection() as conn:
        rows = conn.execute(
            "SELECT id, created_at, topic, text, tags FROM notes "
            "WHERE tags LIKE ? AND created_at >= ? "
            "ORDER BY created_at DESC",
            (f"%overnight_scan,{ticker}%", cutoff),
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_lessons(ticker: str) -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT id, topic, text, tags FROM notes "
            "WHERE source = 'post_mortem' AND (tags LIKE ? OR topic LIKE ?) "
            "ORDER BY id DESC LIMIT 10",
            (f"%,{ticker},%", f"%{ticker}%"),
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_insiders(edgar_mod, ticker: str) -> list[dict]:
    try:
        import asyncio, inspect
        call = edgar_mod.get_insider_transactions(ticker=ticker, limit=20)
        if inspect.isawaitable(call):
            res = asyncio.run(call)
        else:
            res = call
        rows = res.get("transactions", res.get("rows", []))
        # Trim trailing-30-day window.
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=30)
        recent = []
        for r in rows:
            date_str = r.get("transactionDate") or r.get("filedAt", "")[:10]
            try:
                d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
            except Exception:
                continue
            if d >= cutoff:
                recent.append({
                    "date": date_str,
                    "name": r.get("reportingOwnerName") or r.get("owner") or "?",
                    "txn": r.get("transactionType") or r.get("txn_type"),
                    "shares": r.get("transactionShares") or r.get("shares"),
                })
        return recent
    except Exception as e:
        return [{"error": repr(e)}]


def write_brief(ticker: str, overnight: list[dict], lessons: list[dict],
                insiders: list[dict]) -> int:
    parts = [f"Premarket brief {_now()[:10]} — {ticker}"]
    if overnight:
        parts.append(f"OVERNIGHT: {len(overnight)} filing note(s)")
        for n in overnight[:3]:
            parts.append(f"  • {n['text'][:200]}")
    if insiders:
        parts.append(f"INSIDER: {len(insiders)} txn(s) trailing 30d")
        for t in insiders[:5]:
            if "error" in t:
                continue
            parts.append(f"  • {t['date']} {t['name']} {t['txn']} "
                          f"shares={t['shares']}")
    if lessons:
        parts.append(f"LESSONS: {len(lessons)} prior lesson(s) matching")
        for l in lessons[:3]:
            parts.append(f"  • [{l['topic']}] {l['text'][:180]}")
    text = "\n".join(parts)
    tags = f"premarket_brief,{ticker}"
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO notes (created_at, topic, text, tags, source) "
            "VALUES (?, ?, ?, ?, ?)",
            (_now(), f"premarket-{ticker}", text, tags, "research"),
        )
        conn.commit()
        return cur.lastrowid


def main() -> int:
    ensure_dirs()
    migrate(CONFIG.db_path)
    from trading_agent.mcp_servers.edgar import server as E

    watchlist = _get_watchlist()
    summary = {"ts": _now(), "job": "premarket_watchlist",
                "watchlist": watchlist, "briefs": [], "errors": []}
    for t in watchlist:
        try:
            overnight = fetch_recent_overnight(t)
            lessons = fetch_lessons(t)
            insiders = fetch_insiders(E, t)
            nid = write_brief(t, overnight, lessons, insiders)
            summary["briefs"].append({
                "ticker": t, "note_id": nid,
                "overnight": len(overnight),
                "lessons": len(lessons),
                "insiders_30d": len([i for i in insiders if "error" not in i]),
            })
        except Exception as e:
            summary["errors"].append({"ticker": t, "error": repr(e),
                                       "tb": traceback.format_exc(limit=3)})
    print(json.dumps(summary, default=str, indent=2))
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
