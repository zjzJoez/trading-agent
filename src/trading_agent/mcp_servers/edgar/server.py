"""edgar-mcp: MCP server wrapping SEC EDGAR HTTP endpoints.

Design invariants:
- SEC requires a real User-Agent with contact email; anonymous UAs get blocked.
  We construct it from CONFIG.sec_user_agent (built from SEC_UA_EMAIL env).
- Rate limit: SEC publishes a 10 req/s hard cap. We stay well under with
  asyncio.Semaphore(8) + 0.12s post-request sleep (~8.3 req/s peak).
- Disk cache at data/filings_cache/ keyed by sha256(url). Filings never
  change after they're filed, so aggressive caching is free correctness.
- Ticker→CIK mapping is the critical lookup: every SEC API needs CIK, not
  ticker. We hydrate the map at first use from
  https://www.sec.gov/files/company_tickers.json (weekly refresh).
- All tools are async. The httpx.AsyncClient is created lazily at first
  tool call and reused for the server lifetime.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
from lxml import html as lxml_html
from mcp.server.fastmcp import FastMCP

from trading_agent.config import CONFIG, ensure_dirs

# --------- constants ---------

SEC_BASE = "https://www.sec.gov"
EDGAR_DATA = "https://data.sec.gov"
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
COMPANY_TICKERS_URL = f"{SEC_BASE}/files/company_tickers.json"

# Max concurrent SEC requests. SEC cap is 10 req/s; 8 keeps us safely under.
SEC_CONCURRENCY = 8
# Fixed delay after each request. 1/8 ≈ 0.125s; use 0.12 for a tiny margin.
SEC_POST_DELAY = 0.12
TICKER_MAP_TTL = timedelta(days=7)

# Cache header — used by the map fetch but not by individual filing fetches,
# since filings are immutable by nature.
DEFAULT_CACHE_DIR = CONFIG.filings_cache_dir

mcp = FastMCP("edgar-mcp")

_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()
_rate_sem = asyncio.Semaphore(SEC_CONCURRENCY)

# Ticker→CIK map is populated at first lookup.
_ticker_map: dict[str, str] | None = None
_ticker_map_lock = asyncio.Lock()


# --------- plumbing ---------

async def _client_singleton() -> httpx.AsyncClient:
    global _client
    if _client is None:
        async with _client_lock:
            if _client is None:
                _client = httpx.AsyncClient(
                    headers={
                        "User-Agent": CONFIG.sec_user_agent,
                        "Accept-Encoding": "gzip, deflate",
                    },
                    timeout=httpx.Timeout(30.0, connect=10.0),
                    follow_redirects=True,
                )
    return _client


def _cache_path(url: str, suffix: str = ".bin") -> Path:
    h = hashlib.sha256(url.encode()).hexdigest()[:32]
    ensure_dirs()
    return DEFAULT_CACHE_DIR / f"{h}{suffix}"


async def _get(url: str, *, cache: bool = True, suffix: str = ".bin") -> bytes:
    """GET with rate-limit, disk cache, SEC UA, and retry on 5xx.

    SEC's full-text-search (`efts.sec.gov`) throws transient 500s regularly;
    data.sec.gov/www.sec.gov are more stable but occasionally flake under
    load. Three attempts with exponential backoff handles both.
    """
    path = _cache_path(url, suffix=suffix)
    if cache and path.exists():
        return path.read_bytes()

    client = await _client_singleton()
    for attempt in range(3):
        async with _rate_sem:
            try:
                resp = await client.get(url)
            finally:
                # Always pay the post-delay, even on error, to stay friendly.
                await asyncio.sleep(SEC_POST_DELAY)
        if 500 <= resp.status_code < 600 and attempt < 2:
            # Backoff: 0.5s, 1.5s before the final attempt.
            await asyncio.sleep(0.5 * (3 ** attempt))
            continue
        # 2xx path + terminal 4xx / final-attempt 5xx all go through raise_for_status.
        resp.raise_for_status()
        body = resp.content
        if cache:
            path.write_bytes(body)
        return body
    # Unreachable: the loop either returns or raises.
    raise RuntimeError("unreachable: _get retry loop fell through")


async def _get_json(url: str, *, cache: bool = True) -> Any:
    body = await _get(url, cache=cache, suffix=".json")
    return json.loads(body.decode("utf-8"))


# --------- ticker → CIK ---------

def _ticker_map_path() -> Path:
    ensure_dirs()
    return DEFAULT_CACHE_DIR / "ticker_cik.json"


def _ticker_map_stale(path: Path) -> bool:
    if not path.exists():
        return True
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc
    )
    return age > TICKER_MAP_TTL


async def _load_ticker_map() -> dict[str, str]:
    """Return dict mapping upper-case ticker → zero-padded 10-digit CIK."""
    global _ticker_map
    if _ticker_map is not None:
        return _ticker_map

    async with _ticker_map_lock:
        if _ticker_map is not None:
            return _ticker_map

        path = _ticker_map_path()
        if _ticker_map_stale(path):
            # company_tickers.json is small (~500KB) — refresh in-place.
            data = await _get_json(COMPANY_TICKERS_URL, cache=False)
            # SEC shape: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
            mapping: dict[str, str] = {}
            for _, row in data.items():
                ticker = str(row["ticker"]).upper()
                cik = str(row["cik_str"]).zfill(10)
                mapping[ticker] = cik
            path.write_text(json.dumps(mapping))
        else:
            mapping = json.loads(path.read_text())
        _ticker_map = mapping
        return _ticker_map


async def _cik_for(ticker: str) -> str:
    m = await _load_ticker_map()
    t = ticker.strip().upper()
    if t not in m:
        raise ValueError(f"Unknown ticker {ticker!r} — not in SEC company_tickers.json")
    return m[t]


# --------- filings: the submissions endpoint ---------

async def _submissions(cik: str) -> dict:
    """The /submissions/CIK{cik}.json document lists a company's filings.

    SEC keeps only the most recent ~1000 filings inline; older ones are
    paginated into extra files. MVP: use the inline `recent` block only.
    That covers multiple years for a typical ticker and is what we need.
    """
    return await _get_json(f"{EDGAR_DATA}/submissions/CIK{cik}.json")


def _zip_filings(recent: dict) -> list[dict]:
    """Turn SEC's column-oriented `recent` dict into row dicts."""
    keys = [
        "accessionNumber", "filingDate", "reportDate", "form", "primaryDocument",
        "primaryDocDescription", "isXBRL", "isInlineXBRL",
    ]
    cols = [recent.get(k, []) for k in keys]
    n = min((len(c) for c in cols), default=0)
    rows = []
    for i in range(n):
        acc = cols[0][i]
        rows.append({
            "accession_number": acc,
            "filing_date": cols[1][i],
            "report_date": cols[2][i],
            "form": cols[3][i],
            "primary_document": cols[4][i],
            "primary_doc_description": cols[5][i],
            "is_xbrl": bool(cols[6][i]),
            "is_inline_xbrl": bool(cols[7][i]),
        })
    return rows


def _primary_doc_url(cik: str, accession: str, primary_doc: str) -> str:
    """SEC's filing-document URL pattern uses accession without dashes."""
    acc_nodash = accession.replace("-", "")
    # cik in the path is WITHOUT leading zeros
    cik_int = str(int(cik))
    return f"{SEC_BASE}/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary_doc}"


# --------- HTML → text helper for filing bodies ---------

_HTML_WS = re.compile(r"[ \t]+")
_HTML_BLANK_LINES = re.compile(r"\n\s*\n+")


def _html_to_text(body: bytes) -> str:
    """Strip HTML/XBRL chrome and return readable text. Best-effort only —
    10-K/Q filings are messy and some parsers yield junk. For analysis we
    usually only need the first N thousand chars of narrative anyway."""
    try:
        tree = lxml_html.fromstring(body)
        for tag in tree.xpath("//script | //style | //ix:header", namespaces={"ix": "http://www.xbrl.org/2013/inlineXBRL"}):
            tag.getparent().remove(tag)
        text = tree.text_content()
    except Exception:
        # Fall back to a crude regex strip.
        text = re.sub(rb"<[^>]+>", b" ", body).decode("utf-8", errors="replace")
    text = _HTML_WS.sub(" ", text)
    text = _HTML_BLANK_LINES.sub("\n\n", text)
    return text.strip()


# --------- tools ---------

@mcp.tool()
async def get_recent_filings_for_ticker(
    ticker: str,
    limit: int = 10,
    form_types: list[str] | None = None,
) -> dict:
    """Recent SEC filings for a ticker (newest first).

    `form_types`: optional allow-list like ["10-K","10-Q","8-K","4"]. When
    omitted, returns the raw recent feed across all form types. Results come
    from the company's /submissions endpoint (most-recent ~1000 filings).
    """
    cik = await _cik_for(ticker)
    sub = await _submissions(cik)
    recent = sub.get("filings", {}).get("recent", {})
    rows = _zip_filings(recent)
    if form_types:
        allowed = {f.upper() for f in form_types}
        rows = [r for r in rows if str(r["form"]).upper() in allowed]
    rows = rows[:limit]
    for r in rows:
        r["url"] = _primary_doc_url(cik, r["accession_number"], r["primary_document"])
    return {"ticker": ticker.upper(), "cik": cik, "count": len(rows), "rows": rows}


@mcp.tool()
async def search_filings(
    ticker: str,
    form_type: str,
    since: str | None = None,
    limit: int = 20,
) -> dict:
    """Filter a ticker's recent filings to a single form type (e.g. "8-K").

    `since` is ISO date YYYY-MM-DD — only return filings on/after this date.
    Thin wrapper over the /submissions feed; for form types outside the
    recent window, use `full_text_search`.
    """
    cik = await _cik_for(ticker)
    sub = await _submissions(cik)
    rows = _zip_filings(sub.get("filings", {}).get("recent", {}))
    ft = form_type.strip().upper()
    rows = [r for r in rows if str(r["form"]).upper() == ft]
    if since:
        rows = [r for r in rows if (r["filing_date"] or "") >= since]
    rows = rows[:limit]
    for r in rows:
        r["url"] = _primary_doc_url(cik, r["accession_number"], r["primary_document"])
    return {"ticker": ticker.upper(), "cik": cik, "form_type": ft, "count": len(rows), "rows": rows}


@mcp.tool()
async def get_filing_text(
    url: str,
    max_chars: int = 40_000,
) -> dict:
    """Fetch a filing primary document and return cleaned text.

    `url` is the full URL returned by `get_recent_filings_for_ticker` /
    `search_filings`. Output is HTML-stripped and capped at `max_chars` to
    protect context. For deep reads, reduce scope via ticker+form first and
    raise `max_chars` only when you know what you want to read.
    """
    body = await _get(url, cache=True, suffix=".html")
    text = _html_to_text(body)
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]
    return {"url": url, "truncated": truncated, "len_chars": len(text), "text": text}


@mcp.tool()
async def get_insider_transactions(
    ticker: str,
    since: str | None = None,
    limit: int = 30,
) -> dict:
    """Form 4 (insider transactions) for a ticker.

    Lists the filings; doesn't parse the XML detail. To read a specific
    transaction, pass its `url` to `get_filing_text`. Useful for spotting
    cluster buys/sells around earnings or strategic shifts.
    """
    cik = await _cik_for(ticker)
    sub = await _submissions(cik)
    rows = _zip_filings(sub.get("filings", {}).get("recent", {}))
    rows = [r for r in rows if str(r["form"]).upper() == "4"]
    if since:
        rows = [r for r in rows if (r["filing_date"] or "") >= since]
    rows = rows[:limit]
    for r in rows:
        r["url"] = _primary_doc_url(cik, r["accession_number"], r["primary_document"])
    return {"ticker": ticker.upper(), "cik": cik, "count": len(rows), "rows": rows}


@mcp.tool()
async def full_text_search(
    q: str,
    forms: list[str] | None = None,
    since: str | None = None,
    limit: int = 20,
) -> dict:
    """Full-text search across EDGAR.

    Use this for forms where the issuer isn't the filer — e.g. 13F-HR
    (filed by institutional holders, searching for a ticker / company name
    in the holdings) or SC 13G/D (5%+ ownership stakes). `forms` filters
    by form type; `since` is ISO date.

    Backed by https://efts.sec.gov/LATEST/search-index — same engine as
    the human EDGAR full-text search UI.
    """
    params: dict[str, Any] = {"q": q, "dateRange": "custom" if since else None}
    if since:
        params["startdt"] = since
        params["enddt"] = datetime.now(timezone.utc).date().isoformat()
    if forms:
        params["forms"] = ",".join(forms)
    # Drop None values.
    params = {k: v for k, v in params.items() if v is not None}
    qs = urlencode(params)
    url = f"{EDGAR_SEARCH}?{qs}" if qs else EDGAR_SEARCH
    data = await _get_json(url)
    hits = data.get("hits", {}).get("hits", [])[:limit]
    rows = []
    for h in hits:
        src = h.get("_source", {}) or {}
        adsh = src.get("adsh") or ""
        adsh_nodash = adsh.replace("-", "")
        ciks = src.get("ciks", [])
        cik_int = str(int(ciks[0])) if ciks else ""
        doc = (src.get("root_forms") or src.get("display_names") or [""])[0] if isinstance(
            src.get("root_forms"), list
        ) else ""
        rows.append({
            "accession_number": adsh,
            "form": src.get("form"),
            "filing_date": src.get("file_date") or src.get("filedAt"),
            "ciks": ciks,
            "display_names": src.get("display_names"),
            "doc_hint": doc,
            "base_dir": f"{SEC_BASE}/Archives/edgar/data/{cik_int}/{adsh_nodash}/" if cik_int and adsh_nodash else "",
        })
    total = data.get("hits", {}).get("total", {})
    return {"q": q, "total": total, "count": len(rows), "rows": rows}


# --------- entry point ---------

def main() -> None:
    # Surface config at startup so a misconfigured UA email is obvious in logs.
    print(
        f"[edgar-mcp] SEC UA={CONFIG.sec_user_agent!r} "
        f"cache_dir={CONFIG.filings_cache_dir}",
        file=sys.stderr,
    )
    mcp.run()


if __name__ == "__main__":
    main()
