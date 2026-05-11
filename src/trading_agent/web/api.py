"""FastAPI dashboard backend.

Single service that serves both:
  * Static UI under /  (HTML + CSS + JS)
  * Live JSON state under /api/*

Read-only by design (no order placement, no parameter writes). The
existing /halt endpoint stays in halt_endpoint.py — this dashboard
displays the halt state but does not toggle it.

Endpoint inventory:
    GET  /                       index.html
    GET  /static/{file}          css + js + favicon
    GET  /api/status             soak phase · regime label · halt state · timer summary
    GET  /api/pipeline           per-subgraph last-fire + event count
    GET  /api/positions          live open positions + thesis + unrealized PnL
    GET  /api/regime             current regime + 7d history
    GET  /api/risk               latest portfolio snapshot (greeks, factors, heat)
    GET  /api/events             recent agent_events (filterable by severity/agent)
    GET  /api/trades             closed trades (last N)
    GET  /api/learning           param versions + canary status
    GET  /api/quote/{symbol}     OHLC klines for the chart
    GET  /api/watchlist          tickers available in chart selector
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
HALT_FLAG = Path.home() / "trading-agent" / "data" / "halt.flag"

app = FastAPI(
    title="trading-agent dashboard",
    description="Read-only observability dashboard for the autonomous loop.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bare_ticker(code: str) -> str:
    """Strip 'US.' prefix + option-strike suffix → underlying ticker."""
    if not code:
        return ""
    s = code.split(".", 1)[-1]
    m = re.match(r"^([A-Z\.]+?)\d{6,8}[CP]\d+$", s)
    return m.group(1) if m else s


def _is_option(code: str) -> bool:
    return bool(re.search(r"\d{6,8}[CP]\d+$", code or ""))


def _format_option_label(code: str) -> str:
    """US.QQQ260605C665000 -> 'QQQ 6/5 665C'"""
    if not _is_option(code):
        return _bare_ticker(code)
    s = code.split(".", 1)[-1]
    m = re.match(r"^([A-Z\.]+?)(\d{2})(\d{2})(\d{2})([CP])(\d+)$", s)
    if not m:
        return code
    underlying, yy, mm, dd, cp, strike = m.groups()
    strike_int = int(strike) // 1000  # strike is in thousandths
    return f"{underlying} {int(mm)}/{int(dd)} {strike_int}{cp}"


def _ts_relative(ts: datetime | str | None) -> str:
    """Render a TZ-aware timestamp as 'Xm ago' / 'Xh ago' / 'Xd ago'."""
    if ts is None:
        return "—"
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return ts
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - ts
    s = int(delta.total_seconds())
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"


def _pg_cursor():
    """Lazy import — Postgres may be unavailable in dev environments."""
    from trading_agent.store.postgres import cursor as _cursor
    return _cursor()


def _moomoo():
    """Lazy import of moomoo functions."""
    from trading_agent.mcp_servers.moomoo import server as mm
    return mm


def _is_halted() -> bool:
    return HALT_FLAG.exists()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the dashboard SPA."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz")
async def healthz():
    """Liveness probe for systemd / load balancers."""
    return {"ok": True}


# ---------------------------------------------------------------------------
# /api/status — top-level summary
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def api_status():
    """Top banner: soak phase · regime · halt · service info."""
    soak = _read_soak()
    regime = _read_regime_latest()
    return {
        "soak_phase": soak.get("phase", "unknown"),
        "soak": soak,
        "regime_label": regime.get("label") if regime else "UNKNOWN",
        "regime_confidence": regime.get("confidence") if regime else None,
        "halted": _is_halted(),
        "server_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _read_soak() -> dict[str, Any]:
    try:
        from trading_agent.learning.soak import current_phase, soak_audit_dict
        p = current_phase()
        d = soak_audit_dict(p)
        d["phase"] = p.value
        return d
    except Exception as e:
        log.warning("soak read failed: %s", e)
        return {"phase": "unknown"}


def _read_regime_latest() -> Optional[dict[str, Any]]:
    try:
        with _pg_cursor() as cur:
            cur.execute(
                """
                SELECT label, confidence, probabilities, crisis_flags,
                       degradation_level, gate, created_at, as_of
                FROM regime_states
                ORDER BY as_of DESC LIMIT 1
                """,
            )
            r = cur.fetchone()
        if not r:
            return None
        return {
            "label": r[0],
            "confidence": float(r[1]) if r[1] is not None else None,
            "probabilities": r[2],
            "crisis_flags": r[3],
            "degradation_level": int(r[4]) if r[4] is not None else 0,
            "gate": r[5],
            "as_of": r[7].isoformat() if r[7] else None,
        }
    except Exception as e:
        log.warning("regime read failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# /api/pipeline — per-subgraph last-fire / event count
# ---------------------------------------------------------------------------

TRIGGERS = [
    "premarket_scan", "candidate_entry", "intraday_monitor",
    "eod_review", "weekly_learning", "healthcheck",
]


@app.get("/api/pipeline")
async def api_pipeline():
    """Per-subgraph last-fire and 24h event count."""
    rows: list[dict] = []
    try:
        with _pg_cursor() as cur:
            for trigger in TRIGGERS:
                cur.execute(
                    """
                    SELECT MAX(ts), COUNT(*),
                           SUM(CASE WHEN severity >= 2 THEN 1 ELSE 0 END)
                    FROM agent_events
                    WHERE trigger = %s AND ts > NOW() - INTERVAL '24 hours'
                    """,
                    (trigger,),
                )
                row = cur.fetchone()
                if row:
                    last_ts, n_events, n_errors = row
                    rows.append({
                        "trigger": trigger,
                        "last_ts": last_ts.isoformat() if last_ts else None,
                        "last_relative": _ts_relative(last_ts),
                        "events_24h": int(n_events or 0),
                        "errors_24h": int(n_errors or 0),
                    })
                else:
                    rows.append({
                        "trigger": trigger,
                        "last_ts": None,
                        "last_relative": "—",
                        "events_24h": 0,
                        "errors_24h": 0,
                    })
    except Exception as e:
        log.warning("pipeline read failed: %s", e)
        rows = [{"trigger": t, "last_ts": None, "last_relative": "—", "events_24h": 0, "errors_24h": 0} for t in TRIGGERS]
    return {"pipeline": rows}


# ---------------------------------------------------------------------------
# /api/positions — live open positions w/ thesis context
# ---------------------------------------------------------------------------

@app.get("/api/positions")
async def api_positions():
    """Live moomoo positions joined with journal thesis info + live quotes."""
    positions: list[dict] = []
    account = {"equity": 0.0, "cash": 0.0, "fetch_status": "ok"}

    # Step 1: live broker positions
    moomoo_rows: list[dict] = []
    try:
        mm = _moomoo()
        ai = mm.get_account_info()
        rows = ai.get("rows") or []
        if rows:
            r = rows[0]
            account["equity"] = float(r.get("total_assets") or 0)
            account["cash"] = float(r.get("cash") or 0)
        pl = mm.get_positions()
        moomoo_rows = pl.get("rows") or []
    except Exception as e:
        log.warning("moomoo positions failed: %s", e)
        account["fetch_status"] = f"error: {e}"

    # Step 2: journal trades for thesis context (Postgres canonical)
    journal_by_symbol: dict[str, dict] = {}
    try:
        with _pg_cursor() as cur:
            cur.execute(
                """
                SELECT t.id, t.symbol, t.thesis_id, t.opened_at, t.entry_price,
                       t.qty,
                       th.direction, th.thesis_text, th.invalidation,
                       th.return_pct, th.loss_pct
                FROM journal_trades t
                LEFT JOIN journal_theses th ON th.id = t.thesis_id
                WHERE t.outcome = 'OPEN'
                """,
            )
            for row in cur.fetchall():
                journal_by_symbol[row[1]] = {
                    "trade_id": int(row[0]),
                    "thesis_id": int(row[2]) if row[2] else None,
                    "opened_at": row[3].isoformat() if row[3] else None,
                    "age": _ts_relative(row[3]),
                    "entry_price": float(row[4]) if row[4] is not None else None,
                    "qty_journal": float(row[5]) if row[5] is not None else None,
                    "direction": row[6],
                    "thesis_text": (row[7] or "")[:300],
                    "invalidation": row[8],
                    "expected_return_pct": float(row[9]) if row[9] is not None else None,
                    "max_loss_pct": float(row[10]) if row[10] is not None else None,
                }
    except Exception as e:
        log.warning("journal trades read failed: %s", e)

    # Step 3: live quotes for marks + greeks (option rows include greeks)
    symbols_to_quote = [r.get("code") for r in moomoo_rows if r.get("code")]
    quote_by_symbol: dict[str, dict] = {}
    if symbols_to_quote:
        try:
            mm = _moomoo()
            q = mm.get_quote(symbols_to_quote)
            for r in (q.get("rows") or []):
                code = r.get("code") or r.get("symbol")
                if code:
                    quote_by_symbol[code] = r
        except Exception as e:
            log.warning("quote batch failed: %s", e)

    # Step 4: combine
    for r in moomoo_rows:
        code = r.get("code") or r.get("symbol") or ""
        qty = float(r.get("qty") or 0)
        broker_entry = float(r.get("cost_price") or r.get("avg_price") or 0)
        broker_mark = float(r.get("nominal_price") or r.get("current_price") or broker_entry)
        broker_pnl = float(r.get("pl_val") or 0.0)
        is_opt = _is_option(code)
        q = quote_by_symbol.get(code) or {}
        live_mark = float(q.get("last_price") or 0) or broker_mark
        meta = journal_by_symbol.get(code) or {}
        # Prefer journal entry_price for options — broker cost_price can be
        # affected by intra-position adjustments and may report negative.
        entry = float(meta.get("entry_price")) if meta.get("entry_price") else broker_entry
        mult = 100 if is_opt else 1
        live_pnl = (live_mark - entry) * qty * mult if entry > 0 else 0
        pct = ((live_mark - entry) / entry * 100) if entry > 0 else 0
        positions.append({
            "symbol": code,
            "label": _format_option_label(code),
            "underlying": _bare_ticker(code),
            "asset_type": "OPT" if is_opt else "STK",
            "qty": qty,
            "entry": entry,
            "mark": live_mark,
            "broker_pnl": broker_pnl,
            "unrealized_pnl": round(live_pnl, 2),
            "pnl_pct": round(pct, 2),
            "delta": float(q.get("delta")) if q.get("delta") is not None else None,
            "iv": float(q.get("imp_volatility")) if q.get("imp_volatility") is not None else None,
            "thesis": meta,
            "age": meta.get("age", "—"),
        })

    total_pnl = round(sum(p["unrealized_pnl"] for p in positions), 2)
    return {
        "account": account,
        "positions": positions,
        "total_unrealized_pnl": total_pnl,
        "count": len(positions),
    }


# ---------------------------------------------------------------------------
# /api/regime — current + 7d history
# ---------------------------------------------------------------------------

@app.get("/api/regime")
async def api_regime():
    history: list[dict] = []
    try:
        with _pg_cursor() as cur:
            cur.execute(
                """
                SELECT as_of, label, confidence, accuracy_label, spy_next_day_return
                FROM regime_states
                WHERE as_of > NOW() - INTERVAL '7 days'
                ORDER BY as_of DESC
                LIMIT 50
                """,
            )
            for row in cur.fetchall():
                history.append({
                    "ts": row[0].isoformat() if row[0] else None,
                    "label": row[1],
                    "confidence": float(row[2]) if row[2] is not None else None,
                    "accuracy_label": row[3],
                    "spy_next_day_return": float(row[4]) if row[4] is not None else None,
                })
    except Exception as e:
        log.warning("regime history read failed: %s", e)
    return {"current": _read_regime_latest(), "history": history}


# ---------------------------------------------------------------------------
# /api/risk — latest portfolio snapshot
# ---------------------------------------------------------------------------

@app.get("/api/risk")
async def api_risk():
    try:
        with _pg_cursor() as cur:
            cur.execute(
                """
                SELECT created_at, equity, aggregate_greeks, factor_exposures,
                       heat, data_quality, open_positions
                FROM risk_snapshots
                ORDER BY created_at DESC LIMIT 1
                """,
            )
            r = cur.fetchone()
        if not r:
            return {"snapshot": None}
        return {
            "snapshot": {
                "created_at": r[0].isoformat() if r[0] else None,
                "age": _ts_relative(r[0]),
                "equity": float(r[1]) if r[1] is not None else None,
                "aggregate_greeks": r[2],
                "factor_exposures": r[3],
                "heat": r[4],
                "data_quality": r[5],
                "n_positions": len(r[6] or []),
            }
        }
    except Exception as e:
        log.warning("risk read failed: %s", e)
        return {"snapshot": None}


# ---------------------------------------------------------------------------
# /api/events — recent agent activity
# ---------------------------------------------------------------------------

@app.get("/api/events")
async def api_events(
    limit: int = Query(50, ge=1, le=200),
    severity_min: int = Query(0, ge=0, le=3),
    agent: Optional[str] = Query(None),
):
    items: list[dict] = []
    try:
        with _pg_cursor() as cur:
            params: list = [severity_min]
            sql = """
                SELECT ts, run_id, trigger, agent, event_type, severity,
                       payload, cost_usd
                FROM agent_events
                WHERE severity >= %s
            """
            if agent:
                sql += " AND agent = %s"
                params.append(agent)
            sql += " ORDER BY ts DESC LIMIT %s"
            params.append(limit)
            cur.execute(sql, params)
            for row in cur.fetchall():
                payload = row[6]
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except Exception:
                        pass
                items.append({
                    "ts": row[0].isoformat() if row[0] else None,
                    "age": _ts_relative(row[0]),
                    "run_id": row[1],
                    "trigger": row[2],
                    "agent": row[3],
                    "event_type": row[4],
                    "severity": int(row[5] or 0),
                    "payload": payload,
                    "cost_usd": float(row[7]) if row[7] is not None else None,
                })
    except Exception as e:
        log.warning("events read failed: %s", e)
    return {"events": items}


# ---------------------------------------------------------------------------
# /api/trades — closed trade history
# ---------------------------------------------------------------------------

@app.get("/api/trades")
async def api_trades(limit: int = Query(20, ge=1, le=100)):
    trades: list[dict] = []
    try:
        with _pg_cursor() as cur:
            cur.execute(
                """
                SELECT t.id, t.symbol, t.asset_type, t.side, t.qty,
                       t.entry_price, t.exit_price, t.outcome,
                       t.opened_at, t.closed_at, t.close_reason,
                       th.direction, th.thesis_text
                FROM journal_trades t
                LEFT JOIN journal_theses th ON th.id = t.thesis_id
                WHERE t.outcome <> 'OPEN'
                ORDER BY t.closed_at DESC NULLS LAST
                LIMIT %s
                """,
                (limit,),
            )
            for row in cur.fetchall():
                entry = float(row[5]) if row[5] is not None else 0
                exit_ = float(row[6]) if row[6] is not None else 0
                qty = float(row[4]) if row[4] is not None else 0
                mult = 100 if row[2] == "OPT" else 1
                pnl = round((exit_ - entry) * qty * mult, 2) if entry > 0 else 0
                trades.append({
                    "trade_id": int(row[0]),
                    "symbol": row[1],
                    "label": _format_option_label(row[1]),
                    "asset_type": row[2],
                    "side": row[3],
                    "qty": qty,
                    "entry": entry,
                    "exit": exit_,
                    "outcome": row[7],
                    "opened_at": row[8].isoformat() if row[8] else None,
                    "closed_at": row[9].isoformat() if row[9] else None,
                    "close_reason": row[10],
                    "direction": row[11],
                    "thesis": (row[12] or "")[:200],
                    "pnl": pnl,
                })
    except Exception as e:
        log.warning("trades read failed: %s", e)
    # Aggregate stats
    n = len(trades)
    wins = sum(1 for t in trades if t["outcome"] == "WIN")
    pnl_total = round(sum(t["pnl"] for t in trades), 2)
    return {
        "trades": trades,
        "stats": {
            "n": n,
            "wins": wins,
            "losses": sum(1 for t in trades if t["outcome"] == "LOSS"),
            "win_rate": round(wins / n, 3) if n > 0 else 0,
            "pnl_total": pnl_total,
        },
    }


# ---------------------------------------------------------------------------
# /api/learning — param versions + canary status
# ---------------------------------------------------------------------------

@app.get("/api/learning")
async def api_learning():
    items: dict[str, Any] = {"active": None, "canaries": [], "recent_events": []}
    try:
        with _pg_cursor() as cur:
            cur.execute(
                """
                SELECT id, scope, scope_key, status, created_at, promoted_at,
                       traffic_pct, params, rationale, parent_version_id
                FROM param_versions
                WHERE status IN ('ACTIVE', 'CANARY', 'SHADOW')
                ORDER BY created_at DESC
                LIMIT 20
                """,
            )
            for row in cur.fetchall():
                v = {
                    "version_id": int(row[0]),
                    "scope": row[1],
                    "scope_key": row[2],
                    "status": row[3],
                    "created_at": row[4].isoformat() if row[4] else None,
                    "promoted_at": row[5].isoformat() if row[5] else None,
                    "traffic_pct": float(row[6]) if row[6] is not None else None,
                    "params": row[7],
                    "rationale": (row[8] or "")[:200],
                    "parent_version_id": int(row[9]) if row[9] else None,
                }
                if v["status"] == "ACTIVE" and items["active"] is None:
                    items["active"] = v
                elif v["status"] == "CANARY":
                    items["canaries"].append(v)

            cur.execute(
                """
                SELECT created_at, event_type, param_version_id, payload
                FROM learning_events
                ORDER BY created_at DESC LIMIT 10
                """,
            )
            for row in cur.fetchall():
                items["recent_events"].append({
                    "ts": row[0].isoformat() if row[0] else None,
                    "age": _ts_relative(row[0]),
                    "event_type": row[1],
                    "param_version_id": int(row[2]) if row[2] else None,
                    "payload": row[3],
                })
    except Exception as e:
        log.warning("learning read failed: %s", e)
    return items


# ---------------------------------------------------------------------------
# /api/quote/{symbol} — OHLC klines for the chart
# ---------------------------------------------------------------------------

DEFAULT_KLINE_DAYS = 90


@app.get("/api/quote/{symbol}")
async def api_quote(symbol: str, days: int = Query(DEFAULT_KLINE_DAYS, ge=5, le=365)):
    # Normalize to moomoo "US.XXX" format
    sym = symbol.strip().upper()
    if not sym.startswith("US.") and not _is_option(sym):
        sym = "US." + sym
    try:
        mm = _moomoo()
        end = date.today()
        start = end - timedelta(days=days)
        result = mm.get_historical_kline(
            symbol=sym, start=start.isoformat(), end=end.isoformat(),
            ktype="K_DAY", max_count=days + 10,
        )
        rows = result.get("rows") or []
        candles = []
        for r in rows:
            tk = r.get("time_key") or r.get("date") or ""
            ts = tk.split(" ")[0] if " " in tk else tk
            try:
                # lightweight-charts wants string YYYY-MM-DD or unix seconds
                candles.append({
                    "time": ts,
                    "open": float(r.get("open") or 0),
                    "high": float(r.get("high") or 0),
                    "low": float(r.get("low") or 0),
                    "close": float(r.get("close") or 0),
                    "volume": float(r.get("volume") or 0),
                })
            except Exception:
                continue
        # Also include current quote for the latest mark
        latest = None
        try:
            q = mm.get_quote([sym])
            qrows = q.get("rows") or []
            if qrows:
                r = qrows[0]
                latest = {
                    "last": float(r.get("last_price") or 0),
                    "change_pct": float(r.get("change_rate") or 0),
                    "volume": float(r.get("volume") or 0),
                    "high": float(r.get("high_price") or 0),
                    "low": float(r.get("low_price") or 0),
                }
        except Exception:
            pass
        return {"symbol": sym, "candles": candles, "latest": latest}
    except Exception as e:
        log.warning("quote read failed for %s: %s", symbol, e)
        raise HTTPException(status_code=503, detail=str(e))


# ---------------------------------------------------------------------------
# /api/watchlist — chart selector options
# ---------------------------------------------------------------------------

DEFAULT_WATCHLIST = [
    "SPY", "QQQ", "IWM", "DIA", "VIX",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
    "JPM", "XLE", "XLF", "GLD", "TLT",
]


@app.get("/api/watchlist")
async def api_watchlist():
    raw = os.environ.get("WATCHLIST_TICKERS", "")
    tickers = [t.strip() for t in raw.split(",") if t.strip()] or DEFAULT_WATCHLIST
    return {"tickers": tickers}


# ---------------------------------------------------------------------------
# /api/system — static system surface info (no DB hit)
# ---------------------------------------------------------------------------

@app.get("/api/system")
async def api_system():
    """Static system facts (line counts, role counts, etc.) — for the header."""
    try:
        from trading_agent.llm.roles import ROLE_CONFIG
        roles_count = len({r for r in ROLE_CONFIG if not r.endswith("_deepseek")})
    except Exception:
        roles_count = 14
    return {
        "roles_count": roles_count,
        "subgraphs_count": 6,
        "channels": ["claude_code", "codex", "deepseek"],
    }
