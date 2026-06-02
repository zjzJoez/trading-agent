"""Refresh the dynamic watchlist — sources + filter + evict + cap.

Runs once per trading day before the premarket scan (typically 07:30 ET via
systemd timer). Read-only HTTP to yfinance + moomoo; writes only to Postgres
`watchlist_members` + `watchlist_audit`.

PIPELINE:
    1. Read current membership (watchlist_members)
    2. Read held positions (journal_trades OPEN + broker) — these are NEVER evicted
    3. Fetch candidate sources:
         a. moomoo CUSTOM groups (MGGA7, 芯片, 商品, 指数) — user-curated
         b. yfinance predefined screeners (day_gainers / day_losers / most_actives)
    4. Apply liquidity floor:
         market_cap >= $2B, avg_volume_3m >= 1M, US common stock (no OTC)
    5. EVICT pass (rotating only):
         tier='rotating' AND added_at < NOW-14d
            AND (last_scout_score IS NULL OR < 0.4)
            AND (last_proposed_at IS NULL OR < NOW-30d)
            AND ticker NOT IN held_positions
    6. ADD pass:
         For each accepted candidate ticker:
            if not in DB: INSERT (tier='rotating', source=<channel>)
            else: UPDATE last_seen_at, refresh metadata
    7. CAP pass:
         If membership > 40, evict the lowest add_score rotating members
         (add_score derived from metadata.gap_pct * metadata.dollar_volume).
    8. Audit each ADD/EVICT/TOUCH to watchlist_audit.

USAGE:
    python -m trading_agent.scripts.refresh_dynamic_watchlist          # apply
    python -m trading_agent.scripts.refresh_dynamic_watchlist --dry-run
    python -m trading_agent.scripts.refresh_dynamic_watchlist --report

ENV:
    POSTGRES_DSN          — required (read from ~/.env.postgres on EC2)
    WATCHLIST_MAX         — override cap (default 40)
    WATCHLIST_MIN_MCAP    — override $2B floor (in USD)
    WATCHLIST_MIN_AVGVOL  — override 1M floor

Never raises during normal operation — failures of any single source are
logged and the rest proceed. The only way the script fails is DB unreachable.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger("refresh_dynamic_watchlist")

# Tunables (env-overridable)
WATCHLIST_MAX = int(os.environ.get("WATCHLIST_MAX", "100"))                      # hard ceiling (core can be large)
ROTATING_MAX = int(os.environ.get("WATCHLIST_ROTATING_MAX", "10"))              # today's yfinance hot picks
MIN_MARKET_CAP = float(os.environ.get("WATCHLIST_MIN_MCAP", "2000000000"))      # $2B
MIN_AVG_VOLUME = float(os.environ.get("WATCHLIST_MIN_AVGVOL", "1000000"))       # 1M shares
EVICT_AGE_DAYS = int(os.environ.get("WATCHLIST_EVICT_AGE_DAYS", "14"))
EVICT_PROPOSAL_DAYS = int(os.environ.get("WATCHLIST_EVICT_PROPOSAL_DAYS", "30"))
EVICT_SCOUT_THRESHOLD = float(os.environ.get("WATCHLIST_EVICT_SCOUT_THRESHOLD", "0.4"))

# Watchlist composition (redesigned 2026-06-02 per operator request):
#   CORE     = EVERY US stock/ETF in ANY of the operator's moomoo custom
#              watchlist groups (自选). Never evicted. This is the operator's
#              curated universe — they decide what's worth watching.
#   ROTATING = the top-N hottest names from yfinance screeners each day
#              (merged day_gainers + day_losers + most_actives, ranked by
#              liquidity-weighted move). Refreshed daily; capped at
#              ROTATING_MAX.
#
# HOT TODAY mark: a ticker that is in BOTH the operator's 自选 AND today's
# yfinance top-N is flagged metadata.hot_today=true — "your name that is
# moving today", the highest-priority signal for the scout. (Before this,
# the scout only saw prior-day absolute volume pre-open and always locked
# onto NVDA, the highest-volume mega-cap — the 6/2 NVDA-bias audit.)

# yfinance screener IDs merged into the daily top-N hot pool.
SCREENERS = [
    ("day_gainers",  "rotating", "screener:day_gainers"),
    ("day_losers",   "rotating", "screener:day_losers"),
    ("most_actives", "rotating", "screener:most_actives"),
]


@dataclass
class Candidate:
    """A ticker proposed for inclusion + the source + filter-relevant stats."""
    ticker: str
    source: str
    tier: str  # 'core' or 'rotating'
    market_cap: float = 0.0
    avg_volume_3m: float = 0.0
    change_pct: float = 0.0
    last_price: float = 0.0
    is_us_common: bool = True
    raw: dict = field(default_factory=dict)

    @property
    def dollar_volume(self) -> float:
        return self.last_price * self.avg_volume_3m

    @property
    def add_score(self) -> float:
        """Liquidity-weighted move score. Used to rank candidates when capping."""
        return abs(self.change_pct) * self.dollar_volume

    def to_metadata(self) -> dict:
        return {
            "market_cap": self.market_cap,
            "avg_volume_3m": self.avg_volume_3m,
            "change_pct": self.change_pct,
            "last_price": self.last_price,
            "add_score": self.add_score,
            # hot_today = in操作员's自选 AND in today's yfinance top-N movers.
            # Set in refresh() via raw["hot_today"]; the scout prioritizes these.
            "hot_today": bool(self.raw.get("hot_today", False)),
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Source fetchers — each returns list[Candidate]; never raises (logs + empty)
# ---------------------------------------------------------------------------

def _fetch_yfinance_screener(scr_id: str, source_label: str, tier: str,
                              count: int = 25) -> list[Candidate]:
    """Call yfinance.screen(scr_id) and produce Candidate rows.

    Returns [] on any failure (logged at WARNING).
    """
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed — skip screener %s", scr_id)
        return []
    try:
        res = yf.screen(scr_id, count=count)
    except Exception as e:
        log.warning("yfinance.screen(%s) failed: %s", scr_id, e)
        return []
    quotes = (res or {}).get("quotes") if isinstance(res, dict) else None
    if not quotes:
        return []
    out: list[Candidate] = []
    for q in quotes:
        sym = (q.get("symbol") or "").upper().strip()
        if not sym or "." in sym or "-" in sym:
            # Skip warrants (RAMP-WT), preferred shares (BAC.PRA), etc.
            continue
        # Skip OTC / pink sheet exchanges — Yahoo uses 'PNK' / 'OTC' / 'OBB'.
        exch = (q.get("fullExchangeName") or "").upper()
        if any(tag in exch for tag in ("PINK", "OTC", "OBB", "BULLETIN")):
            continue
        qt = (q.get("quoteType") or "").upper()
        if qt and qt not in ("EQUITY", "ETF"):
            # No mutual funds, options, futures.
            continue
        out.append(Candidate(
            ticker=sym,
            source=source_label,
            tier=tier,
            market_cap=float(q.get("marketCap") or 0),
            avg_volume_3m=float(q.get("averageDailyVolume3Month") or
                                 q.get("regularMarketVolume") or 0),
            change_pct=float(q.get("regularMarketChangePercent") or 0),
            last_price=float(q.get("regularMarketPrice") or 0),
            is_us_common=(qt == "EQUITY"),
            raw=q,
        ))
    log.info("yfinance %s: %d candidates after exchange/type filter", scr_id, len(out))
    return out


def _fetch_moomoo_group(group_name: str, tier: str) -> list[Candidate]:
    """Pull a moomoo CUSTOM watchlist group and convert each US.XXX entry."""
    try:
        from trading_agent.mcp_servers.moomoo.server import (
            _get_watchlist as moomoo_get_watchlist,  # type: ignore
        )
    except Exception:
        # Fallback: direct moomoo SDK import via the MCP module's public func.
        try:
            from trading_agent.mcp_servers.moomoo import server as moomoo_mod
            moomoo_get_watchlist = getattr(moomoo_mod, "get_watchlist", None)
        except Exception as e:
            log.warning("moomoo server module not importable: %s", e)
            return []
        if moomoo_get_watchlist is None:
            log.warning("moomoo get_watchlist not available")
            return []
    try:
        res = moomoo_get_watchlist(group_name=group_name)
    except Exception as e:
        log.warning("moomoo get_watchlist(%s) failed: %s", group_name, e)
        return []
    rows = (res or {}).get("rows") or []
    out: list[Candidate] = []
    for r in rows:
        code = (r.get("code") or "").strip()
        if not code.startswith("US."):
            continue
        bare = code.split(".", 1)[1]
        if not bare or "." in bare:  # skip ".VIX" / ".SPX" pseudo-tickers
            continue
        st = (r.get("stock_type") or "").upper()
        if st not in ("STOCK", "ETF"):  # no futures / indices / forex
            continue
        if r.get("delisting"):
            continue
        out.append(Candidate(
            ticker=bare,
            source=f"moomoo:{group_name}",
            tier=tier,
            raw=r,
        ))
    log.info("moomoo group '%s': %d candidates", group_name, len(out))
    return out


# Optional allowlist of moomoo custom group names to use as the watchlist
# core. Comma-separated, env-overridable. When set, ONLY these groups are
# pulled (e.g. "美股"). When empty, ALL custom groups are pulled. The
# operator created a dedicated 美股 group as the single source of truth —
# set WATCHLIST_MOOMOO_GROUPS=美股 on EC2 to use only it.
WATCHLIST_MOOMOO_GROUPS = [
    g.strip() for g in os.environ.get("WATCHLIST_MOOMOO_GROUPS", "").split(",")
    if g.strip()
]


def _fetch_all_moomoo_us_stocks() -> list[Candidate]:
    """Pull EVERY US stock/ETF from the operator's moomoo CUSTOM groups.

    Which groups: if WATCHLIST_MOOMOO_GROUPS is set, ONLY those named groups;
    otherwise ALL custom groups. Every name becomes tier='core' — never
    evicted, always presented to the scout.

    Source label is moomoo:<group>. Dedups across groups. Non-US,
    non-stock/ETF (futures, indices, options, HK/CN) are filtered out by
    the US. prefix + stock_type check.
    """
    out: list[Candidate] = []
    seen: set[str] = set()
    try:
        from trading_agent.mcp_servers.moomoo.server import (
            get_watchlist as moomoo_get_watchlist,
            get_watchlist_groups as moomoo_get_groups,
        )
    except Exception as e:
        log.warning("moomoo server not importable for自选 pull: %s", e)
        return []
    try:
        groups = moomoo_get_groups("CUSTOM")
    except Exception as e:
        log.warning("get_watchlist_groups(CUSTOM) failed: %s", e)
        return []
    allow = set(WATCHLIST_MOOMOO_GROUPS)
    if allow:
        log.info("watchlist restricted to moomoo groups: %s", sorted(allow))
    for g in (groups.get("rows") or []):
        gn = g.get("group_name")
        if not gn:
            continue
        if allow and gn not in allow:
            continue
        try:
            wl = moomoo_get_watchlist(group_name=gn)
        except Exception as e:
            log.warning("get_watchlist(%s) failed: %s", gn, e)
            continue
        n_us = 0
        for r in (wl.get("rows") or []):
            code = (r.get("code") or "").strip()
            if not code.startswith("US."):
                continue
            bare = code.split(".", 1)[1]
            if not bare or "." in bare:  # skip .VIX / .SPX pseudo-tickers
                continue
            st = (r.get("stock_type") or "").upper()
            if st not in ("STOCK", "ETF"):  # no futures / indices / forex
                continue
            if r.get("delisting"):
                continue
            if bare in seen:
                continue
            seen.add(bare)
            n_us += 1
            out.append(Candidate(
                ticker=bare,
                source=f"moomoo:{gn}",
                tier="core",  # ALL自选 names are core — never evicted
                raw=r,
            ))
        log.info("moomoo自选 group '%s': %d US stock/etf", gn, n_us)
    log.info("moomoo自选 total: %d unique US stock/etf → core", len(out))
    return out


def _select_yfinance_top_n(screener_cands: list[Candidate], n: int) -> list[Candidate]:
    """From the merged screener candidates, pick the top-N hottest by
    liquidity-weighted move (add_score = |chg_pct| × dollar_volume).

    These become the daily ROTATING tier — "today's hot names". Capped at
    n so the operator's curated 自选 (core) dominates the watchlist and the
    rotating slots reflect only genuinely-moving liquid names.
    """
    # Dedup by ticker, keeping the highest-add_score instance.
    by_ticker: dict[str, Candidate] = {}
    for c in screener_cands:
        ex = by_ticker.get(c.ticker)
        if ex is None or c.add_score > ex.add_score:
            by_ticker[c.ticker] = c
    ranked = sorted(by_ticker.values(), key=lambda c: c.add_score, reverse=True)
    top = ranked[:n]
    log.info("yfinance top-%d hot: %s", n, [c.ticker for c in top])
    return top


def _enrich_with_quote_stats(cands: list[Candidate]) -> list[Candidate]:
    """Fill in market_cap / avg_volume_3m / last_price for moomoo-sourced
    candidates (yfinance screen rows already have these).

    Uses yfinance.Tickers(...).tickers[t].info for batch enrichment.
    Tolerates partial failures — candidates without enriched stats are
    dropped if they're moomoo-sourced (we can't apply the liquidity floor
    without stats); screener-sourced rows already have stats and are kept.
    """
    needs_enrich = [c for c in cands if c.market_cap == 0 and c.last_price == 0]
    if not needs_enrich:
        return cands
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed — moomoo-sourced rows will lack stats and be dropped")
        return [c for c in cands if c not in needs_enrich]

    tickers_str = " ".join(c.ticker for c in needs_enrich)
    log.info("enriching %d moomoo-only candidates: %s", len(needs_enrich), tickers_str)
    try:
        tobj = yf.Tickers(tickers_str)
    except Exception as e:
        log.warning("yf.Tickers batch failed: %s — dropping unenriched moomoo rows", e)
        return [c for c in cands if c not in needs_enrich]

    enriched: list[Candidate] = []
    dropped = 0
    for c in needs_enrich:
        try:
            info = tobj.tickers[c.ticker].info or {}
            c.market_cap = float(info.get("marketCap") or 0)
            c.avg_volume_3m = float(info.get("averageDailyVolume3Month") or
                                     info.get("averageVolume10days") or 0)
            c.last_price = float(info.get("regularMarketPrice") or
                                  info.get("currentPrice") or 0)
            qt = (info.get("quoteType") or "").upper()
            c.is_us_common = qt == "EQUITY"
            if c.market_cap > 0 or c.avg_volume_3m > 0:
                enriched.append(c)
            else:
                dropped += 1
        except Exception as e:
            log.warning("enrich %s failed: %s", c.ticker, e)
            dropped += 1
    if dropped:
        log.info("dropped %d candidates with no enrichable stats", dropped)
    # already-enriched stay
    return [c for c in cands if c not in needs_enrich] + enriched


# ---------------------------------------------------------------------------
# Liquidity floor
# ---------------------------------------------------------------------------

def _passes_liquidity_floor(c: Candidate) -> tuple[bool, str | None]:
    """Return (ok, reject_reason)."""
    if c.market_cap and c.market_cap < MIN_MARKET_CAP:
        return False, f"mcap_{c.market_cap:.0f}_below_floor"
    if c.avg_volume_3m and c.avg_volume_3m < MIN_AVG_VOLUME:
        return False, f"vol_{c.avg_volume_3m:.0f}_below_floor"
    if not c.is_us_common and c.tier == "rotating" and c.market_cap == 0:
        # ETFs from screeners are OK (they have non-zero mcap); but unknown
        # quote types with zero mcap shouldn't auto-add.
        return False, "non_equity_zero_mcap"
    return True, None


# ---------------------------------------------------------------------------
# DB ops
# ---------------------------------------------------------------------------

def _held_tickers() -> set[str]:
    """Tickers we currently have exposure to — never evict these."""
    held: set[str] = set()
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                "SELECT DISTINCT symbol FROM journal_trades "
                "WHERE outcome = 'OPEN' AND qty > 0"
            )
            for (sym,) in cur.fetchall():
                if not sym:
                    continue
                # Strip option-code → bare ticker (US.NVDA260620C500 → NVDA)
                import re
                bare = sym.split(".", 1)[-1]
                m = re.match(r"^([A-Z\.]+?)\d{6,8}[CP]\d+$", bare)
                held.add((m.group(1) if m else bare).upper())
    except Exception as e:
        log.warning("held-position fetch failed: %s — treating as empty", e)
    return held


def _current_members() -> dict[str, dict]:
    """Map ticker → current row (as dict)."""
    out: dict[str, dict] = {}
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                """
                SELECT ticker, tier, source, added_at, last_seen_at,
                       last_scout_score, last_scout_ts, last_proposed_at,
                       eligible, metadata
                FROM watchlist_members
                """
            )
            for row in cur.fetchall():
                out[row[0]] = {
                    "ticker": row[0], "tier": row[1], "source": row[2],
                    "added_at": row[3], "last_seen_at": row[4],
                    "last_scout_score": row[5], "last_scout_ts": row[6],
                    "last_proposed_at": row[7], "eligible": row[8],
                    "metadata": row[9],
                }
    except Exception as e:
        log.error("current_members read failed: %s", e)
        raise
    return out


def _audit(cur, *, run_id: str, ticker: str, action: str,
           tier_before: str | None = None, tier_after: str | None = None,
           source: str | None = None, reason: str | None = None,
           metadata: dict | None = None) -> None:
    cur.execute(
        """
        INSERT INTO watchlist_audit
          (ticker, action, tier_before, tier_after, source, run_id, reason, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        """,
        (ticker, action, tier_before, tier_after, source, run_id, reason,
         json.dumps(metadata or {})),
    )


# ---------------------------------------------------------------------------
# Main refresh logic
# ---------------------------------------------------------------------------

def refresh(*, dry_run: bool = False) -> dict[str, Any]:
    """One refresh pass. Returns a summary dict."""
    run_id = f"watchlist_refresh_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    log.info("=== refresh start (dry_run=%s, run_id=%s) ===", dry_run, run_id)

    held = _held_tickers()
    log.info("currently held tickers (never evict): %s", sorted(held) or "(none)")

    current = _current_members()
    log.info("current membership: %d (core=%d, rotating=%d)",
             len(current),
             sum(1 for r in current.values() if r["tier"] == "core"),
             sum(1 for r in current.values() if r["tier"] == "rotating"))

    # ---- Source 1: ALL moomoo自选 US stocks → core ----
    core_cands = _fetch_all_moomoo_us_stocks()
    core_tickers = {c.ticker for c in core_cands}

    # ---- Source 2: yfinance top-N hot → rotating ----
    screener_cands: list[Candidate] = []
    for scr_id, tier, label in SCREENERS:
        screener_cands += _fetch_yfinance_screener(scr_id, label, tier)
    top_rotating = _select_yfinance_top_n(screener_cands, ROTATING_MAX)
    top_rotating_tickers = {c.ticker for c in top_rotating}

    # ---- HOT TODAY = 自选 ∩ today's top-N movers ----
    # The operator's curated names that are ALSO moving today. The single
    # highest-priority signal for the scout — "you like it AND it's active".
    hot_today = core_tickers & top_rotating_tickers
    log.info("HOT TODAY (自选 ∩ top-%d movers): %s", ROTATING_MAX, sorted(hot_today) or "(none)")

    candidates = core_cands + top_rotating

    # Enrich moomoo-only candidates with yfinance stats (market_cap, avg_vol)
    candidates = _enrich_with_quote_stats(candidates)

    # Stamp hot_today onto each candidate's raw so to_metadata can pick it up.
    for c in candidates:
        c.raw["hot_today"] = c.ticker in hot_today

    # Deduplicate by ticker: prefer the higher-tier / earlier source.
    # core > rotating; among same tier, first-encountered wins.
    tier_rank = {"core": 0, "rotating": 1}
    by_ticker: dict[str, Candidate] = {}
    for c in candidates:
        existing = by_ticker.get(c.ticker)
        if not existing:
            by_ticker[c.ticker] = c
            continue
        if tier_rank[c.tier] < tier_rank[existing.tier]:
            by_ticker[c.ticker] = c

    # Apply liquidity floor — only for tier='rotating'. Core never evicted on stats.
    accepted: dict[str, Candidate] = {}
    rejected: list[tuple[str, str]] = []
    for t, c in by_ticker.items():
        if c.tier == "core":
            accepted[t] = c
            continue
        ok, why = _passes_liquidity_floor(c)
        if ok:
            accepted[t] = c
        else:
            rejected.append((t, why or "unknown"))
    log.info("after liquidity floor: %d accepted, %d rejected", len(accepted), len(rejected))
    if rejected:
        log.info("rejected sample: %s", rejected[:10])

    # Plan: ADD / TOUCH / EVICT
    plan: list[dict] = []
    now = datetime.now(timezone.utc)
    cutoff_age = now.replace(microsecond=0)

    # 1. ADD / TOUCH new candidates
    for t, c in accepted.items():
        if t in current:
            row = current[t]
            # TOUCH: refresh last_seen_at + metadata + possibly tier upgrade
            new_tier = c.tier if tier_rank[c.tier] < tier_rank[row["tier"]] else row["tier"]
            plan.append({
                "action": "TOUCH",
                "ticker": t,
                "tier_before": row["tier"],
                "tier_after": new_tier,
                "source": c.source,
                "metadata": c.to_metadata(),
            })
        else:
            plan.append({
                "action": "ADD",
                "ticker": t,
                "tier_before": None,
                "tier_after": c.tier,
                "source": c.source,
                "metadata": c.to_metadata(),
            })

    # 2. EVICT stale rotating
    accepted_set = set(accepted.keys())
    for t, row in current.items():
        if row["tier"] != "rotating":
            continue
        if t in held:
            continue
        added_at = row["added_at"]
        last_proposed = row["last_proposed_at"]
        last_score = row["last_scout_score"]
        age_ok = added_at is not None and (now - added_at).days >= EVICT_AGE_DAYS
        score_ok = (last_score is None) or (last_score < EVICT_SCOUT_THRESHOLD)
        prop_ok = (last_proposed is None) or ((now - last_proposed).days >= EVICT_PROPOSAL_DAYS)
        not_in_sources = t not in accepted_set
        # Evict criteria: ALL of age_ok, score_ok, prop_ok, not_in_sources.
        # not_in_sources means today's screeners + moomoo groups didn't surface it.
        if age_ok and score_ok and prop_ok and not_in_sources:
            plan.append({
                "action": "EVICT",
                "ticker": t,
                "tier_before": row["tier"],
                "tier_after": None,
                "source": row.get("source"),
                "reason": f"stale_age={(now-added_at).days}d_score={last_score}_lastproposed={last_proposed}",
            })

    # 3. CAP enforcement: project post-plan size, prune lowest-score rotating.
    projected_members = {t: row.copy() for t, row in current.items()}
    for p in plan:
        if p["action"] == "ADD":
            projected_members[p["ticker"]] = {"tier": p["tier_after"], "ticker": p["ticker"]}
        elif p["action"] == "EVICT":
            projected_members.pop(p["ticker"], None)
        elif p["action"] == "TOUCH" and p["tier_after"] != p["tier_before"]:
            projected_members[p["ticker"]]["tier"] = p["tier_after"]

    # CAP enforcement: bound the ROTATING tier to ROTATING_MAX. Core (the
    # operator's 自选) is NEVER capped — every自选 name stays. Only the
    # daily yfinance hot picks compete for the rotating slots. This is the
    # 6/2 redesign: the old WATCHLIST_MAX total-cap silently pruned the
    # operator's curated semis/commodities; now their list is sacrosanct.
    rotating_count = sum(1 for r in projected_members.values() if r.get("tier") == "rotating")
    if rotating_count > ROTATING_MAX:
        # Compute add_score for everyone (existing rows use metadata.add_score if present, else 0)
        def _score(t: str) -> float:
            if t in accepted:
                return accepted[t].add_score
            md = (current.get(t) or {}).get("metadata") or {}
            return float(md.get("add_score") or 0)

        # Eligible for cap-eviction: tier='rotating', not held, not in held set
        rot_candidates = [
            t for t, r in projected_members.items()
            if r.get("tier") == "rotating" and t not in held
        ]
        rot_candidates.sort(key=_score)  # ascending — lowest scores evicted first
        over_by = rotating_count - ROTATING_MAX
        for t in rot_candidates[:over_by]:
            # If we were about to ADD this ticker, drop the ADD instead of inserting + evicting.
            existing_add = next((p for p in plan if p["action"] == "ADD" and p["ticker"] == t), None)
            if existing_add:
                plan.remove(existing_add)
            else:
                plan.append({
                    "action": "EVICT",
                    "ticker": t,
                    "tier_before": "rotating",
                    "tier_after": None,
                    "source": (current.get(t) or {}).get("source"),
                    "reason": f"cap_enforcement_low_score={_score(t):.2f}",
                })
            projected_members.pop(t, None)

    # ---- Execute ----
    summary = {
        "run_id": run_id,
        "ts": now.isoformat(),
        "dry_run": dry_run,
        "held": sorted(held),
        "current_members_n": len(current),
        "candidates_n": len(by_ticker),
        "accepted_n": len(accepted),
        "rejected_n": len(rejected),
        "rejected_sample": rejected[:20],
        "plan_counts": {
            "ADD":    sum(1 for p in plan if p["action"] == "ADD"),
            "TOUCH":  sum(1 for p in plan if p["action"] == "TOUCH"),
            "EVICT":  sum(1 for p in plan if p["action"] == "EVICT"),
        },
        "plan": plan,
        "projected_n": len(projected_members),
    }

    if dry_run:
        log.info("[dry-run] plan: ADD=%d TOUCH=%d EVICT=%d (projected_n=%d)",
                 summary["plan_counts"]["ADD"],
                 summary["plan_counts"]["TOUCH"],
                 summary["plan_counts"]["EVICT"],
                 summary["projected_n"])
        return summary

    # Apply within a single transaction
    from trading_agent.store.postgres import conn
    with conn() as c:
        with c.cursor() as cur:
            for p in plan:
                t = p["ticker"]
                if p["action"] == "ADD":
                    cur.execute(
                        """
                        INSERT INTO watchlist_members
                          (ticker, tier, source, added_by_run_id, last_seen_at, metadata)
                        VALUES (%s, %s, %s, %s, NOW(), %s::jsonb)
                        ON CONFLICT (ticker) DO UPDATE SET
                            last_seen_at = NOW(),
                            metadata = EXCLUDED.metadata
                        """,
                        (t, p["tier_after"], p["source"], run_id,
                         json.dumps(p["metadata"])),
                    )
                    _audit(cur, run_id=run_id, ticker=t, action="ADD",
                           tier_after=p["tier_after"], source=p["source"],
                           reason="new_member_from_source", metadata=p["metadata"])
                elif p["action"] == "TOUCH":
                    cur.execute(
                        """
                        UPDATE watchlist_members
                           SET last_seen_at = NOW(),
                               metadata     = %s::jsonb,
                               tier         = %s
                         WHERE ticker = %s
                        """,
                        (json.dumps(p["metadata"]), p["tier_after"], t),
                    )
                    if p["tier_before"] != p["tier_after"]:
                        _audit(cur, run_id=run_id, ticker=t, action="TIER_CHANGE",
                               tier_before=p["tier_before"], tier_after=p["tier_after"],
                               source=p["source"], reason="tier_upgraded_from_source")
                    else:
                        _audit(cur, run_id=run_id, ticker=t, action="TOUCH",
                               tier_before=p["tier_before"], tier_after=p["tier_after"],
                               source=p["source"], reason="re-seen_from_source",
                               metadata=p["metadata"])
                elif p["action"] == "EVICT":
                    cur.execute("DELETE FROM watchlist_members WHERE ticker = %s", (t,))
                    _audit(cur, run_id=run_id, ticker=t, action="EVICT",
                           tier_before=p["tier_before"], source=p.get("source"),
                           reason=p.get("reason", "stale_or_capped"))
        c.commit()

    log.info("APPLIED: ADD=%d TOUCH=%d EVICT=%d (final_n=%d)",
             summary["plan_counts"]["ADD"],
             summary["plan_counts"]["TOUCH"],
             summary["plan_counts"]["EVICT"],
             summary["projected_n"])
    return summary


# ---------------------------------------------------------------------------
# Helper for the rest of the codebase
# ---------------------------------------------------------------------------

def get_active_watchlist() -> list[str]:
    """Return the current watchlist tickers (eligible only), tier-sorted.

    Used by jobs/premarket_watchlist.py instead of the old DEFAULT_WATCHLIST.
    Falls back to a hardcoded safety list if DB is unreachable, so the
    premarket scan never silently runs on an empty watchlist.
    """
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                """
                SELECT ticker
                  FROM watchlist_members
                 WHERE eligible = TRUE
                 ORDER BY (tier = 'core') DESC, last_seen_at DESC
                """
            )
            rows = [r[0] for r in cur.fetchall()]
            if rows:
                return rows
    except Exception as e:
        log.warning("get_active_watchlist DB read failed: %s — using fallback", e)
    # Fallback — same as the old DEFAULT_WATCHLIST core
    return ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AMZN"]


def get_active_watchlist_detailed() -> list[dict]:
    """Like get_active_watchlist but returns tier + hot_today per ticker.

    The scout prompt uses this to structure a HOT TODAY section (自选 names
    moving today) ahead of the rest — the 6/2 NVDA-bias fix. Each item:
    {ticker, tier, hot_today}. Empty list on DB error (caller falls back to
    the flat list).
    """
    out: list[dict] = []
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                """
                SELECT ticker, tier,
                       COALESCE((metadata->>'hot_today')::boolean, FALSE) AS hot_today,
                       (metadata->>'avg_volume_3m')::float AS avg_vol_3m
                  FROM watchlist_members
                 WHERE eligible = TRUE
                 ORDER BY
                   COALESCE((metadata->>'hot_today')::boolean, FALSE) DESC,
                   (tier = 'rotating') DESC,
                   last_seen_at DESC
                """
            )
            for ticker, tier, hot, avg_vol in cur.fetchall():
                out.append({
                    "ticker": ticker, "tier": tier, "hot_today": bool(hot),
                    "avg_volume_3m": float(avg_vol) if avg_vol is not None else None,
                })
    except Exception as e:
        log.warning("get_active_watchlist_detailed DB read failed: %s", e)
    return out


def get_universe_snapshot() -> dict:
    """Snapshot of current watchlist for embedding in shadow_proposals.

    Caller (insert_shadow_proposal) writes this JSONB into
    shadow_proposals.universe_snapshot so later analysis can normalize
    LLM proposal quality by the universe it picked from.
    """
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                """
                SELECT ticker, tier, source
                  FROM watchlist_members
                 WHERE eligible = TRUE
                 ORDER BY ticker
                """
            )
            tickers = [
                {"t": r[0], "tier": r[1], "source": r[2]}
                for r in cur.fetchall()
            ]
        return {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "universe_size": len(tickers),
            "tickers": tickers,
        }
    except Exception as e:
        log.warning("get_universe_snapshot failed: %s", e)
        return {"captured_at": datetime.now(timezone.utc).isoformat(),
                "universe_size": 0, "tickers": []}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _report() -> dict:
    """Print current state without changing anything."""
    members = _current_members()
    held = _held_tickers()
    core = [t for t, r in members.items() if r["tier"] == "core"]
    rotating = sorted(
        [(t, r) for t, r in members.items() if r["tier"] == "rotating"],
        key=lambda kv: (kv[1].get("last_proposed_at") or
                        kv[1].get("last_seen_at") or
                        datetime(1970, 1, 1, tzinfo=timezone.utc)),
        reverse=True,
    )
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "total": len(members),
        "core_n": len(core),
        "rotating_n": len(rotating),
        "core": sorted(core),
        "rotating": [t for t, _ in rotating],
        "held": sorted(held),
        "cap": WATCHLIST_MAX,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh dynamic watchlist")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute plan but don't write to DB")
    ap.add_argument("--report", action="store_true",
                    help="Print current state and exit")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.report:
            print(json.dumps(_report(), default=str, indent=2))
            return 0
        summary = refresh(dry_run=args.dry_run)
        print(json.dumps(summary, default=str, indent=2))
        return 0
    except Exception as e:
        log.error("refresh failed: %s\n%s", e, traceback.format_exc())
        return 1
    finally:
        # moomoo's OpenQuoteContext / OpenSecTradeContext spawn non-daemon
        # background threads (callback_executor + network_manager socket
        # reader). Without explicit shutdown they keep the interpreter
        # alive after main() returns. 5/22 incident: watchlist-refresh
        # finished its work in 12s but hung for 4+ minutes until systemd
        # TERM (TimeoutStartSec=300), then healthcheck.failed_units_check
        # fired hourly sev-2 ntfy alerts for "failed" state until the user
        # complained. shutdown() is idempotent and swallows errors.
        try:
            from trading_agent.mcp_servers.moomoo.server import (
                shutdown as moomoo_shutdown,
            )
            moomoo_shutdown()
        except Exception as e:
            log.warning("moomoo shutdown failed (non-fatal): %s", e)


if __name__ == "__main__":
    sys.exit(main())
