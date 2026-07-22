#!/usr/bin/env python3
"""M1-0.4 gate-feasibility replay + managed-payoff expectancy for the
credit_vertical_index_30_45 sleeve.

OFFLINE analysis over pre-fetched daily option aggregates. stdlib + numpy
only — no pandas, no network, no project imports.

Inputs (all under --data-dir):
  underlying_{SYM}.json            daily aggregates for the underlying
                                   ({"results":[{t,o,h,l,c,...}]} or a bare
                                   list of bars; t = ms epoch or ISO date)
  listing_{SYM}_{EXPIRY}.json      (optional) contract reference rows used to
                                   resolve strike / contract_type when the OCC
                                   ticker cannot be parsed
  contracts_{SYM}_{EXPIRY}.jsonl   one line per contract:
                                   {ticker, strike, bars:[{t,o,h,l,c,vw,v,n}]}
  (EXPIRY in filenames may be ISO "2026-06-19" or compact "20260619".)

--batch-plan batch_plan.json declares what to replay. Accepted shapes:
  {"batches": [{"symbol": "SPY", "expiry": "2026-06-19",
                "entry_dates": ["2026-05-08", ...]}, ...]}
  or a bare list of such batch objects, or a list of flat rows
  {"entry_date": ..., "symbol": ..., "expiry": ...}. Key aliases:
  expiry|expiration|exp, entry_dates|entry_date|dates.

Per (entry_date, symbol) the engine replays the sleeve gate stack
(short put |delta| in [0.20, 0.35], width 5 preferred / 10 fallback,
credit >= width/4) and, when a vertical qualifies, walks the managed
payoff (50% profit-take, forced exit at expiry-21d) on daily closes.

MODELING APPROXIMATIONS (also recorded in summary.json):
  * IV is inverted from a EUROPEAN Black-Scholes put on daily closes.
    SPY/QQQ options are American, but for OTM short-dated index puts the
    early-exercise premium is small; the delta band is tolerant to it.
  * Daily close is used as the mark (mid proxy); friction is charged on
    top of it from the measured spread widths.
  * The long wing only needs a bar on entry day (no n>=30 gate) — the
    liquidity gate is applied to the short leg, where the fill risk is.
  * On a profit-take day the exit debit is exactly 0.5*credit (resting
    order assumption); friction legs are priced off that day's closes.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants

RISK_FREE_RATE = 0.043
DIV_YIELD = {"SPY": 0.012, "QQQ": 0.006}
DEFAULT_SPREAD_PCT = {"SPY": 0.0049, "QQQ": 0.0094}  # M1-0.3 measured, frac of mid
FEE_PER_LEG_PER_DIRECTION = 1.0                       # $1 x 4 per round trip

DELTA_LO, DELTA_HI = 0.20, 0.35
TARGET_DELTA = 0.5 * (DELTA_LO + DELTA_HI)            # deterministic tie-break
WIDTHS = (5.0, 10.0)                                  # 5 preferred, 10 fallback
CREDIT_FLOOR_FRAC = 0.25                              # credit >= width/4
MIN_SHORT_TRADES = 30                                 # bar n >= 30 on entry day
DTE_RANGE = (30, 45)
PT_FRAC = 0.5                                         # 50% profit take
FORCED_EXIT_DTE = 21                                  # exit at expiry - 21d

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260722
HIST_EDGES = [round(x, 2) for x in np.arange(-1.5, 0.5001, 0.1)]

APPROXIMATION_NOTES = [
    "IV inverted from EUROPEAN Black-Scholes put (bisection) although "
    "SPY/QQQ options are American; acceptable for OTM short-dated index "
    "puts, biases IV/delta slightly.",
    "Daily close used as the mark (mid proxy); measured half-spread x 2 "
    "legs x 2 directions + $1 x 4 fees charged on top.",
    "Long wing requires only a bar on entry day (no n>=30 trade-count "
    "gate); the n>=30 gate applies to the short leg.",
    "Profit-take exits fill at exactly 0.5 x credit; friction for that "
    "exit is priced off the trigger day's leg closes.",
    "r = 0.043 fixed; continuous dividend yield q: SPY 0.012, QQQ 0.006.",
]

OCC_RE = re.compile(r"(\d{6})([CP])(\d{8})$")

# ------------------------------------------------------------- BS math


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bs_put_price(spot: float, strike: float, t_years: float, r: float,
                 q: float, sigma: float) -> float:
    if t_years <= 0 or sigma <= 0:
        return max(strike - spot, 0.0)
    sq = sigma * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t_years) / sq
    d2 = d1 - sq
    return (strike * math.exp(-r * t_years) * norm_cdf(-d2)
            - spot * math.exp(-q * t_years) * norm_cdf(-d1))


def bs_put_delta(spot: float, strike: float, t_years: float, r: float,
                 q: float, sigma: float) -> float:
    if t_years <= 0 or sigma <= 0:
        return -1.0 if spot < strike else 0.0
    sq = sigma * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t_years) / sq
    return -math.exp(-q * t_years) * norm_cdf(-d1)


def implied_vol_put(price: float, spot: float, strike: float, t_years: float,
                    r: float, q: float, lo: float = 1e-4, hi: float = 5.0,
                    tol: float = 1e-8, max_iter: int = 200) -> float | None:
    """Bisection inversion of the European BS put. None when the price is
    outside the attainable [price(lo), price(hi)] bracket (stale/bad mark)."""
    if price <= 0 or t_years <= 0:
        return None
    p_lo = bs_put_price(spot, strike, t_years, r, q, lo)
    p_hi = bs_put_price(spot, strike, t_years, r, q, hi)
    if price < p_lo or price > p_hi:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = bs_put_price(spot, strike, t_years, r, q, mid)
        if abs(p_mid - price) < tol or (hi - lo) < tol:
            return mid
        if p_mid < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------- data loading


def parse_date(v) -> date:
    """ISO date/datetime string, date, or epoch in s/ms."""
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, (int, float)):
        ts = float(v)
        if ts > 1e11:  # ms epoch
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc).date()
    return date.fromisoformat(str(v)[:10])


def _bars_of(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    for key in ("results", "bars"):
        if isinstance(payload, dict) and isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError("unrecognized bars payload (expected list or "
                     "{'results': [...]}/{'bars': [...]})")


def _expiry_names(expiry: date) -> tuple[str, str]:
    return expiry.isoformat(), expiry.strftime("%Y%m%d")


def _find_file(data_dir: Path, stem_candidates: list[str]) -> Path | None:
    for stem in stem_candidates:
        p = data_dir / stem
        if p.exists():
            return p
    return None


def load_underlying_closes(data_dir: Path, symbol: str) -> dict[date, float]:
    path = _find_file(data_dir, [f"underlying_{symbol}.json"])
    if path is None:
        hits = sorted(data_dir.glob(f"underlying_{symbol}_*.json"))
        if not hits:
            raise FileNotFoundError(f"underlying_{symbol}.json in {data_dir}")
        path = hits[0]
    closes: dict[date, float] = {}
    for bar in _bars_of(json.loads(path.read_text())):
        closes[parse_date(bar["t"])] = float(bar["c"])
    return closes


def load_listing(data_dir: Path, symbol: str, expiry: date) -> dict[str, dict]:
    """Optional ticker -> {strike, contract_type} map from listing files."""
    iso, compact = _expiry_names(expiry)
    path = _find_file(data_dir, [f"listing_{symbol}_{iso}.json",
                                 f"listing_{symbol}_{compact}.json",
                                 f"listing_{symbol}.json"])
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    rows = payload.get("results", payload) if isinstance(payload, dict) else payload
    out: dict[str, dict] = {}
    for row in rows or []:
        tkr = row.get("ticker")
        if not tkr:
            continue
        out[tkr] = {
            "strike": row.get("strike_price", row.get("strike")),
            "contract_type": (row.get("contract_type") or row.get("type") or "").lower(),
        }
    return out


def _classify_contract(ticker: str, listing: dict[str, dict]) -> tuple[str | None, float | None]:
    """(contract_type 'put'/'call'/None, strike) from OCC ticker, listing fallback."""
    m = OCC_RE.search(ticker or "")
    if m:
        return ("put" if m.group(2) == "P" else "call"), int(m.group(3)) / 1000.0
    meta = listing.get(ticker)
    if meta:
        ct = meta.get("contract_type") or None
        strike = meta.get("strike")
        return ct, (float(strike) if strike is not None else None)
    return None, None


def load_put_contracts(data_dir: Path, symbol: str, expiry: date) -> dict[float, dict]:
    """strike -> {ticker, bars: {date: bar}} for the puts of one batch."""
    iso, compact = _expiry_names(expiry)
    path = _find_file(data_dir, [f"contracts_{symbol}_{iso}.jsonl",
                                 f"contracts_{symbol}_{compact}.jsonl"])
    if path is None:
        raise FileNotFoundError(
            f"contracts_{symbol}_{iso}.jsonl (or _{compact}) in {data_dir}")
    listing = load_listing(data_dir, symbol, expiry)
    puts: dict[float, dict] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ticker = row.get("ticker", "")
            ctype, occ_strike = _classify_contract(ticker, listing)
            if ctype == "call":
                continue  # sleeve replays short PUT verticals only
            strike = row.get("strike", occ_strike)
            if strike is None:
                continue
            strike = float(strike)
            bars = {parse_date(b["t"]): b for b in row.get("bars", [])}
            puts[strike] = {"ticker": ticker, "bars": bars}
    return puts


def load_batch_plan(path: Path) -> list[dict]:
    """Normalize the batch plan to [{entry_date, symbol, expiry}, ...]."""
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        rows = payload.get("batches", payload.get("entries"))
        if rows is None:
            raise ValueError("batch plan dict needs 'batches' or 'entries'")
    else:
        rows = payload
    out: list[dict] = []
    for row in rows:
        symbol = row["symbol"].upper()
        expiry = parse_date(row.get("expiry") or row.get("expiration") or row["exp"])
        dates = row.get("entry_dates") or row.get("dates")
        if dates is None:
            dates = [row["entry_date"]]
        for d in dates:
            out.append({"entry_date": parse_date(d), "symbol": symbol,
                        "expiry": expiry})
    out.sort(key=lambda e: (e["entry_date"], e["symbol"], e["expiry"]))
    return out


# ------------------------------------------------- vertical selection


def _entry_mark(contract: dict | None, entry_date: date,
                min_trades: int | None = None) -> float | None:
    if contract is None:
        return None
    bar = contract["bars"].get(entry_date)
    if bar is None:
        return None
    if min_trades is not None and float(bar.get("n", 0) or 0) < min_trades:
        return None
    close = float(bar["c"])
    return close if close > 0 else None


def select_vertical(entry_date: date, expiry: date, spot: float,
                    puts: dict[float, dict], q: float,
                    r: float = RISK_FREE_RATE) -> dict:
    """Replay the gate stack on one (entry_date, symbol, expiry).

    Returns {qualified: bool, reason: str|None, ...selection fields}.
    Failure reasons (most-progressed wins):
      no-strike-in-band -> wing-missing -> credit-below-floor
    """
    t_years = (expiry - entry_date).days / 365.0
    in_band: list[dict] = []
    for strike in sorted(puts):
        mark = _entry_mark(puts[strike], entry_date, MIN_SHORT_TRADES)
        if mark is None:
            continue
        iv = implied_vol_put(mark, spot, strike, t_years, r, q)
        if iv is None:
            continue
        delta = bs_put_delta(spot, strike, t_years, r, q, iv)
        if DELTA_LO <= abs(delta) <= DELTA_HI:
            in_band.append({"strike": strike, "mark": mark, "iv": iv,
                            "delta": delta})
    if not in_band:
        return {"qualified": False, "reason": "no-strike-in-band"}

    # Deterministic preference: |delta| closest to the band midpoint,
    # higher strike breaking ties.
    in_band.sort(key=lambda c: (abs(abs(c["delta"]) - TARGET_DELTA), -c["strike"]))
    any_wing = False
    for cand in in_band:
        short_strike = cand["strike"]
        for width in WIDTHS:  # 5 preferred, 10 only if the 5-wing has no bar
            long_strike = short_strike - width
            long_mark = _entry_mark(puts.get(long_strike), entry_date)
            if long_mark is None:
                continue
            any_wing = True
            credit = cand["mark"] - long_mark
            if credit >= width * CREDIT_FLOOR_FRAC and credit < width:
                return {
                    "qualified": True, "reason": None,
                    "short_strike": short_strike, "long_strike": long_strike,
                    "width": width, "credit": credit,
                    "short_mark": cand["mark"], "long_mark": long_mark,
                    "short_iv": cand["iv"], "short_delta": cand["delta"],
                    "short_ticker": puts[short_strike]["ticker"],
                    "long_ticker": puts[long_strike]["ticker"],
                }
            break  # a wing with a bar existed; do not widen for more credit
    return {"qualified": False,
            "reason": "credit-below-floor" if any_wing else "wing-missing"}


# ---------------------------------------------------- managed payoff walk


def managed_walk(entry_date: date, expiry: date, credit: float,
                 short_bars: dict[date, dict],
                 long_bars: dict[date, dict]) -> dict:
    """Walk daily spread marks after entry until the first exit trigger.

    Exit priority on a given day: profit-take (mark <= 0.5*credit) first,
    then the forced 21-DTE exit; running out of bars exits at the last
    mark with data_end=True.
    """
    forced_date = expiry - timedelta(days=FORCED_EXIT_DTE)
    days = sorted(d for d in short_bars
                  if d in long_bars and entry_date < d <= expiry)
    last = None
    for d in days:
        s_close = float(short_bars[d]["c"])
        l_close = float(long_bars[d]["c"])
        mark = s_close - l_close
        last = (d, mark, s_close, l_close)
        if mark <= PT_FRAC * credit:
            return {"exit_date": d, "exit_debit": PT_FRAC * credit,
                    "exit_reason": "profit_take", "data_end": False,
                    "short_exit_mark": s_close, "long_exit_mark": l_close}
        if d >= forced_date:
            return {"exit_date": d, "exit_debit": mark,
                    "exit_reason": "dte_21", "data_end": False,
                    "short_exit_mark": s_close, "long_exit_mark": l_close}
    if last is None:  # no post-entry bars at all: flat exit at entry mark
        return {"exit_date": entry_date, "exit_debit": credit,
                "exit_reason": "data_end", "data_end": True,
                "short_exit_mark": None, "long_exit_mark": None}
    d, mark, s_close, l_close = last
    return {"exit_date": d, "exit_debit": mark, "exit_reason": "data_end",
            "data_end": True, "short_exit_mark": s_close,
            "long_exit_mark": l_close}


def friction_dollars(spread_pct: float, short_entry: float, long_entry: float,
                     short_exit: float, long_exit: float) -> float:
    """Half-spread x 2 legs x 2 directions on the respective marks
    + $1 x 4 fees, per contract-set (x100 multiplier)."""
    half = spread_pct / 2.0
    crossed = half * (short_entry + long_entry + short_exit + long_exit) * 100.0
    return crossed + 4.0 * FEE_PER_LEG_PER_DIRECTION


# --------------------------------------------------------- per-entry replay

_REASON_RANK = {  # most-progressed failure wins across candidate expiries
    "no-data": 0, "no-underlying-bar": 1, "dte-out-of-range": 2,
    "no-strike-in-band": 3, "wing-missing": 4, "credit-below-floor": 5,
}


def replay_entry(entry_date: date, symbol: str, expiries: list[date],
                 closes: dict[date, float], puts_by_expiry: dict[date, dict],
                 spread_pct: float, q: float) -> dict:
    """One record per (entry_date, symbol): try candidate expiries in
    ascending DTE order, keep the first qualifying vertical; otherwise
    report the most-progressed failure reason."""
    rec = {"entry_date": entry_date.isoformat(), "symbol": symbol,
           "qualified": False, "reason": "no-data"}
    spot = closes.get(entry_date)
    if spot is None:
        rec["reason"] = "no-underlying-bar"
        return rec
    rec["spot"] = spot
    best_reason = "no-data"
    for expiry in sorted(expiries):
        dte = (expiry - entry_date).days
        if not (DTE_RANGE[0] <= dte <= DTE_RANGE[1]):
            reason = "dte-out-of-range"
        else:
            sel = select_vertical(entry_date, expiry, spot,
                                  puts_by_expiry[expiry], q)
            if sel["qualified"]:
                puts = puts_by_expiry[expiry]
                walk = managed_walk(entry_date, expiry, sel["credit"],
                                    puts[sel["short_strike"]]["bars"],
                                    puts[sel["long_strike"]]["bars"])
                credit, width = sel["credit"], sel["width"]
                pnl_gross = (credit - walk["exit_debit"]) * 100.0
                friction = friction_dollars(
                    spread_pct, sel["short_mark"], sel["long_mark"],
                    walk["short_exit_mark"] if walk["short_exit_mark"] is not None
                    else sel["short_mark"],
                    walk["long_exit_mark"] if walk["long_exit_mark"] is not None
                    else sel["long_mark"])
                risk = (width - credit) * 100.0
                pnl_net = pnl_gross - friction
                rec.update(sel)
                rec.update({
                    "qualified": True, "reason": None,
                    "expiry": expiry.isoformat(), "dte": dte,
                    "exit_date": walk["exit_date"].isoformat(),
                    "exit_reason": walk["exit_reason"],
                    "exit_debit": walk["exit_debit"],
                    "data_end": walk["data_end"],
                    "pnl_gross": pnl_gross, "friction": friction,
                    "pnl_net": pnl_net, "risk_dollars": risk,
                    "result_r": pnl_net / risk,
                })
                return rec
            reason = sel["reason"]
        if _REASON_RANK[reason] > _REASON_RANK[best_reason]:
            best_reason = reason
    rec["reason"] = best_reason
    return rec


# ------------------------------------------------------------- statistics


def block_bootstrap_lb95_mean(entries: list[dict],
                              resamples: int = BOOTSTRAP_RESAMPLES,
                              seed: int = BOOTSTRAP_SEED) -> float | None:
    """LB95 of mean(result_r) with entry-ISO-weeks as bootstrap blocks.

    Resampled mean = sum(week sums)/sum(week counts) over weeks drawn with
    replacement; the 5th percentile of the resampled means is the one-sided
    95% lower bound. Deterministic for a fixed seed.
    """
    by_week: dict[str, list[float]] = defaultdict(list)
    for e in entries:
        iso = date.fromisoformat(e["entry_date"]).isocalendar()
        by_week[f"{iso[0]}-W{iso[1]:02d}"].append(e["result_r"])
    weeks = sorted(by_week)
    if not weeks:
        return None
    sums = np.array([sum(by_week[w]) for w in weeks])
    counts = np.array([len(by_week[w]) for w in weeks], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(weeks), size=(resamples, len(weeks)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.percentile(means, 5.0))


def max_drawdown_r(entries: list[dict]) -> float:
    """Max peak-to-trough drop of the cumulative result_r path, entries in
    (entry_date, symbol) order. Positive number in R units; 0.0 if empty."""
    ordered = sorted(entries, key=lambda e: (e["entry_date"], e["symbol"]))
    cum = peak = 0.0
    dd = 0.0
    for e in ordered:
        cum += e["result_r"]
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return dd


def _availability(records: list[dict]) -> dict:
    n = len(records)
    q = sum(1 for r in records if r["qualified"])
    return {"n_entries": n, "n_qualified": q,
            "rate": (q / n) if n else None}


def summarize(records: list[dict], resamples: int, seed: int,
              spread_pct: dict[str, float]) -> dict:
    qualified = [r for r in records if r["qualified"]]
    by_symbol = defaultdict(list)
    by_month = defaultdict(list)
    for r in records:
        by_symbol[r["symbol"]].append(r)
        by_month[r["entry_date"][:7]].append(r)

    rs = np.array([r["result_r"] for r in qualified])
    hist_counts, _ = (np.histogram(rs, bins=HIST_EDGES)
                      if rs.size else (np.zeros(len(HIST_EDGES) - 1, dtype=int), None))
    exit_counts = Counter(r["exit_reason"] for r in qualified)
    reason_counts = Counter(r["reason"] for r in records if not r["qualified"])

    n_q = len(qualified)
    summary = {
        "config": {
            "risk_free_rate": RISK_FREE_RATE, "div_yield": DIV_YIELD,
            "spread_pct": spread_pct, "delta_band": [DELTA_LO, DELTA_HI],
            "widths": list(WIDTHS), "credit_floor_frac": CREDIT_FLOOR_FRAC,
            "min_short_trades": MIN_SHORT_TRADES, "dte_range": list(DTE_RANGE),
            "pt_frac": PT_FRAC, "forced_exit_dte": FORCED_EXIT_DTE,
            "bootstrap_resamples": resamples, "bootstrap_seed": seed,
            "fees_per_round_trip": 4.0 * FEE_PER_LEG_PER_DIRECTION,
        },
        "approximation_notes": APPROXIMATION_NOTES,
        "availability": {
            "acceptance_criterion": ">=60% of entry days must have a "
                                    "qualifying vertical (revival plan M1-0.4)",
            "overall": _availability(records),
            "per_symbol": {s: _availability(v) for s, v in sorted(by_symbol.items())},
            "per_month": {m: _availability(v) for m, v in sorted(by_month.items())},
            "unqualified_reasons": dict(sorted(reason_counts.items())),
        },
        "managed_payoff": {
            "n_trades": n_q,
            "win_rate": float(np.mean(rs > 0)) if n_q else None,
            "mean_r": float(rs.mean()) if n_q else None,
            "median_r": float(np.median(rs)) if n_q else None,
            "lb95_mean_r": block_bootstrap_lb95_mean(qualified, resamples, seed),
            "max_drawdown_r": max_drawdown_r(qualified),
            "histogram_r": {"edges": HIST_EDGES,
                            "counts": hist_counts.tolist(),
                            "underflow": int(np.sum(rs < HIST_EDGES[0])) if n_q else 0,
                            "overflow": int(np.sum(rs >= HIST_EDGES[-1])) if n_q else 0},
        },
        "exit_degradation": {
            "counts": {k: exit_counts.get(k, 0)
                       for k in ("profit_take", "dte_21", "data_end")},
            "fractions": {k: (exit_counts.get(k, 0) / n_q if n_q else None)
                          for k in ("profit_take", "dte_21", "data_end")},
            "n_data_end_flagged": sum(1 for r in qualified if r.get("data_end")),
        },
    }
    return summary


# ------------------------------------------------------------------ driver


def parse_spread_pct(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        sym, _, val = chunk.partition("=")
        if not val:
            raise ValueError(f"bad --spread-pct chunk {chunk!r} (want SYM=0.0049)")
        out[sym.strip().upper()] = float(val)
    return out


def run(data_dir: Path, batch_plan: Path, out_dir: Path,
        spread_pct: dict[str, float], resamples: int = BOOTSTRAP_RESAMPLES,
        seed: int = BOOTSTRAP_SEED) -> dict:
    plan = load_batch_plan(batch_plan)

    # group candidate expiries per (entry_date, symbol)
    grouped: dict[tuple[date, str], list[date]] = defaultdict(list)
    for row in plan:
        key = (row["entry_date"], row["symbol"])
        if row["expiry"] not in grouped[key]:
            grouped[key].append(row["expiry"])

    closes_cache: dict[str, dict[date, float]] = {}
    puts_cache: dict[tuple[str, date], dict] = {}
    records: list[dict] = []
    for (entry_date, symbol), expiries in sorted(grouped.items()):
        if symbol not in closes_cache:
            closes_cache[symbol] = load_underlying_closes(data_dir, symbol)
        for expiry in expiries:
            if (symbol, expiry) not in puts_cache:
                puts_cache[(symbol, expiry)] = load_put_contracts(
                    data_dir, symbol, expiry)
        if symbol not in spread_pct:
            raise ValueError(f"no --spread-pct entry for {symbol}")
        records.append(replay_entry(
            entry_date, symbol, expiries, closes_cache[symbol],
            {e: puts_cache[(symbol, e)] for e in expiries},
            spread_pct[symbol], DIV_YIELD.get(symbol, 0.0)))

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "entries.jsonl").open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    summary = summarize(records, resamples, seed, spread_pct)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--batch-plan", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--spread-pct", default="SPY=0.0049,QQQ=0.0094",
                    help="comma list SYM=frac_of_mid (measured full spread)")
    ap.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args(argv)

    summary = run(args.data_dir, args.batch_plan, args.out_dir,
                  parse_spread_pct(args.spread_pct), args.resamples, args.seed)
    avail = summary["availability"]["overall"]
    mp = summary["managed_payoff"]
    print(f"entries={avail['n_entries']} qualified={avail['n_qualified']} "
          f"availability={avail['rate'] if avail['rate'] is None else round(avail['rate'], 3)}")
    print(f"trades={mp['n_trades']} WR={mp['win_rate']} mean_r={mp['mean_r']} "
          f"median_r={mp['median_r']} lb95_mean_r={mp['lb95_mean_r']} "
          f"maxDD_r={mp['max_drawdown_r']}")
    print(f"wrote {args.out_dir / 'entries.jsonl'} and {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
