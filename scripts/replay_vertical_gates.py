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

MISSING DATA IS NEVER FABRICATED. Every bar that is absent, unparsable or
non-positive is skipped and counted; no interpolation, no carry-forward, no
synthetic marks. Entries are triaged along a funnel so that a *data* gap is
never reported as a *gate* rejection:

  L0  no usable data at all       (plan/underlying/contract file/bars missing)
  L1  entry-day option marks exist                       -> "marks_present"
  L2  at least one complete vertical is CONSTRUCTIBLE from the fetched
      strikes (a short strike and its 5- or 10-wide wing both have an
      entry-day bar)                                     -> "data_adequate"
  L3+ the gate stack actually ran and either qualified or rejected

  data_coverage_rate = |L2| / |all planned entries|
  availability_rate  = |qualified| / |L2|      <- the >=60% acceptance test

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
CREDIT_FLOOR_GRID = (0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25)
MIN_SHORT_TRADES = 30                                 # bar n >= 30 on entry day
DTE_RANGE = (30, 45)
PT_FRAC = 0.5                                         # 50% profit take
FORCED_EXIT_DTE = 21                                  # exit at expiry - 21d

AVAILABILITY_FLOOR = 0.60                             # revival plan M1-0.4

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
    "Missing/limit-truncated data is skipped and counted, never "
    "interpolated: availability is measured only over entries whose "
    "fetched strikes could construct at least one vertical.",
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


def _close_of(bar: dict) -> float | None:
    """Positive float close, or None for a missing/unusable print."""
    try:
        close = float(bar["c"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(close) or close <= 0:
        return None
    return close


def load_underlying_closes(data_dir: Path, symbol: str) -> dict[date, float]:
    path = _find_file(data_dir, [f"underlying_{symbol}.json"])
    if path is None:
        hits = sorted(data_dir.glob(f"underlying_{symbol}_*.json"))
        if not hits:
            raise FileNotFoundError(f"underlying_{symbol}.json in {data_dir}")
        path = hits[0]
    closes: dict[date, float] = {}
    for bar in _bars_of(json.loads(path.read_text())):
        close = _close_of(bar)
        if close is None:
            continue  # skip, never fabricate
        closes[parse_date(bar["t"])] = close
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


def load_put_contracts(data_dir: Path, symbol: str,
                       expiry: date) -> tuple[dict[float, dict] | None, dict]:
    """(strike -> {ticker, bars: {date: bar}}, file diagnostics).

    Returns (None, diag) when the batch file was never fetched — a data gap,
    not an error. Unparsable lines, call rows, strike-less rows and
    non-positive closes are skipped and counted.
    """
    iso, compact = _expiry_names(expiry)
    diag = {"file": None, "rows": 0, "bad_json": 0, "calls": 0,
            "no_strike": 0, "empty_bars": 0, "bad_bars": 0,
            "dup_strikes": 0, "contracts": 0, "contracts_with_bars": 0}
    path = _find_file(data_dir, [f"contracts_{symbol}_{iso}.jsonl",
                                 f"contracts_{symbol}_{compact}.jsonl"])
    if path is None:
        return None, diag
    diag["file"] = path.name
    listing = load_listing(data_dir, symbol, expiry)
    puts: dict[float, dict] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                diag["bad_json"] += 1
                continue
            diag["rows"] += 1
            ticker = row.get("ticker", "")
            ctype, occ_strike = _classify_contract(ticker, listing)
            if ctype == "call":
                diag["calls"] += 1
                continue  # sleeve replays short PUT verticals only
            strike = row.get("strike", occ_strike)
            if strike is None:
                diag["no_strike"] += 1
                continue
            try:
                strike = float(strike)
            except (TypeError, ValueError):
                diag["no_strike"] += 1
                continue
            raw_bars = row.get("bars") or []
            if not raw_bars:
                diag["empty_bars"] += 1
            bars: dict[date, dict] = {}
            for b in raw_bars:
                if not isinstance(b, dict) or "t" not in b:
                    diag["bad_bars"] += 1
                    continue
                try:
                    day = parse_date(b["t"])
                except (ValueError, TypeError, OSError, OverflowError):
                    diag["bad_bars"] += 1
                    continue
                if _close_of(b) is None:
                    diag["bad_bars"] += 1
                    continue
                bars[day] = b
            if strike in puts:  # duplicate contract row: union the bars
                diag["dup_strikes"] += 1
                puts[strike]["bars"].update(bars)
                continue
            puts[strike] = {"ticker": ticker, "bars": bars}
    diag["contracts"] = len(puts)
    diag["contracts_with_bars"] = sum(1 for c in puts.values() if c["bars"])
    return puts, diag


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
    """Entry-day close for a contract, or None when there is no usable print.

    min_trades applies the short-leg liquidity screen (bar trade count n).
    """
    if contract is None:
        return None
    bar = contract["bars"].get(entry_date)
    if bar is None:
        return None
    if min_trades is not None:
        try:
            n_trades = float(bar.get("n", 0) or 0)
        except (TypeError, ValueError):
            n_trades = 0.0
        if n_trades < min_trades:
            return None
    return _close_of(bar)


def count_constructible(entry_date: date, puts: dict[float, dict]) -> int:
    """How many strikes have an entry-day print AND a 5- or 10-wide wing with
    one. Zero means the fetched strike set cannot form ANY vertical, i.e. the
    entry is data-limited rather than gate-rejected."""
    have = {s for s in puts if _entry_mark(puts[s], entry_date) is not None}
    return sum(1 for s in have if any((s - w) in have for w in WIDTHS))


def select_vertical(entry_date: date, expiry: date, spot: float,
                    puts: dict[float, dict], q: float,
                    r: float = RISK_FREE_RATE,
                    width_policy: str = "five-first",
                    credit_floor_frac: float = CREDIT_FLOOR_FRAC) -> dict:
    """Replay the gate stack on one (entry_date, symbol, expiry).

    width_policy controls how the $5-10 width freedom in the sleeve spec is
    resolved once an in-band short leg is chosen:
      "five-first" (default, engine's original rule) — take the $5 wing; only
        widen to $10 when the $5 wing has no entry-day print. A $5 wing that
        exists but leaves credit under the floor does NOT get widened.
      "any" — accept the narrowest width in WIDTHS whose credit clears the
        floor. This is the literal reading of the spec's "width $5-10" and is
        reported as a sensitivity, never as the primary result.

    Returns {qualified: bool, reason: str|None, diag: {...}, ...selection}.
    Failure reasons, least- to most-progressed:
      no-entry-day-mark (data)   -> no strike had a usable entry-day print
      short-leg-illiquid (gate)  -> prints exist but all fail n >= 30
      no-invertible-mark (data)  -> liquid prints exist, none invert to an IV
      no-strike-in-band (gate)   -> no |delta| in [0.20, 0.35]
      wing-missing (data-limited)-> in-band short, but no wing print fetched
      inverted-credit-mark (data)-> wing priced above the short (stale print)
      credit-below-floor (gate)  -> vertical exists, credit < width/4
    """
    if width_policy not in ("five-first", "any"):
        raise ValueError(f"unknown width_policy {width_policy!r}")
    t_years = (expiry - entry_date).days / 365.0
    diag = {"n_contracts": len(puts), "n_entry_mark": 0, "n_liquid": 0,
            "n_iv_ok": 0, "n_in_band": 0, "n_constructible": 0,
            "wing_unfetched": 0, "wing_no_bar": 0, "n_bad_credit": 0,
            "width_policy": width_policy, "best_credit_frac": None}
    diag["n_constructible"] = count_constructible(entry_date, puts)
    in_band: list[dict] = []
    for strike in sorted(puts):
        mark = _entry_mark(puts[strike], entry_date)
        if mark is None:
            continue
        diag["n_entry_mark"] += 1
        if _entry_mark(puts[strike], entry_date, MIN_SHORT_TRADES) is None:
            continue
        diag["n_liquid"] += 1
        iv = implied_vol_put(mark, spot, strike, t_years, r, q)
        if iv is None:
            continue
        diag["n_iv_ok"] += 1
        delta = bs_put_delta(spot, strike, t_years, r, q, iv)
        if DELTA_LO <= abs(delta) <= DELTA_HI:
            diag["n_in_band"] += 1
            in_band.append({"strike": strike, "mark": mark, "iv": iv,
                            "delta": delta})
    if diag["n_entry_mark"] == 0:
        return {"qualified": False, "reason": "no-entry-day-mark", "diag": diag}
    if diag["n_liquid"] == 0:
        return {"qualified": False, "reason": "short-leg-illiquid", "diag": diag}
    if diag["n_iv_ok"] == 0:
        return {"qualified": False, "reason": "no-invertible-mark", "diag": diag}
    if not in_band:
        return {"qualified": False, "reason": "no-strike-in-band", "diag": diag}

    # Deterministic preference: |delta| closest to the band midpoint,
    # higher strike breaking ties.
    in_band.sort(key=lambda c: (abs(abs(c["delta"]) - TARGET_DELTA), -c["strike"]))
    any_wing = False
    any_sane_credit = False
    for cand in in_band:
        short_strike = cand["strike"]
        for width in WIDTHS:  # 5 preferred, 10 only if the 5-wing has no bar
            long_strike = short_strike - width
            wing = puts.get(long_strike)
            if wing is None:
                diag["wing_unfetched"] += 1
                continue
            long_mark = _entry_mark(wing, entry_date)
            if long_mark is None:
                diag["wing_no_bar"] += 1
                continue
            any_wing = True
            credit = cand["mark"] - long_mark
            if credit <= 0 or credit >= width:
                # Inverted / impossible mark pair: a stale print, not a
                # tradeable rejection. Skip this candidate, never repair it.
                diag["n_bad_credit"] += 1
                break
            any_sane_credit = True
            frac = credit / width
            if diag["best_credit_frac"] is None or frac > diag["best_credit_frac"]:
                diag["best_credit_frac"] = frac
            if credit >= width * credit_floor_frac:
                return {
                    "qualified": True, "reason": None, "diag": diag,
                    "short_strike": short_strike, "long_strike": long_strike,
                    "width": width, "credit": credit,
                    "short_mark": cand["mark"], "long_mark": long_mark,
                    "short_iv": cand["iv"], "short_delta": cand["delta"],
                    "short_ticker": puts[short_strike]["ticker"],
                    "long_ticker": puts[long_strike]["ticker"],
                }
            if width_policy == "five-first":
                break  # a wing with a bar existed; do not widen for credit
    if not any_wing:
        reason = "wing-missing"
    elif not any_sane_credit:
        reason = "inverted-credit-mark"
    else:
        reason = "credit-below-floor"
    return {"qualified": False, "reason": reason, "diag": diag}


# ---------------------------------------------------- managed payoff walk


def managed_walk(entry_date: date, expiry: date, credit: float,
                 short_bars: dict[date, dict],
                 long_bars: dict[date, dict]) -> dict:
    """Walk daily spread marks after entry until the first exit trigger.

    Days where either leg lacks a usable close are SKIPPED (not filled in);
    the count is reported as skipped_days. Exit priority on a given day:
    profit-take (mark <= 0.5*credit) first, then the forced 21-DTE exit;
    running out of bars exits at the last mark with data_end=True.
    """
    forced_date = expiry - timedelta(days=FORCED_EXIT_DTE)
    days = sorted(d for d in short_bars
                  if d in long_bars and entry_date < d <= expiry)
    last = None
    skipped = 0
    for d in days:
        s_close = _close_of(short_bars[d])
        l_close = _close_of(long_bars[d])
        if s_close is None or l_close is None:
            skipped += 1
            continue
        mark = s_close - l_close
        last = (d, mark, s_close, l_close)
        if mark <= PT_FRAC * credit:
            return {"exit_date": d, "exit_debit": PT_FRAC * credit,
                    "exit_reason": "profit_take", "data_end": False,
                    "short_exit_mark": s_close, "long_exit_mark": l_close,
                    "skipped_days": skipped, "walk_days": len(days)}
        if d >= forced_date:
            return {"exit_date": d, "exit_debit": mark,
                    "exit_reason": "dte_21", "data_end": False,
                    "short_exit_mark": s_close, "long_exit_mark": l_close,
                    "skipped_days": skipped, "walk_days": len(days)}
    if last is None:  # no post-entry bars at all: flat exit at entry mark
        return {"exit_date": entry_date, "exit_debit": credit,
                "exit_reason": "data_end", "data_end": True,
                "short_exit_mark": None, "long_exit_mark": None,
                "skipped_days": skipped, "walk_days": len(days)}
    d, mark, s_close, l_close = last
    return {"exit_date": d, "exit_debit": mark, "exit_reason": "data_end",
            "data_end": True, "short_exit_mark": s_close,
            "long_exit_mark": l_close, "skipped_days": skipped,
            "walk_days": len(days)}


def friction_dollars(spread_pct: float, short_entry: float, long_entry: float,
                     short_exit: float, long_exit: float) -> float:
    """Half-spread x 2 legs x 2 directions on the respective marks
    + $1 x 4 fees, per contract-set (x100 multiplier)."""
    half = spread_pct / 2.0
    crossed = half * (short_entry + long_entry + short_exit + long_exit) * 100.0
    return crossed + 4.0 * FEE_PER_LEG_PER_DIRECTION


# --------------------------------------------------------- per-entry replay

# Failure taxonomy. rank = how far down the funnel the entry got (the most
# progressed candidate expiry wins). klass = whether the DATA was missing or
# the GATE genuinely rejected a tradeable chain.
REASON_INFO: dict[str, tuple[int, str]] = {
    "no-data": (0, "data"),
    "no-contract-file": (1, "data"),
    "no-underlying-bar": (2, "data"),
    "dte-out-of-range": (3, "data"),
    "no-contract-bars": (4, "data"),
    "no-entry-day-mark": (5, "data"),
    "short-leg-illiquid": (6, "gate"),
    "no-invertible-mark": (7, "data"),
    "no-strike-in-band": (8, "gate"),
    "wing-missing": (9, "data"),
    "inverted-credit-mark": (10, "data"),
    "credit-below-floor": (11, "gate"),
}
_REASON_RANK = {k: v[0] for k, v in REASON_INFO.items()}
REASON_CLASS = {k: v[1] for k, v in REASON_INFO.items()}


def replay_entry(entry_date: date, symbol: str, expiries: list[date],
                 closes: dict[date, float],
                 puts_by_expiry: dict[date, dict | None],
                 spread_pct: float, q: float,
                 width_policy: str = "five-first",
                 credit_floor_frac: float = CREDIT_FLOOR_FRAC) -> dict:
    """One record per (entry_date, symbol): try candidate expiries in
    ascending DTE order, keep the first qualifying vertical; otherwise
    report the most-progressed failure reason plus the data funnel flags."""
    rec = {"entry_date": entry_date.isoformat(), "symbol": symbol,
           "qualified": False, "reason": "no-data",
           "marks_present": False, "data_adequate": False,
           "n_entry_marks": 0, "n_constructible": 0,
           "best_credit_frac": None}
    spot = closes.get(entry_date)
    if spot is None:
        rec["reason"] = "no-underlying-bar"
        rec["reason_class"] = "data"
        return rec
    rec["spot"] = spot
    best_reason = "no-data"
    for expiry in sorted(expiries):
        dte = (expiry - entry_date).days
        puts = puts_by_expiry.get(expiry)
        if puts is None:
            reason = "no-contract-file"
        elif not (DTE_RANGE[0] <= dte <= DTE_RANGE[1]):
            reason = "dte-out-of-range"
        elif not any(c["bars"] for c in puts.values()):
            reason = "no-contract-bars"
        else:
            sel = select_vertical(entry_date, expiry, spot, puts, q,
                                  width_policy=width_policy,
                                  credit_floor_frac=credit_floor_frac)
            diag = sel["diag"]
            rec["n_entry_marks"] = max(rec["n_entry_marks"], diag["n_entry_mark"])
            rec["n_constructible"] = max(rec["n_constructible"],
                                         diag["n_constructible"])
            rec["marks_present"] = rec["marks_present"] or diag["n_entry_mark"] > 0
            rec["data_adequate"] = rec["data_adequate"] or diag["n_constructible"] > 0
            if diag["best_credit_frac"] is not None:
                rec["best_credit_frac"] = max(
                    rec["best_credit_frac"] or 0.0, diag["best_credit_frac"])
            if sel["qualified"]:
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
                    "qualified": True, "reason": None, "reason_class": None,
                    "marks_present": True, "data_adequate": True,
                    "expiry": expiry.isoformat(), "dte": dte,
                    "exit_date": walk["exit_date"].isoformat(),
                    "exit_reason": walk["exit_reason"],
                    "exit_debit": walk["exit_debit"],
                    "data_end": walk["data_end"],
                    "walk_days": walk["walk_days"],
                    "skipped_days": walk["skipped_days"],
                    "pnl_gross": pnl_gross, "friction": friction,
                    "pnl_net": pnl_net, "risk_dollars": risk,
                    "result_r": pnl_net / risk,
                })
                return rec
            reason = sel["reason"]
        if _REASON_RANK[reason] > _REASON_RANK[best_reason]:
            best_reason = reason
    rec["reason"] = best_reason
    rec["reason_class"] = REASON_CLASS[best_reason]
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
    """Coverage/availability funnel for a slice of entry records.

    n_entries        planned (entry_date, symbol) pairs
    n_marks_present  at least one usable entry-day option print
    n_data_adequate  at least one CONSTRUCTIBLE vertical in the fetched data
    n_qualified      the gate stack produced a tradeable vertical

    data_coverage_rate = n_data_adequate / n_entries
    availability_rate  = n_qualified / n_data_adequate   <- the >=60% test
    availability_rate_strict uses n_marks_present as the denominator, i.e.
    it charges wing-fetch gaps against the gate (conservative bound).
    """
    n = len(records)
    marks = sum(1 for r in records if r.get("marks_present"))
    adeq = sum(1 for r in records if r.get("data_adequate"))
    q = sum(1 for r in records if r["qualified"])
    return {
        "n_entries": n,
        "n_marks_present": marks,
        "n_data_adequate": adeq,
        "n_qualified": q,
        "data_coverage_rate": (adeq / n) if n else None,
        "availability_rate": (q / adeq) if adeq else None,
        "availability_rate_strict": (q / marks) if marks else None,
        "rate": (q / adeq) if adeq else None,  # back-compat alias
    }


def credit_floor_sensitivity(records: list[dict],
                             floor_used: float = CREDIT_FLOOR_FRAC,
                             grid: tuple[float, ...] = CREDIT_FLOOR_GRID) -> dict:
    """How availability would move if only the credit floor were retuned.

    Uses best_credit_frac = the best credit/width actually observed while
    scanning the delta-ordered candidates. EXACT for floors <= the floor the
    run used (a qualifying entry already clears every lower floor, and a
    rejected entry's scan is exhaustive); the SELECTED strike/width — and
    therefore the payoff — can differ, so this projects availability only,
    never P&L.
    """
    adequate = [r for r in records if r.get("data_adequate")]
    out = {}
    for floor in grid:
        if floor > floor_used:
            continue
        n_q = sum(1 for r in adequate
                  if r["qualified"]
                  or (r.get("best_credit_frac") is not None
                      and r["best_credit_frac"] >= floor))
        out[f"{floor:g}"] = {
            "n_data_adequate": len(adequate), "n_qualified": n_q,
            "availability_rate": (n_q / len(adequate)) if adequate else None,
        }
    return out


def summarize(records: list[dict], resamples: int, seed: int,
              spread_pct: dict[str, float],
              width_policy: str = "five-first",
              credit_floor_frac: float = CREDIT_FLOOR_FRAC) -> dict:
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
    class_counts = Counter(r.get("reason_class") for r in records
                           if not r["qualified"])

    def _payoff(sub: list[dict]) -> dict:
        arr = np.array([r["result_r"] for r in sub])
        if not arr.size:
            return {"n_trades": 0, "win_rate": None, "mean_r": None,
                    "median_r": None, "sum_r": 0.0, "net_pnl_dollars": 0.0}
        return {"n_trades": int(arr.size),
                "win_rate": float(np.mean(arr > 0)),
                "mean_r": float(arr.mean()),
                "median_r": float(np.median(arr)),
                "sum_r": float(arr.sum()),
                "net_pnl_dollars": float(sum(r["pnl_net"] for r in sub))}

    n_q = len(qualified)
    summary = {
        "config": {
            "risk_free_rate": RISK_FREE_RATE, "div_yield": DIV_YIELD,
            "spread_pct": spread_pct, "delta_band": [DELTA_LO, DELTA_HI],
            "widths": list(WIDTHS), "credit_floor_frac": credit_floor_frac,
            "credit_floor_frac_spec": CREDIT_FLOOR_FRAC,
            "min_short_trades": MIN_SHORT_TRADES, "dte_range": list(DTE_RANGE),
            "pt_frac": PT_FRAC, "forced_exit_dte": FORCED_EXIT_DTE,
            "bootstrap_resamples": resamples, "bootstrap_seed": seed,
            "fees_per_round_trip": 4.0 * FEE_PER_LEG_PER_DIRECTION,
            "availability_floor": AVAILABILITY_FLOOR,
            "width_policy": width_policy,
        },
        "approximation_notes": APPROXIMATION_NOTES,
        "availability": {
            "acceptance_criterion":
                f">={AVAILABILITY_FLOOR:.0%} of DATA-ADEQUATE entry days must "
                "have a qualifying vertical (revival plan M1-0.4). Entries "
                "whose option data was never fetched are excluded from the "
                "denominator and reported as data_coverage_rate instead.",
            "overall": _availability(records),
            "per_symbol": {s: _availability(v) for s, v in sorted(by_symbol.items())},
            "per_month": {m: _availability(v) for m, v in sorted(by_month.items())},
            "unqualified_reasons": dict(sorted(reason_counts.items())),
            "unqualified_reason_classes": {
                k: v for k, v in sorted(class_counts.items(), key=lambda kv: str(kv[0]))},
            "reason_classes": REASON_CLASS,
            "credit_floor_sensitivity": credit_floor_sensitivity(
                records, credit_floor_frac),
        },
        "managed_payoff": {
            "n_trades": n_q,
            "win_rate": float(np.mean(rs > 0)) if n_q else None,
            "mean_r": float(rs.mean()) if n_q else None,
            "median_r": float(np.median(rs)) if n_q else None,
            "std_r": float(rs.std(ddof=1)) if n_q > 1 else None,
            "sum_r": float(rs.sum()) if n_q else 0.0,
            "net_pnl_dollars": float(sum(r["pnl_net"] for r in qualified)),
            "lb95_mean_r": block_bootstrap_lb95_mean(qualified, resamples, seed),
            "max_drawdown_r": max_drawdown_r(qualified),
            "per_symbol": {s: _payoff([r for r in v if r["qualified"]])
                           for s, v in sorted(by_symbol.items())},
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
            "mean_r_by_exit": {
                k: (float(np.mean([r["result_r"] for r in qualified
                                   if r["exit_reason"] == k]))
                    if exit_counts.get(k) else None)
                for k in ("profit_take", "dte_21", "data_end")},
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
        seed: int = BOOTSTRAP_SEED, width_policy: str = "five-first",
        credit_floor_frac: float = CREDIT_FLOOR_FRAC) -> dict:
    plan = load_batch_plan(batch_plan)

    # group candidate expiries per (entry_date, symbol)
    grouped: dict[tuple[date, str], list[date]] = defaultdict(list)
    for row in plan:
        key = (row["entry_date"], row["symbol"])
        if row["expiry"] not in grouped[key]:
            grouped[key].append(row["expiry"])

    closes_cache: dict[str, dict[date, float]] = {}
    puts_cache: dict[tuple[str, date], dict | None] = {}
    file_diags: dict[str, dict] = {}
    records: list[dict] = []
    for (entry_date, symbol), expiries in sorted(grouped.items()):
        if symbol not in closes_cache:
            closes_cache[symbol] = load_underlying_closes(data_dir, symbol)
        for expiry in expiries:
            if (symbol, expiry) not in puts_cache:
                puts, diag = load_put_contracts(data_dir, symbol, expiry)
                puts_cache[(symbol, expiry)] = puts
                file_diags[f"{symbol}_{expiry.isoformat()}"] = diag
        if symbol not in spread_pct:
            raise ValueError(f"no --spread-pct entry for {symbol}")
        records.append(replay_entry(
            entry_date, symbol, expiries, closes_cache[symbol],
            {e: puts_cache[(symbol, e)] for e in expiries},
            spread_pct[symbol], DIV_YIELD.get(symbol, 0.0), width_policy,
            credit_floor_frac))

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "entries.jsonl").open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    summary = summarize(records, resamples, seed, spread_pct, width_policy,
                        credit_floor_frac)
    summary["data_inventory"] = {
        "batches_planned": len(file_diags),
        "batches_missing_file": sum(1 for d in file_diags.values()
                                    if d["file"] is None),
        "batches_no_bars": sum(1 for d in file_diags.values()
                               if d["file"] is not None
                               and d["contracts_with_bars"] == 0),
        "contract_rows": sum(d["contracts"] for d in file_diags.values()),
        "contract_rows_with_bars": sum(d["contracts_with_bars"]
                                       for d in file_diags.values()),
        "skipped_bad_json_lines": sum(d["bad_json"] for d in file_diags.values()),
        "skipped_bad_bars": sum(d["bad_bars"] for d in file_diags.values()),
        "skipped_call_rows": sum(d["calls"] for d in file_diags.values()),
        "skipped_no_strike_rows": sum(d["no_strike"] for d in file_diags.values()),
        "duplicate_strike_rows": sum(d["dup_strikes"] for d in file_diags.values()),
        "per_batch": {k: file_diags[k] for k in sorted(file_diags)},
    }
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
    ap.add_argument("--width-policy", choices=("five-first", "any"),
                    default="five-first",
                    help="how to resolve the spec's $5-10 width freedom "
                         "(default five-first = engine's original rule)")
    ap.add_argument("--credit-floor-frac", type=float, default=CREDIT_FLOOR_FRAC,
                    help="min credit as a fraction of width (spec: 0.25); "
                         "lower values are RETUNE SCENARIOS, not the spec")
    args = ap.parse_args(argv)

    summary = run(args.data_dir, args.batch_plan, args.out_dir,
                  parse_spread_pct(args.spread_pct), args.resamples, args.seed,
                  args.width_policy, args.credit_floor_frac)
    av = summary["availability"]["overall"]
    mp = summary["managed_payoff"]
    inv = summary["data_inventory"]

    def _r(x):
        return None if x is None else round(x, 4)

    print(f"entries={av['n_entries']} marks_present={av['n_marks_present']} "
          f"data_adequate={av['n_data_adequate']} qualified={av['n_qualified']}")
    print(f"data_coverage_rate={_r(av['data_coverage_rate'])} "
          f"availability_rate={_r(av['availability_rate'])} "
          f"(strict={_r(av['availability_rate_strict'])})")
    print(f"trades={mp['n_trades']} WR={_r(mp['win_rate'])} "
          f"mean_r={_r(mp['mean_r'])} median_r={_r(mp['median_r'])} "
          f"lb95_mean_r={_r(mp['lb95_mean_r'])} maxDD_r={_r(mp['max_drawdown_r'])}")
    print(f"unqualified reasons: {summary['availability']['unqualified_reasons']}")
    print("credit-floor sensitivity (availability): "
          + ", ".join(f"{k}->{_r(v['availability_rate'])}" for k, v in
                      summary["availability"]["credit_floor_sensitivity"].items()))
    print(f"data: batches={inv['batches_planned']} missing_file="
          f"{inv['batches_missing_file']} no_bars={inv['batches_no_bars']} "
          f"contracts={inv['contract_rows']} with_bars="
          f"{inv['contract_rows_with_bars']}")
    print(f"wrote {args.out_dir / 'entries.jsonl'} and {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
