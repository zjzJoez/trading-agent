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

INFORMATION SET (ONE, declared once, applied to every gate — see
INFORMATION_SET below): ENTER-AT-THE-CLOSE. The decision is taken on the
completed entry-day daily bar and filled at that bar's close, so every field
of that bar — close, high, low, vwap and the full-day trade count n — is
inside the information set, and no bar dated after the entry day feeds
selection anywhere. The 'prior' liquidity screen is therefore a robustness
choice (it also survives a decide-before-the-close reading), NOT a look-ahead
fix; 'same-day' is legal under the declared convention.

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
import hashlib
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

# --- daily mark conventions (B1) -----------------------------------------
# A daily aggregate offers several equally defensible "the price that day"
# proxies. They are NOT interchangeable for a 5-wide vertical: the spread mark
# is a DIFFERENCE of two legs, so any leg-vs-leg timing inconsistency lands
# directly on the credit. The four supported conventions:
#   close  last trade of the day, per leg   (the original engine rule)
#   vw     the bar's volume-weighted average price, per leg
#   hl2    (high + low) / 2, per leg
#   smile  reprice BOTH legs off ONE same-day IV curve fitted to that day's
#          liquid closes  -> cross-leg consistent by construction
# `smile` is the PRIMARY convention because it is the only one that is
# cross-leg consistent; `close` is retained and reported as a disclosed UPPER
# BOUND (it is the convention that produced the original, rejected numbers).
MARK_CONVENTIONS = ("smile", "close", "vw", "hl2")
PRIMARY_MARK = "smile"
UPPER_BOUND_MARK = "close"
SMILE_MIN_POINTS = 5        # strikes needed to fit a same-day curve
SMILE_MIN_TRADES = 10       # per-strike trade count to enter the fit
SMILE_IV_BOUNDS = (0.03, 2.0)

# --- THE DECLARED INFORMATION SET (B7) -----------------------------------
# ONE information set governs this whole engine, and every gate is judged
# against it:
#
#   ENTER-AT-THE-CLOSE. The decision is taken once the entry day's daily bar
#   is complete, and the fill is at that same bar's close.
#
# Consequences, applied consistently:
#   * Entry-day leg closes as selection marks: LEGAL. A close-to-close model
#     is a convention, not future information.
#   * The entry day's own FULL-DAY trade count n: also LEGAL. It is part of
#     the same completed bar as the closes the selection already reads. It is
#     therefore WRONG to call the "same-day" screen look-ahead while pricing
#     the trade off the same bar — that was the artifact's one self-
#     contradiction, and this is where it is resolved.
#   * Same-day full-day HIGH/LOW/VWAP (the hl2 and vw marks): LEGAL for the
#     same reason.
#   * What is NOT legal under any reading, and does not occur anywhere here:
#     any bar dated after the entry day feeding the entry decision.
#
# So why is "prior" still the PRIMARY screen? Not as a look-ahead fix, but as
# a ROBUSTNESS choice: it is the only variant that is also valid under the
# stricter decide-before-the-close information set, under which the entry
# day's own bar does not exist yet. Selecting on the prior session's trade
# count costs one trade and buys validity under both readings. The cost of
# the choice is measured and published (sensitivity row "liquidity-same-day").
#
# NOTE the asymmetry this leaves, stated rather than hidden: this engine does
# NOT implement a decide-before-the-close variant of the MARKS. Under that
# stricter information set the entry-day closes would be unknowable too, and
# nothing in this replay would be legal. That variant is not measured here.
INFORMATION_SET = "enter-at-the-close"
INFORMATION_SET_STATEMENT = (
    "ENTER-AT-THE-CLOSE: the decision is taken on the completed entry-day "
    "daily bar and filled at that bar's close. Every field of that bar "
    "(close, high, low, vwap, full-day trade count n) is therefore inside "
    "the information set; no bar dated after the entry day feeds selection "
    "anywhere. The 'prior' liquidity screen is a stricter-than-required "
    "robustness choice, not a look-ahead fix; the 'same-day' screen is legal "
    "under this declared convention. A decide-BEFORE-the-close variant of "
    "the marks is not implemented and not measured."
)
LIQUIDITY_LAGS = ("prior", "same-day")
PRIMARY_LIQUIDITY_LAG = "prior"

# --- the acceptance criterion this replay was commissioned to test --------
PLAN_CRITERION_SOURCE = "docs/REVIVAL_PLAN_2026-07-20.md line 81 (M1-0.4)"
PLAN_CRITERION_TEXT = (
    "验收:**允许交易的 regime 内合格 vertical 存在于 ≥60% 快照日** — "
    "i.e. a qualifying vertical must exist on >=60% of snapshot days "
    "WITHIN THE REGIMES THE SLEEVE IS ALLOWED TO TRADE."
)
CRITERION_NOT_EVALUATED_REASON = (
    "NOT EVALUATED AS SPECIFIED: this replay carries no regime labels, so "
    "availability was measured UNCONDITIONALLY over all planned entry days. "
    "The criterion is conditional on the allowed regimes; an unconditional "
    "rate is neither the criterion nor a bound on it."
)

DELTA_LO, DELTA_HI = 0.20, 0.35
TARGET_DELTA = 0.5 * (DELTA_LO + DELTA_HI)            # deterministic tie-break
WIDTHS = (5.0, 10.0)                                  # 5 preferred, 10 fallback
CREDIT_FLOOR_FRAC = 0.25                              # credit >= width/4
CREDIT_FLOOR_GRID = (0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25)
# The credit is a DIFFERENCE of two decimal-quoted closes, so a spread that is
# exactly on the spec boundary need not compare equal to it in binary. 8.20 -
# 6.95 = 1.2499999999999991 on a $5 width -> credit/width 0.24999999999999983,
# which a bare `>=` rejects while an identical 1.25 quoted directly passes.
# One real entry (2026-01-02 QQQ 600/595) was rejected for exactly this, so
# the gate and the credit-floor projection both compare with this tolerance.
CREDIT_FLOOR_TOL = 1e-9
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
    "ONE information set: enter-at-the-close. The completed entry-day bar "
    "(close/high/low/vwap/n) is knowable; nothing dated after it feeds "
    "selection. The 'prior' liquidity screen is stricter than this requires, "
    "not a look-ahead fix. A decide-before-the-close variant of the MARKS is "
    "not implemented and not measured.",
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


_MARK_CONVENTION = PRIMARY_MARK


def set_mark_convention(name: str) -> str:
    """Select the active daily mark convention; returns the previous one."""
    global _MARK_CONVENTION
    if name not in MARK_CONVENTIONS:
        raise ValueError(f"unknown mark convention {name!r}; "
                         f"expected one of {MARK_CONVENTIONS}")
    prev, _MARK_CONVENTION = _MARK_CONVENTION, name
    return prev


def mark_convention() -> str:
    return _MARK_CONVENTION


def _positive(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    return v


def _raw_close(bar: dict) -> float | None:
    """Positive float close, or None for a missing/unusable print."""
    try:
        return _positive(bar["c"])
    except (KeyError, TypeError):
        return None


def _close_of(bar: dict) -> float | None:
    """The bar's mark under the active convention, or None if unusable.

    "smile" reads the same field as "close" — the cross-leg-consistent
    repricing happens once per chain in apply_smile_repricing(), which
    OVERWRITES each bar's close with the curve-implied price. A convention
    whose own field is missing/unusable falls back to the raw close and the
    fallback is visible in the per-run mark diagnostics.
    """
    conv = _MARK_CONVENTION
    if conv in ("close", "smile"):
        return _raw_close(bar)
    if conv == "vw":
        v = _positive(bar.get("vw"))
    else:  # hl2
        try:
            v = _positive((float(bar["h"]) + float(bar["l"])) / 2.0)
        except (KeyError, TypeError, ValueError):
            v = None
    return v if v is not None else _raw_close(bar)


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


# ------------------------------------------- cross-leg-consistent marks


def fit_iv_curve(puts: dict[float, dict], day: date, spot: float,
                 t_years: float, q: float,
                 min_points: int = SMILE_MIN_POINTS,
                 min_trades: int = SMILE_MIN_TRADES) -> np.ndarray | None:
    """Weighted quadratic-in-moneyness IV curve for ONE day of ONE chain.

    Fits sigma(K/S) = a + b*m + c*m^2 to every strike whose bar has at least
    `min_trades` trades and inverts to a sane IV, weighting by sqrt(trades).
    None when fewer than `min_points` usable strikes exist — the day then
    keeps its raw closes and is counted, never fabricated.
    """
    if spot is None or t_years <= 0.005:
        return None
    lo, hi = SMILE_IV_BOUNDS
    xs, ys, ws = [], [], []
    for strike, contract in puts.items():
        bar = contract["bars"].get(day)
        if bar is None:
            continue
        px = _close_of(bar)
        if px is None or _trade_count(bar) < min_trades:
            continue
        iv = implied_vol_put(px, spot, strike, t_years, RISK_FREE_RATE, q)
        if iv is None or not (lo < iv < hi):
            continue
        xs.append(strike / spot)
        ys.append(iv)
        ws.append(math.sqrt(_trade_count(bar)))
    if len(xs) < min_points:
        return None
    x = np.asarray(xs)
    w = np.asarray(ws)
    design = np.vstack([np.ones_like(x), x, x * x]).T * w[:, None]
    coef, *_ = np.linalg.lstsq(design, np.asarray(ys) * w, rcond=None)
    return coef


def apply_smile_repricing(puts: dict[float, dict], expiry: date, q: float,
                          closes: dict[date, float]) -> dict:
    """Overwrite every bar's close with a same-day-curve-consistent price.

    Both legs of any vertical then come off ONE IV curve, so the credit no
    longer inherits the leg-vs-leg last-trade timing gap. Days with no
    fittable curve, and strikes the curve extrapolates to an insane IV, keep
    their raw close and are counted in the returned stats.

    The kept-raw count is SPLIT by cause, because only one of the two can
    break cross-leg consistency (finding 12):

      bars_kept_raw_on_unfittable_days  the whole day had no curve, so EVERY
                                        strike that day kept its raw close.
                                        A pair drawn from such a day is
                                        raw+raw — one convention, no mixing.
      bars_kept_raw_on_fitted_days      a curve existed but THIS strike
                                        extrapolated to an insane IV. This is
                                        the ONLY way a curve-priced leg can
                                        end up paired with a raw-priced leg,
                                        so if this count is 0 no mixed pair
                                        is possible anywhere in the run.

    Every kept-raw bar is also stamped `smile_kept_raw=True` (and every
    repriced bar `smile_repriced=True`) so a specific trade's marked days can
    be audited after the fact — see raw_mark_exposure().
    """
    stats = {"days_fitted": 0, "days_unfittable": 0,
             "bars_repriced": 0, "bars_kept_raw": 0,
             "bars_kept_raw_on_unfittable_days": 0,
             "bars_kept_raw_on_fitted_days": 0}
    lo, hi = SMILE_IV_BOUNDS
    days: set[date] = set()
    for contract in puts.values():
        days |= set(contract["bars"])
    for day in sorted(days):
        spot = closes.get(day)
        t_years = (expiry - day).days / 365.0
        coef = fit_iv_curve(puts, day, spot, t_years, q)
        if coef is None:
            stats["days_unfittable"] += 1
            for contract in puts.values():
                bar = contract["bars"].get(day)
                if bar is None:
                    continue
                stats["bars_kept_raw"] += 1
                stats["bars_kept_raw_on_unfittable_days"] += 1
                contract["bars"][day] = {**bar, "smile_kept_raw": True,
                                         "smile_kept_raw_cause": "day-unfittable"}
            continue
        stats["days_fitted"] += 1
        for strike, contract in puts.items():
            bar = contract["bars"].get(day)
            if bar is None:
                continue
            m = strike / spot
            iv = float(coef[0] + coef[1] * m + coef[2] * m * m)
            px = (bs_put_price(spot, strike, t_years, RISK_FREE_RATE, q, iv)
                  if lo < iv < hi else None)
            if px is None or px <= 0:
                stats["bars_kept_raw"] += 1
                stats["bars_kept_raw_on_fitted_days"] += 1
                contract["bars"][day] = {
                    **bar, "smile_kept_raw": True,
                    "smile_kept_raw_cause": "strike-extrapolated-insane-iv"}
                continue
            repriced = dict(bar)
            repriced["c"] = px
            repriced["smile_repriced"] = True
            contract["bars"][day] = repriced
            stats["bars_repriced"] += 1
    return stats


def raw_mark_exposure(qualified: list[dict],
                      puts_by_key: dict[tuple[str, date], dict | None]) -> dict:
    """Which booked trades touched a bar that kept its RAW close.

    Makes the report's cross-leg-consistency claim provable from the artifact
    instead of asserted (finding 12). For every qualified trade, walk the days
    that actually produced a mark it used — the entry day and every day of the
    managed walk up to and including the exit — and ask whether either leg's
    bar on that day was left raw by apply_smile_repricing().

    The distinction that matters is MIXED vs raw+raw:
      n_trades_with_any_raw_marked_leg_day  any leg-day left raw at all
      n_trades_with_a_mixed_pair_day        a day where ONE leg was repriced
                                            and the other was left raw. This
                                            is the only cross-leg-inconsistent
                                            case; the claim is that it is 0.
    """
    out = {
        "what": ("per-trade audit of whether a booked trade's marked leg-days "
                 "used a raw (un-repriced) close, and whether any single day "
                 "mixed a repriced leg with a raw one."),
        "n_trades": len(qualified),
        "n_trades_with_any_raw_marked_leg_day": 0,
        "n_trades_with_a_mixed_pair_day": 0,
        "n_marked_leg_days": 0,
        "n_marked_leg_days_raw": 0,
        "n_marked_pair_days": 0,
        "n_marked_pair_days_both_raw": 0,
        "n_marked_pair_days_mixed": 0,
        "trades_with_raw": [],
        "trades_with_mixed_pair": [],
    }
    for r in qualified:
        expiry = date.fromisoformat(r["expiry"])
        puts = puts_by_key.get((r["symbol"], expiry))
        if not puts:
            continue
        short_c = puts.get(r["short_strike"])
        long_c = puts.get(r["long_strike"])
        if short_c is None or long_c is None:
            continue
        entry = date.fromisoformat(r["entry_date"])
        exit_day = date.fromisoformat(r["exit_date"])
        days = sorted(d for d in short_c["bars"]
                      if d in long_c["bars"] and entry <= d <= exit_day)
        any_raw = mixed = False
        for d in days:
            s_raw = bool(short_c["bars"][d].get("smile_kept_raw"))
            l_raw = bool(long_c["bars"][d].get("smile_kept_raw"))
            out["n_marked_leg_days"] += 2
            out["n_marked_leg_days_raw"] += int(s_raw) + int(l_raw)
            out["n_marked_pair_days"] += 1
            if s_raw and l_raw:
                out["n_marked_pair_days_both_raw"] += 1
            elif s_raw or l_raw:
                out["n_marked_pair_days_mixed"] += 1
                mixed = True
            any_raw = any_raw or s_raw or l_raw
        key = {"entry_date": r["entry_date"], "symbol": r["symbol"],
               "short_strike": r["short_strike"],
               "long_strike": r["long_strike"]}
        if any_raw:
            out["n_trades_with_any_raw_marked_leg_day"] += 1
            out["trades_with_raw"].append(key)
        if mixed:
            out["n_trades_with_a_mixed_pair_day"] += 1
            out["trades_with_mixed_pair"].append(key)
    out["no_mixed_pair_anywhere"] = out["n_marked_pair_days_mixed"] == 0
    return out


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


def _trade_count(bar: dict) -> float:
    """The bar's trade count n, 0.0 when absent or unparsable."""
    try:
        return float(bar.get("n", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _prior_bar_day(contract: dict, entry_date: date) -> date | None:
    """Latest session with a usable bar STRICTLY BEFORE entry_date."""
    earlier = [d for d in contract["bars"] if d < entry_date]
    return max(earlier) if earlier else None


def short_leg_is_liquid(contract: dict, entry_date: date, min_trades: int,
                        lag: str = PRIMARY_LIQUIDITY_LAG) -> bool:
    """The n >= min_trades liquidity screen. See INFORMATION_SET above.

    Both variants are legal under the declared enter-at-the-close information
    set; they differ in how much they additionally survive.

    lag="prior"     screen the most recent session strictly BEFORE entry.
                    PRIMARY — stricter than the declared information set
                    requires, and the only variant that is also valid if the
                    decision were taken before the close.
    lag="same-day"  screen the entry day's own FULL-DAY trade count. This is
                    part of the same completed bar whose close the selection
                    already uses as its mark, so under the declared
                    convention it is NOT look-ahead. Reported as a
                    sensitivity so the cost of the stricter choice is visible.
    A contract with no prior session at all fails the "prior" screen: an
    untraded/newly-listed strike is exactly what the screen exists to reject.
    """
    if lag not in LIQUIDITY_LAGS:
        raise ValueError(f"unknown liquidity lag {lag!r}; expected {LIQUIDITY_LAGS}")
    if lag == "same-day":
        bar = contract["bars"].get(entry_date)
        return bar is not None and _trade_count(bar) >= min_trades
    day = _prior_bar_day(contract, entry_date)
    return day is not None and _trade_count(contract["bars"][day]) >= min_trades


def _entry_mark(contract: dict | None, entry_date: date,
                min_trades: int | None = None,
                liquidity_lag: str = PRIMARY_LIQUIDITY_LAG) -> float | None:
    """Entry-day mark for a contract, or None when there is no usable print.

    min_trades applies the short-leg liquidity screen; see
    short_leg_is_liquid() for what information set the screen may use.
    """
    if contract is None:
        return None
    bar = contract["bars"].get(entry_date)
    if bar is None:
        return None
    if min_trades is not None and not short_leg_is_liquid(
            contract, entry_date, min_trades, liquidity_lag):
        return None
    return _close_of(bar)


def count_constructible(entry_date: date, puts: dict[float, dict],
                        widths: tuple[float, ...] = WIDTHS) -> int:
    """How many strikes have an entry-day print AND a wing of one of `widths`
    with one. Zero means the fetched strike set cannot form ANY vertical, i.e.
    the entry is data-limited rather than gate-rejected."""
    have = {s for s in puts if _entry_mark(puts[s], entry_date) is not None}
    return sum(1 for s in have if any((s - w) in have for w in widths))


def select_vertical(entry_date: date, expiry: date, spot: float,
                    puts: dict[float, dict], q: float,
                    r: float = RISK_FREE_RATE,
                    width_policy: str = "five-first",
                    credit_floor_frac: float = CREDIT_FLOOR_FRAC,
                    liquidity_lag: str = PRIMARY_LIQUIDITY_LAG) -> dict:
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
      band-not-fetched (data)    -> the fetched strike grid does not BRACKET
                                    [0.20, 0.35]: min|delta| > 0.20 or
                                    max|delta| < 0.35, so the band was never
                                    observable. A data gap, not a rejection.
      no-strike-in-band (gate)   -> the grid DOES bracket the band and still no
                                    strike lands inside it (a real rejection)
      wing-missing (data-limited)-> in-band short, but no wing print fetched
      inverted-credit-mark (data)-> wing priced above the short (stale print)
      credit-below-floor (gate)  -> vertical exists, credit < width/4
    """
    if width_policy not in ("five-first", "any"):
        raise ValueError(f"unknown width_policy {width_policy!r}")
    t_years = (expiry - entry_date).days / 365.0
    diag = {"n_contracts": len(puts), "n_entry_mark": 0, "n_liquid": 0,
            "n_iv_ok": 0, "n_in_band": 0, "n_constructible": 0,
            "n_constructible_by_width": {}, "wing_unfetched": 0,
            "wing_no_bar": 0, "n_bad_credit": 0, "width_policy": width_policy,
            "liquidity_lag": liquidity_lag, "credit_floor_frac": credit_floor_frac,
            "mark_convention": _MARK_CONVENTION, "best_credit_frac": None,
            "best_credit_frac_by_width": {}, "delta_min": None,
            "delta_max": None, "brackets_delta_band": None,
            # The STRIKE grid behind delta_min/delta_max (finding 5). Without
            # it a band-not-fetched row cannot be read: "n invertible strikes"
            # alone does not say where they were, and n_contracts (how many
            # were FETCHED) is a different number from how many survived
            # liquidity + invertibility.
            "strike_min_fetched": (min(puts) if puts else None),
            "strike_max_fetched": (max(puts) if puts else None),
            "strike_min_invertible": None, "strike_max_invertible": None}
    diag["n_constructible"] = count_constructible(entry_date, puts)
    diag["n_constructible_by_width"] = {
        f"{w:g}": count_constructible(entry_date, puts, widths=(w,))
        for w in WIDTHS}
    in_band: list[dict] = []
    deltas: list[float] = []
    invertible_strikes: list[float] = []
    for strike in sorted(puts):
        mark = _entry_mark(puts[strike], entry_date)
        if mark is None:
            continue
        diag["n_entry_mark"] += 1
        if _entry_mark(puts[strike], entry_date, MIN_SHORT_TRADES,
                       liquidity_lag) is None:
            continue
        diag["n_liquid"] += 1
        iv = implied_vol_put(mark, spot, strike, t_years, r, q)
        if iv is None:
            continue
        diag["n_iv_ok"] += 1
        invertible_strikes.append(strike)
        delta = bs_put_delta(spot, strike, t_years, r, q, iv)
        deltas.append(abs(delta))
        if DELTA_LO <= abs(delta) <= DELTA_HI:
            diag["n_in_band"] += 1
            in_band.append({"strike": strike, "mark": mark, "iv": iv,
                            "delta": delta})
    if invertible_strikes:
        diag["strike_min_invertible"] = min(invertible_strikes)
        diag["strike_max_invertible"] = max(invertible_strikes)
    if deltas:
        diag["delta_min"] = min(deltas)
        diag["delta_max"] = max(deltas)
        # The band is OBSERVABLE only if the fetched, liquid, invertible grid
        # straddles it. Otherwise "no strike in band" says nothing about the
        # gate — it says the strikes that would answer the question were
        # never fetched.
        diag["brackets_delta_band"] = (diag["delta_min"] <= DELTA_LO
                                       and diag["delta_max"] >= DELTA_HI)
    if diag["n_entry_mark"] == 0:
        return {"qualified": False, "reason": "no-entry-day-mark", "diag": diag}
    if diag["n_liquid"] == 0:
        return {"qualified": False, "reason": "short-leg-illiquid", "diag": diag}
    if diag["n_iv_ok"] == 0:
        return {"qualified": False, "reason": "no-invertible-mark", "diag": diag}
    if not in_band:
        reason = ("no-strike-in-band" if diag["brackets_delta_band"]
                  else "band-not-fetched")
        return {"qualified": False, "reason": reason, "diag": diag}

    # Deterministic preference: |delta| closest to the band midpoint,
    # higher strike breaking ties.
    in_band.sort(key=lambda c: (abs(abs(c["delta"]) - TARGET_DELTA), -c["strike"]))
    any_wing = False
    any_sane_credit = False
    for cand in in_band:
        short_strike = cand["strike"]
        for width in WIDTHS:  # 5 preferred, 10 explored per width_policy
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
                # tradeable rejection. Skip THIS WIDTH and keep exploring —
                # a stale $5 wing print must not silently veto the $10 wing
                # (the original engine `break`-ed here, which is one of the
                # two reasons the 10-wide branch never fired).
                diag["n_bad_credit"] += 1
                continue
            any_sane_credit = True
            frac = credit / width
            if diag["best_credit_frac"] is None or frac > diag["best_credit_frac"]:
                diag["best_credit_frac"] = frac
            key = f"{width:g}"
            prev = diag["best_credit_frac_by_width"].get(key)
            if prev is None or frac > prev:
                diag["best_credit_frac_by_width"][key] = frac
            # Tolerant compare: an exactly-on-the-boundary credit must not be
            # rejected by float representation error (CREDIT_FLOOR_TOL).
            if credit >= width * credit_floor_frac - CREDIT_FLOOR_TOL:
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
                # A priced wing existed at the narrowest width; the spec's
                # default reading does not widen merely to reach the floor.
                break
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
    "band-not-fetched": (8, "data"),
    "no-strike-in-band": (9, "gate"),
    "wing-missing": (10, "data"),
    "inverted-credit-mark": (11, "data"),
    "credit-below-floor": (12, "gate"),
}
_REASON_RANK = {k: v[0] for k, v in REASON_INFO.items()}
REASON_CLASS = {k: v[1] for k, v in REASON_INFO.items()}


def replay_entry(entry_date: date, symbol: str, expiries: list[date],
                 closes: dict[date, float],
                 puts_by_expiry: dict[date, dict | None],
                 spread_pct: float, q: float,
                 width_policy: str = "five-first",
                 credit_floor_frac: float = CREDIT_FLOOR_FRAC,
                 liquidity_lag: str = PRIMARY_LIQUIDITY_LAG) -> dict:
    """One record per (entry_date, symbol): try candidate expiries in
    ascending DTE order, keep the first qualifying vertical; otherwise
    report the most-progressed failure reason plus the data funnel flags.

    The per-expiry selection diagnostics are persisted on EVERY record,
    qualified or not (diag = the diag of the expiry the record's reason came
    from, diag_by_expiry = all of them), so a rejection can be re-audited
    from entries.jsonl alone.
    """
    rec = {"entry_date": entry_date.isoformat(), "symbol": symbol,
           "qualified": False, "reason": "no-data",
           "marks_present": False, "data_adequate": False,
           "n_entry_marks": 0, "n_constructible": 0,
           "best_credit_frac": None, "brackets_delta_band": None,
           "diag": None, "diag_by_expiry": {},
           "mark_convention": _MARK_CONVENTION, "liquidity_lag": liquidity_lag}
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
                                  credit_floor_frac=credit_floor_frac,
                                  liquidity_lag=liquidity_lag)
            diag = sel["diag"]
            rec["diag_by_expiry"][expiry.isoformat()] = {
                **diag, "dte": dte, "reason": sel["reason"],
                "qualified": sel["qualified"]}
            rec["n_entry_marks"] = max(rec["n_entry_marks"], diag["n_entry_mark"])
            rec["n_constructible"] = max(rec["n_constructible"],
                                         diag["n_constructible"])
            rec["marks_present"] = rec["marks_present"] or diag["n_entry_mark"] > 0
            rec["data_adequate"] = rec["data_adequate"] or diag["n_constructible"] > 0
            if diag["brackets_delta_band"] is not None:
                rec["brackets_delta_band"] = bool(
                    rec["brackets_delta_band"]) or diag["brackets_delta_band"]
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
                    "short_exit_mark": walk["short_exit_mark"],
                    "long_exit_mark": walk["long_exit_mark"],
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
    # Persist the diag of the expiry the reported reason came from, so an
    # unqualified record is auditable from entries.jsonl alone (finding 11).
    for exp_iso, d in rec["diag_by_expiry"].items():
        if d["reason"] == best_reason:
            rec["diag"] = d
            rec["reason_expiry"] = exp_iso
            break
    return rec


# ------------------------------------------------------------- statistics


def entry_week_blocks(entries: list[dict]) -> dict[str, list[float]]:
    """Blocks = entry ISO week. Cheap, but a 1-week block cannot contain an
    18-day exposure, so simultaneously-open trades can land in DIFFERENT
    blocks and be resampled as if independent (finding 9)."""
    blocks: dict[str, list[float]] = defaultdict(list)
    for e in entries:
        iso = date.fromisoformat(e["entry_date"]).isocalendar()
        blocks[f"{iso[0]}-W{iso[1]:02d}"].append(e["result_r"])
    return dict(blocks)


def exposure_cluster_blocks(entries: list[dict]) -> dict[str, list[float]]:
    """Blocks = connected components of overlapping [entry, exit] intervals.

    Two trades that are open at the same moment share the same market shock,
    so they must never be drawn independently. Transitive closure of the
    overlap relation gives the coarsest honest block: every trade inside a
    component is resampled with the whole component.
    """
    items = sorted(entries, key=lambda e: (e["entry_date"], e["symbol"]))
    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    spans = [(date.fromisoformat(e["entry_date"]),
              date.fromisoformat(e.get("exit_date") or e["entry_date"]))
             for e in items]
    for i in range(n):
        for j in range(i + 1, n):
            if spans[i][0] <= spans[j][1] and spans[j][0] <= spans[i][1]:
                union(i, j)
    blocks: dict[str, list[float]] = defaultdict(list)
    for i, e in enumerate(items):
        root = find(i)
        blocks[f"cluster-{root:03d}@{items[root]['entry_date']}"].append(
            e["result_r"])
    return dict(blocks)


def _bootstrap_lb95(blocks: dict[str, list[float]], resamples: int,
                    seed: int) -> float | None:
    """5th percentile of the block-resampled mean; deterministic per seed."""
    keys = sorted(blocks)
    if not keys:
        return None
    sums = np.array([sum(blocks[k]) for k in keys], dtype=float)
    counts = np.array([len(blocks[k]) for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(resamples, len(keys)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.percentile(means, 5.0))


def block_bootstrap_lb95_mean(entries: list[dict],
                              resamples: int = BOOTSTRAP_RESAMPLES,
                              seed: int = BOOTSTRAP_SEED,
                              blocking: str = "entry-week") -> float | None:
    """LB95 of mean(result_r) under the named blocking scheme.

    blocking="entry-week"       entry ISO week (original)
    blocking="exposure-cluster" connected components of overlapping holds
    """
    if blocking == "entry-week":
        blocks = entry_week_blocks(entries)
    elif blocking == "exposure-cluster":
        blocks = exposure_cluster_blocks(entries)
    else:
        raise ValueError(f"unknown blocking {blocking!r}")
    return _bootstrap_lb95(blocks, resamples, seed)


def max_drawdown_r(entries: list[dict]) -> float:
    """Max peak-to-trough drop of the cumulative result_r path.

    ORDERING: (exit_date, entry_date, symbol). A drawdown is a property of
    when P&L is BOOKED, not of when the position was opened — an equity curve
    walked in entry order credits a trade's result before the days it was
    actually still open, which both mis-times and mis-measures the trough.
    Trades sharing an exit_date are ordered by (entry_date, symbol). That
    secondary key is a CONVENTION, not a measurement: at daily granularity the
    order in which same-day exits book is not observable, and it can move the
    within-day path (hence, in principle, the trough) even though it cannot
    move the day's net. The key is fixed so the number is reproducible, and
    the choice is disclosed here rather than left implicit. An entry with no
    exit_date falls back to its entry_date (only data_end/flat records lack
    one).

    Positive number in R units; 0.0 if empty.
    """
    ordered = sorted(entries, key=lambda e: (e.get("exit_date")
                                             or e["entry_date"],
                                             e["entry_date"], e["symbol"]))
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

    EVERY rate here is a fraction whose numerator is a SUBSET of its
    denominator. That is not automatic: `qualified` implies marks_present and
    data_adequate (the gate cannot fire without them), but it does NOT imply
    brackets_delta_band — the bracket test asks whether the fetched liquid
    invertible grid STRADDLES [0.20, 0.35], and an entry can qualify on an
    in-band strike while the grid stops short of one of the two edges. The
    band-bracketed rate therefore has its own numerator,
    n_qualified_band_bracketed, restricted to the denominator's own
    population. Dividing all qualified entries by the bracketed count is
    arithmetically invalid and can exceed 1.
    """
    n = len(records)
    marks = sum(1 for r in records if r.get("marks_present"))
    adeq = sum(1 for r in records if r.get("data_adequate"))
    brack = sum(1 for r in records
                if r.get("data_adequate") and r.get("brackets_delta_band"))
    q = sum(1 for r in records if r["qualified"])
    q_brack = sum(1 for r in records
                  if r["qualified"] and r.get("data_adequate")
                  and r.get("brackets_delta_band"))
    floor = AVAILABILITY_FLOOR
    rate_adeq = (q / adeq) if adeq else None
    rate_plan = (q / n) if n else None
    rate_brack = (q_brack / brack) if brack else None
    return {
        "n_entries": n,
        "n_marks_present": marks,
        "n_data_adequate": adeq,
        "n_band_bracketed": brack,
        "n_qualified": q,
        # The band-bracketed rate's OWN numerator (finding 1): qualified AND
        # data_adequate AND brackets_delta_band, i.e. the same population as
        # n_band_bracketed. n_qualified is NOT a subset of n_band_bracketed.
        "n_qualified_band_bracketed": q_brack,
        "data_coverage_rate": (adeq / n) if n else None,
        # BOTH denominators, side by side, each with its own verdict (finding 5).
        "availability_rate": rate_adeq,          # q / data-adequate
        "availability_rate_planned": rate_plan,  # q / all planned entry days
        "availability_rate_strict": (q / marks) if marks else None,
        "availability_rate_band_bracketed": rate_brack,
        "availability_rate_band_bracketed_definition": (
            "n_qualified_band_bracketed / n_band_bracketed — numerator and "
            "denominator are the SAME population (qualified AND "
            "data_adequate AND brackets_delta_band)"),
        "passes_floor_on_data_adequate": (None if rate_adeq is None
                                          else rate_adeq >= floor),
        "passes_floor_on_planned": (None if rate_plan is None
                                    else rate_plan >= floor),
        "passes_floor_on_band_bracketed": (None if rate_brack is None
                                           else rate_brack >= floor),
        "rate": rate_adeq,  # back-compat alias
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

    Compares with CREDIT_FLOOR_TOL, identically to the gate itself, so the
    projection at the floor the run used reproduces the run's own count.
    """
    adequate = [r for r in records if r.get("data_adequate")]
    out = {}
    for floor in grid:
        if floor > floor_used:
            continue
        n_q = sum(1 for r in adequate
                  if r["qualified"]
                  or (r.get("best_credit_frac") is not None
                      and r["best_credit_frac"] >= floor - CREDIT_FLOOR_TOL))
        out[f"{floor:g}"] = {
            "n_data_adequate": len(adequate), "n_qualified": n_q,
            "availability_rate": (n_q / len(adequate)) if adequate else None,
        }
    return out


# ------------------------------------------- friction provenance (B8)

# Candidate key names for "full quoted spread as a fraction of mid". Lane A is
# rewriting data/execution_costs.json, so read whatever is there rather than
# hardcoding one schema; an unrecognized file degrades to the compiled-in
# DEFAULT_SPREAD_PCT and says so loudly.
_SPREAD_KEYS_FULL = ("median_spread_pct_mid", "spread_pct_of_mid_median",
                     "full_spread_pct_of_mid", "median_full_spread_pct_mid",
                     "spread_pct_of_mid", "spread_pct")
_SPREAD_KEYS_HALF = ("opt_half_spread_pct_of_mark", "half_spread_pct_of_mark",
                     "median_half_spread_pct_mark")


def _first_fraction(blob: dict, keys: tuple[str, ...]) -> tuple[float, str] | None:
    for k in keys:
        v = blob.get(k)
        if isinstance(v, (int, float)) and 0 < float(v) < 1:
            return float(v), k
    return None


def load_spread_calibration(path: Path | None,
                            symbols: tuple[str, ...]) -> tuple[dict, dict]:
    """(spread_pct per symbol, provenance) from data/execution_costs.json.

    Per-underlying entry wins; then the file's global figure; then the
    compiled-in DEFAULT_SPREAD_PCT. Every fallback is recorded in provenance
    with a warning, so summary.json never presents an inherited default as a
    measurement (finding 14).
    """
    prov: dict = {"path": str(path) if path else None, "file_present": False,
                  "sha256": None, "calibrated_at": None, "file_globals": {},
                  "per_symbol": {}, "warnings": []}
    raw: dict = {}
    if path is not None and path.exists():
        try:
            text = path.read_text()
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                raw = loaded
                prov["file_present"] = True
                prov["sha256"] = hashlib.sha256(text.encode()).hexdigest()
                prov["calibrated_at"] = raw.get("calibrated_at")
                prov["file_globals"] = {
                    k: v for k, v in raw.items()
                    if isinstance(v, (int, float, str))}
            else:
                prov["warnings"].append(
                    f"{path} is not a JSON object; ignored")
        except (OSError, json.JSONDecodeError) as exc:
            prov["warnings"].append(f"could not read {path}: {exc}")
    else:
        prov["warnings"].append(
            f"calibration file {path} absent — every spread below is a "
            "compiled-in DEFAULT_SPREAD_PCT, NOT a measurement")
    per_und = raw.get("per_underlying") if isinstance(
        raw.get("per_underlying"), dict) else {}
    global_full = _first_fraction(raw, _SPREAD_KEYS_FULL)
    if global_full is None:
        half = _first_fraction(raw, _SPREAD_KEYS_HALF)
        if half is not None:
            global_full = (2.0 * half[0], f"2 x {half[1]}")
    out: dict[str, float] = {}
    for sym in symbols:
        entry = per_und.get(sym)
        if not isinstance(entry, dict):
            entry = per_und.get(f"US.{sym}") if isinstance(
                per_und.get(f"US.{sym}"), dict) else None
        hit = _first_fraction(entry, _SPREAD_KEYS_FULL) if entry else None
        if hit is not None:
            out[sym] = hit[0]
            prov["per_symbol"][sym] = {
                "spread_pct_of_mid": hit[0], "source": "per_underlying",
                "key": hit[1],
                "n_quotes": entry.get("n_quotes", entry.get("n_samples")),
                "zone": entry.get("zone", entry.get("delta_zone")),
                "calibrated_at": entry.get("calibrated_at"),
            }
            continue
        if global_full is not None:
            out[sym] = global_full[0]
            prov["per_symbol"][sym] = {
                "spread_pct_of_mid": global_full[0],
                "source": "file_global", "key": global_full[1],
                "n_quotes": raw.get("n_samples"), "zone": None,
                "calibrated_at": raw.get("calibrated_at"),
            }
            prov["warnings"].append(
                f"{sym}: no per-underlying calibration — using the file's "
                f"GLOBAL {global_full[1]}={global_full[0]:.4f}, which was "
                "calibrated on a single-name watchlist, not this index chain")
            continue
        out[sym] = DEFAULT_SPREAD_PCT.get(sym, 0.02)
        prov["per_symbol"][sym] = {
            "spread_pct_of_mid": out[sym], "source": "compiled_default",
            "key": "DEFAULT_SPREAD_PCT", "n_quotes": None,
            "zone": None, "calibrated_at": None}
        prov["warnings"].append(
            f"{sym}: NO calibration of any kind found — falling back to the "
            f"compiled-in DEFAULT_SPREAD_PCT {out[sym]:.4f}. Treat every "
            f"{sym} friction figure in this report as an assumption.")
    return out, prov


def friction_comparison(qualified: list[dict],
                        spread_pct: dict[str, float],
                        recompute_replay: bool = False) -> dict:
    """Replay friction vs the two production leg-mark models, in R.

    replay              spread charged on all FOUR realized leg marks
                        (entry short+long, exit short+long) — the most
                        informed of the three, and what this replay booked.
    production_deployed execution_costs.friction_r as deployed: 4 x
                        half_spread(net_credit), i.e. leg-mid-sum proxied by
                        2 x net_credit. Exact only at wing ratio r = 1/3.
    production_honest   the same 2-legs x 2-directions stack charged on the
                        ACTUAL entry leg marks (short + long).
    Also reports the measured wing ratio long_mark/short_mark, which is what
    decides whether the deployed proxy is conservative or optimistic.
    """
    if not qualified:
        return {"n_trades": 0}
    rows = []
    for r in qualified:
        pct = spread_pct[r["symbol"]]
        risk = r["risk_dollars"]
        cr, s, l = r["credit"], r["short_mark"], r["long_mark"]
        fees = 4.0 * FEE_PER_LEG_PER_DIRECTION
        deployed = 2.0 * pct * cr * 100.0 + fees
        honest = pct * (s + l) * 100.0 + fees
        if recompute_replay:
            sx = r.get("short_exit_mark")
            lx = r.get("long_exit_mark")
            replay = friction_dollars(pct, s, l,
                                      s if sx is None else sx,
                                      l if lx is None else lx)
        else:
            replay = r["friction"]
        rows.append((replay / risk, deployed / risk, honest / risk,
                     replay, deployed, honest, l / s, (s + l) / cr))
    a = np.array(rows)

    def st(col: int) -> dict:
        return {"mean": float(a[:, col].mean()),
                "median": float(np.median(a[:, col]))}

    return {
        "n_trades": len(rows),
        "replay_friction_r": st(0),
        "production_deployed_friction_r": st(1),
        "production_honest_leg_mark_friction_r": st(2),
        "replay_friction_dollars": st(3),
        "production_deployed_friction_dollars": st(4),
        "production_honest_leg_mark_friction_dollars": st(5),
        "ratio_replay_over_deployed": {
            "mean": float((a[:, 0] / a[:, 1]).mean()),
            "min": float((a[:, 0] / a[:, 1]).min()),
            "max": float((a[:, 0] / a[:, 1]).max())},
        "ratio_honest_over_deployed": {
            "mean": float((a[:, 2] / a[:, 1]).mean()),
            "median": float(np.median(a[:, 2] / a[:, 1]))},
        "wing_ratio_long_over_short": st(6),
        "leg_mid_sum_over_net_credit": st(7),
        "deployed_model_assumes_leg_mid_sum_over_credit": 2.0,
        "deployed_model_exact_at_wing_ratio": 1.0 / 3.0,
        "note": (
            "execution_costs.friction_r proxies each leg's mark by net_credit "
            "(leg-mid-sum = 2 x net_credit), exact only at wing ratio 1/3. "
            "Compare wing_ratio_long_over_short against 1/3: the deployed "
            "model's spread leg is off by exactly "
            "leg_mid_sum_over_net_credit / 2 (>1 = understated)."),
    }


def summarize(records: list[dict], resamples: int, seed: int,
              spread_pct: dict[str, float],
              width_policy: str = "five-first",
              credit_floor_frac: float = CREDIT_FLOOR_FRAC,
              liquidity_lag: str = PRIMARY_LIQUIDITY_LAG,
              mark: str = PRIMARY_MARK) -> dict:
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
    overall = _availability(records)
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
            "mark_convention": mark,
            "liquidity_lag": liquidity_lag,
            # ONE declared information set, applied to every gate (finding 2).
            "information_set": INFORMATION_SET,
            "information_set_statement": INFORMATION_SET_STATEMENT,
            "liquidity_screen_uses_future_information": False,
            "liquidity_screen_stricter_than_declared_information_set": (
                liquidity_lag == "prior"),
        },
        "approximation_notes": APPROXIMATION_NOTES,
        "verdict": {
            "criterion_source": PLAN_CRITERION_SOURCE,
            "criterion_text": PLAN_CRITERION_TEXT,
            "evaluated_as_specified": False,
            "why_not": CRITERION_NOT_EVALUATED_REASON,
            "regime_labels_present": False,
            "what_was_measured": (
                "UNCONDITIONAL availability of a qualifying vertical over the "
                "planned entry days, under one mark convention at a time."),
            "denominators": {
                "data_adequate": {
                    "definition": "planned entries whose FETCHED strikes could "
                                  "construct at least one vertical",
                    "n": overall["n_data_adequate"],
                    "n_qualified": overall["n_qualified"],
                    "rate": overall["availability_rate"],
                    "floor": AVAILABILITY_FLOOR,
                    "passes": overall["passes_floor_on_data_adequate"],
                },
                "planned": {
                    "definition": "every planned (entry_date, symbol) pair, "
                                  "data gaps charged against availability",
                    "n": overall["n_entries"],
                    "n_qualified": overall["n_qualified"],
                    "rate": overall["availability_rate_planned"],
                    "floor": AVAILABILITY_FLOOR,
                    "passes": overall["passes_floor_on_planned"],
                },
            },
            "overall_verdict": (
                "CRITERION NOT EVALUATED AS SPECIFIED (no regime labels). "
                "The unconditional rate FAILS the "
                f"{AVAILABILITY_FLOOR:.0%} floor on both denominators."
                if not (overall["passes_floor_on_data_adequate"]
                        or overall["passes_floor_on_planned"])
                else "CRITERION NOT EVALUATED AS SPECIFIED (no regime "
                     "labels); the unconditional rate clears the floor on at "
                     "least one denominator, which is NOT the criterion."),
        },
        "friction_comparison": friction_comparison(qualified, spread_pct),
        "availability": {
            "acceptance_criterion":
                f">={AVAILABILITY_FLOOR:.0%} of DATA-ADEQUATE entry days must "
                "have a qualifying vertical (revival plan M1-0.4). Entries "
                "whose option data was never fetched are excluded from the "
                "denominator and reported as data_coverage_rate instead. "
                "NOTE: the plan's own criterion is CONDITIONAL on the allowed "
                "regimes and is NOT what this number measures — see .verdict.",
            "overall": overall,
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
            "lb95_mean_r": block_bootstrap_lb95_mean(
                qualified, resamples, seed, "entry-week"),
            "lb95_mean_r_by_blocking": {
                "entry-week": block_bootstrap_lb95_mean(
                    qualified, resamples, seed, "entry-week"),
                "exposure-cluster": block_bootstrap_lb95_mean(
                    qualified, resamples, seed, "exposure-cluster"),
            },
            "n_blocks": {
                "entry-week": len(entry_week_blocks(qualified)),
                "exposure-cluster": len(exposure_cluster_blocks(qualified)),
            },
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


def replay_records(data_dir: Path, batch_plan: Path,
                   spread_pct: dict[str, float],
                   width_policy: str = "five-first",
                   credit_floor_frac: float = CREDIT_FLOOR_FRAC,
                   liquidity_lag: str = PRIMARY_LIQUIDITY_LAG,
                   mark: str = PRIMARY_MARK) -> tuple[list[dict], dict, dict]:
    """(records, file diagnostics, mark diagnostics) for ONE configuration.

    Chains are re-loaded per configuration because the "smile" convention
    REWRITES bar closes in place; sharing a cache across conventions would
    silently contaminate them.
    """
    prev_mark = set_mark_convention(mark)
    try:
        plan = load_batch_plan(batch_plan)
        grouped: dict[tuple[date, str], list[date]] = defaultdict(list)
        for row in plan:
            key = (row["entry_date"], row["symbol"])
            if row["expiry"] not in grouped[key]:
                grouped[key].append(row["expiry"])

        closes_cache: dict[str, dict[date, float]] = {}
        puts_cache: dict[tuple[str, date], dict | None] = {}
        file_diags: dict[str, dict] = {}
        mark_diag = {"convention": mark, "smile": {
            "days_fitted": 0, "days_unfittable": 0,
            "bars_repriced": 0, "bars_kept_raw": 0,
            "bars_kept_raw_on_unfittable_days": 0,
            "bars_kept_raw_on_fitted_days": 0}}
        records: list[dict] = []
        for (entry_date, symbol), expiries in sorted(grouped.items()):
            if symbol not in closes_cache:
                closes_cache[symbol] = load_underlying_closes(data_dir, symbol)
            for expiry in expiries:
                if (symbol, expiry) not in puts_cache:
                    puts, diag = load_put_contracts(data_dir, symbol, expiry)
                    if puts and mark == "smile":
                        stats = apply_smile_repricing(
                            puts, expiry, DIV_YIELD.get(symbol, 0.0),
                            closes_cache[symbol])
                        for k, v in stats.items():
                            mark_diag["smile"][k] += v
                    puts_cache[(symbol, expiry)] = puts
                    file_diags[f"{symbol}_{expiry.isoformat()}"] = diag
            if symbol not in spread_pct:
                raise ValueError(f"no spread_pct entry for {symbol}")
            records.append(replay_entry(
                entry_date, symbol, expiries, closes_cache[symbol],
                {e: puts_cache[(symbol, e)] for e in expiries},
                spread_pct[symbol], DIV_YIELD.get(symbol, 0.0), width_policy,
                credit_floor_frac, liquidity_lag))
        if mark == "smile":
            # Prove, per booked trade, that no leg pair mixed a curve-priced
            # leg with a raw one (finding 12).
            mark_diag["raw_mark_exposure"] = raw_mark_exposure(
                [r for r in records if r["qualified"]], puts_cache)
        return records, file_diags, mark_diag
    finally:
        set_mark_convention(prev_mark)


def _headline(records: list[dict], resamples: int, seed: int,
              spread_pct: dict[str, float]) -> dict:
    """The few numbers a sensitivity row needs, computed the same way as the
    primary summary so rows are directly comparable."""
    av = _availability(records)
    q = [r for r in records if r["qualified"]]
    rs = np.array([r["result_r"] for r in q])
    exits = Counter(r["exit_reason"] for r in q)
    fc = friction_comparison(q, spread_pct)
    return {
        "n_data_adequate": av["n_data_adequate"],
        "n_entries": av["n_entries"],
        "n_qualified": av["n_qualified"],
        "availability_rate_data_adequate": av["availability_rate"],
        "availability_rate_planned": av["availability_rate_planned"],
        "passes_floor_on_data_adequate": av["passes_floor_on_data_adequate"],
        "passes_floor_on_planned": av["passes_floor_on_planned"],
        "n_trades": int(rs.size),
        "win_rate": float(np.mean(rs > 0)) if rs.size else None,
        "mean_r": float(rs.mean()) if rs.size else None,
        "median_r": float(np.median(rs)) if rs.size else None,
        "lb95_mean_r_entry_week": block_bootstrap_lb95_mean(
            q, resamples, seed, "entry-week"),
        "lb95_mean_r_exposure_cluster": block_bootstrap_lb95_mean(
            q, resamples, seed, "exposure-cluster"),
        "max_drawdown_r": max_drawdown_r(q),
        "exit_profit_take": exits.get("profit_take", 0),
        "exit_dte_21": exits.get("dte_21", 0),
        "exit_data_end": exits.get("data_end", 0),
        "replay_friction_r_mean": (fc.get("replay_friction_r") or {}).get("mean"),
        "unqualified_reasons": dict(sorted(Counter(
            r["reason"] for r in records if not r["qualified"]).items())),
    }


def _sensitivity_table(data_dir: Path, batch_plan: Path, out_dir: Path,
                       spread_pct: dict[str, float], resamples: int, seed: int,
                       width_policy: str, credit_floor_frac: float,
                       liquidity_lag: str, mark: str) -> dict:
    """Headline numbers for every mark convention, both liquidity-screen
    information sets, and both width policies — one row per configuration,
    each row's entries.jsonl written next to the primary artifact.

    This is the artifact's central honesty device: the reader can see that
    the choice of daily mark moves the headline by more than the entire
    claimed edge, without having to rerun anything.
    """
    sens_dir = out_dir / "sensitivity"
    sens_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict] = {}

    def add(tag: str, *, m: str, lag: str, wp: str, note: str) -> None:
        recs, _, mdiag = replay_records(data_dir, batch_plan, spread_pct, wp,
                                       credit_floor_frac, lag, m)
        with (sens_dir / f"entries_{tag}.jsonl").open("w") as fh:
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
        rows[tag] = {"mark": m, "liquidity_lag": lag, "width_policy": wp,
                     "note": note, "entries_file": f"sensitivity/entries_{tag}.jsonl",
                     "smile_diagnostics": mdiag["smile"] if m == "smile" else None,
                     "raw_mark_exposure": mdiag.get("raw_mark_exposure"),
                     **_headline(recs, resamples, seed, spread_pct)}

    for m in MARK_CONVENTIONS:
        note = {
            "smile": "PRIMARY — cross-leg consistent: both legs off ONE "
                     "same-day IV curve",
            "close": "DISCLOSED UPPER BOUND — last trade per leg; the credit "
                     "inherits the full leg-vs-leg timing gap",
            "vw": "volume-weighted average price per leg",
            "hl2": "(high + low) / 2 per leg",
        }[m]
        add(f"mark-{m}", m=m, lag=liquidity_lag, wp=width_policy, note=note)
    for lag in LIQUIDITY_LAGS:
        if lag == liquidity_lag:
            continue
        add(f"liquidity-{lag}", m=mark, lag=lag, wp=width_policy,
            note=("LEGAL UNDER THE DECLARED INFORMATION SET (enter-at-the-"
                  "close): screens the entry day's own full-day trade count, "
                  "which belongs to the same completed bar the selection "
                  "already marks off. Shown so the cost of the stricter "
                  "'prior' primary is visible."
                  if lag == "same-day" else
                  "screens the last session before entry — stricter than the "
                  "declared information set requires, and the only variant "
                  "also valid under a decide-before-the-close reading"))
    for wp in ("five-first", "any"):
        if wp == width_policy:
            continue
        add(f"width-{wp}", m=mark, lag=liquidity_lag, wp=wp,
            note=("'any' takes the narrowest width in WIDTHS whose credit "
                  "clears the floor — the literal reading of the spec's "
                  "'width $5-10'"))
    return {
        "primary": {"mark": mark, "liquidity_lag": liquidity_lag,
                    "width_policy": width_policy,
                    "credit_floor_frac": credit_floor_frac},
        "how_to_read": (
            "Each row is a full independent replay of the same data under one "
            "defensible convention change. Compare mean_r across the mark-* "
            "rows: the spread of that column IS the identification limit of "
            "daily trade-print aggregates."),
        "rows": rows,
    }


def run(data_dir: Path, batch_plan: Path, out_dir: Path,
        spread_pct: dict[str, float], resamples: int = BOOTSTRAP_RESAMPLES,
        seed: int = BOOTSTRAP_SEED, width_policy: str = "five-first",
        credit_floor_frac: float = CREDIT_FLOOR_FRAC,
        liquidity_lag: str = PRIMARY_LIQUIDITY_LAG,
        mark: str = PRIMARY_MARK,
        sensitivity: bool = False,
        calibration: dict | None = None) -> dict:
    records, file_diags, mark_diag = replay_records(
        data_dir, batch_plan, spread_pct, width_policy, credit_floor_frac,
        liquidity_lag, mark)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "entries.jsonl").open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    summary = summarize(records, resamples, seed, spread_pct, width_policy,
                        credit_floor_frac, liquidity_lag, mark)
    summary["config"]["spread_pct_provenance"] = calibration or {
        "path": None, "file_present": False, "per_symbol": {},
        "warnings": ["no calibration provenance was passed to run(); the "
                     "spread_pct values above are caller-supplied"]}
    summary["mark_diagnostics"] = mark_diag
    # What the DEPLOYED cost model would actually charge, if it were asked
    # today, from whatever is in data/execution_costs.json (finding 14).
    file_pct = dict((calibration or {}).get("resolved_from_file") or {})
    qualified = [r for r in records if r["qualified"]]
    if file_pct and all(r["symbol"] in file_pct for r in qualified):
        summary["friction_under_calibration_file"] = {
            "spread_pct": file_pct,
            "note": ("recomputed from the SAME booked marks using only what "
                     "the calibration file actually contains — this is what "
                     "production would charge today"),
            **friction_comparison(qualified, file_pct, recompute_replay=True),
        }
    if sensitivity:
        summary["sensitivity"] = _sensitivity_table(
            data_dir, batch_plan, out_dir, spread_pct, resamples, seed,
            width_policy, credit_floor_frac, liquidity_lag, mark)
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
    ap.add_argument("--spread-pct", default=None,
                    help="comma list SYM=frac_of_mid (measured full spread); "
                         "omit to read data/execution_costs.json instead")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="path to execution_costs.json (default: "
                         "<repo>/data/execution_costs.json)")
    ap.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--width-policy", choices=("five-first", "any"),
                    default="five-first",
                    help="how to resolve the spec's $5-10 width freedom "
                         "(default five-first = engine's original rule)")
    ap.add_argument("--credit-floor-frac", type=float, default=CREDIT_FLOOR_FRAC,
                    help="min credit as a fraction of width (spec: 0.25); "
                         "lower values are RETUNE SCENARIOS, not the spec")
    ap.add_argument("--mark", choices=MARK_CONVENTIONS, default=PRIMARY_MARK,
                    help="daily mark convention (default smile = the only "
                         "cross-leg-consistent one; close is an upper bound)")
    ap.add_argument("--liquidity-lag", choices=LIQUIDITY_LAGS,
                    default=PRIMARY_LIQUIDITY_LAG,
                    help="which session the n>=30 screen reads. Both are "
                         "legal under the declared enter-at-the-close "
                         "information set; 'prior' (default) is the stricter "
                         "variant that also survives a decide-before-the-"
                         "close reading")
    ap.add_argument("--symbols", default="SPY,QQQ",
                    help="symbols to resolve calibrated spreads for")
    ap.add_argument("--no-sensitivity", action="store_true",
                    help="skip the per-convention sensitivity re-runs")
    args = ap.parse_args(argv)

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    calib_path = args.calibration or (
        Path(__file__).resolve().parent.parent / "data" / "execution_costs.json")
    spread_pct, calib = load_spread_calibration(calib_path, symbols)
    calib["resolved_from_file"] = dict(spread_pct)
    if args.spread_pct:
        # An explicit override still records what the FILE said, and shouts
        # when the two disagree — a hardcoded number that silently disagrees
        # with the deployed calibration is exactly finding 14.
        override = parse_spread_pct(args.spread_pct)
        for sym, val in override.items():
            was = spread_pct.get(sym)
            spread_pct[sym] = val
            entry = calib["per_symbol"].setdefault(sym, {})
            entry.update({"spread_pct_of_mid": val, "source": "cli_override",
                          "overrides_file_value": was})
            if was is not None and abs(was - val) > 1e-12:
                calib["warnings"].append(
                    f"{sym}: --spread-pct override {val:.4f} DISAGREES with "
                    f"the calibration file's {was:.4f}; the file is what "
                    f"production charges, the override is what this replay "
                    f"charged")
    for w in calib["warnings"]:
        print(f"WARNING (friction provenance): {w}")

    summary = run(args.data_dir, args.batch_plan, args.out_dir,
                  spread_pct, args.resamples, args.seed,
                  args.width_policy, args.credit_floor_frac,
                  args.liquidity_lag, args.mark,
                  sensitivity=not args.no_sensitivity, calibration=calib)
    av = summary["availability"]["overall"]
    mp = summary["managed_payoff"]
    inv = summary["data_inventory"]

    def _r(x):
        return None if x is None else round(x, 4)

    print(f"VERDICT: {summary['verdict']['overall_verdict']}")
    print(f"mark={args.mark} liquidity_lag={args.liquidity_lag} "
          f"width_policy={args.width_policy} floor={args.credit_floor_frac}")
    print(f"entries={av['n_entries']} marks_present={av['n_marks_present']} "
          f"data_adequate={av['n_data_adequate']} qualified={av['n_qualified']}")
    print(f"data_coverage_rate={_r(av['data_coverage_rate'])} "
          f"availability(data-adequate)={_r(av['availability_rate'])} "
          f"availability(planned)={_r(av['availability_rate_planned'])} "
          f"(strict={_r(av['availability_rate_strict'])})")
    print(f"trades={mp['n_trades']} WR={_r(mp['win_rate'])} "
          f"mean_r={_r(mp['mean_r'])} median_r={_r(mp['median_r'])} "
          f"lb95_week={_r(mp['lb95_mean_r_by_blocking']['entry-week'])} "
          f"lb95_exposure={_r(mp['lb95_mean_r_by_blocking']['exposure-cluster'])} "
          f"maxDD_r={_r(mp['max_drawdown_r'])}")
    print(f"unqualified reasons: {summary['availability']['unqualified_reasons']}")
    if "sensitivity" in summary:
        print("sensitivity (tag: avail_adequate/avail_planned n meanR lb95wk):")
        for tag, row in summary["sensitivity"]["rows"].items():
            print(f"  {tag:22} {_r(row['availability_rate_data_adequate'])}"
                  f"/{_r(row['availability_rate_planned'])} "
                  f"n={row['n_trades']:3d} meanR={_r(row['mean_r'])} "
                  f"lb95={_r(row['lb95_mean_r_entry_week'])}")
    fc = summary["friction_comparison"]
    if fc.get("n_trades"):
        print(f"friction R: replay {_r(fc['replay_friction_r']['mean'])} vs "
              f"deployed-model {_r(fc['production_deployed_friction_r']['mean'])} "
              f"vs honest-leg-mark "
              f"{_r(fc['production_honest_leg_mark_friction_r']['mean'])} "
              f"(wing ratio median "
              f"{_r(fc['wing_ratio_long_over_short']['median'])})")
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
