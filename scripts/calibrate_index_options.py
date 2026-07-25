"""M1-0.3 — per-underlying, per-ZONE spread calibration for the index sleeve.

Measures what the sleeve actually trades and writes it under the
``per_underlying`` key of ``data/execution_costs.json`` (the GLOBAL
watchlist-calibrated figures in that file are left untouched).
``execution_costs.half_spread_cost(..., underlying=, zone=)`` and
``execution_costs.friction_r(...)`` prefer these per-name figures.

WHAT CHANGED 2026-07-25 (zone bug + snapshot mode)
--------------------------------------------------
The 2026-07-22 run sampled ``|delta| 0.03-0.40`` — a union of "short-leg
band" and "everything cheaper" — and reported IWM at 3.12% of mid. That zone
is dominated by far-OTM strikes whose PERCENT spread is tighter (a $0.05
spread on a $2.50 mid is 2%; the same nickel on a $0.40 mid is 12.5%), so the
pooled median described strikes the sleeve never sells. The zone the sleeve
does trade measures materially wider, and the two legs differ from each other:

    MEASURED (option_chain_snapshots, IWM PUTS, 30-37 DTE inside the 30-45
    band, short |delta| 0.20-0.35, bid>0 & ask>bid, mid=(bid+ask)/2):
      $5-wide  n=32 pairs: short-leg spread 4.24% of mid, wing 5.04%,
                           wing ratio r=long/short 0.731, credit/width 0.191
      $10-wide n=25 pairs: short-leg spread 3.74%, wing 5.74%,
                           wing ratio 0.530, credit/width 0.176

So this runner now (a) classifies the SHORT zone by the sleeve's own delta
band and the WING zone as the actual $5/$10 wing strikes of those shorts,
(b) reports the two zones SEPARATELY plus the wing ratio and credit/width
distribution per underlying AND per width, and (c) recomputes the decision
line with the honest ``execution_costs.friction_r`` (which no longer proxies
per-leg marks by the net credit).

``--source snapshots`` reads ``option_chain_snapshots`` from Postgres instead
of live OpenD quotes: repeatable, auditable, no market-hours dependency, and
the same feed that accumulates daily. It is the default when the US market is
closed. ``--source live`` samples OpenD in-session (requires MoomooOpenD, i.e.
EC2 next to trading-agent-opend.service).

REPORT ONLY — the spec/whitelist change is the operator's call.

Usage:
    .venv/bin/python scripts/calibrate_index_options.py \
        --underlyings SPY,QQQ,IWM [--source auto|snapshots|live] \
        [--since-days 120] [--min-quotes 40] [--min-pairs 10] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

# Sleeve-1 zone (REVIVAL_PLAN M1-1): puts only, 30-45 DTE, short leg
# |delta| 0.20-0.35 — the spec's own entry gates, not a wider proxy.
DTE_MIN, DTE_MAX = 30, 45
SHORT_DELTA_MIN, SHORT_DELTA_MAX = 0.20, 0.35
# The wing zone is DERIVED, never delta-filtered: it is exactly the strikes a
# $5 / $10 wide vertical buys against an in-zone short leg.
WIDTHS: tuple[float, ...] = (5.0, 10.0)
# Live mode strike prefilter. Must reach at least the widest wing below the
# LOWEST in-zone short strike: at 30-45 DTE a 0.20-delta index put sits ~5-8%
# OTM and its $10 wing another ~1.5% — and on a low-priced underlying $10 is a
# much larger fraction of spot — so the floor is 0.80x spot.
# The cap must not cut that band off: a $1-strike chain (SPY/QQQ) needs ~120
# strikes to span it, and cutting at 40 would drop exactly the wing strikes the
# pairing needs (a failure the old runner hid behind its delta filter, which
# discarded the wings anyway). get_market_snapshot accepts hundreds of codes
# per call (the EOD chain job batches 300), so one batch per expiry stays well
# inside the futu request budget while covering the whole band.
STRIKE_MIN_FRAC_OF_SPOT = 0.80
QUOTE_BATCH = 120         # codes per get_quote call
MAX_CODES_PER_EXPIRY = 120
# futu OpenAPI limits get_option_chain to ~10 requests per 30 s. Cap expiries
# per name and space the chain calls so a default SPY,QQQ,IWM run never
# breaches the limit mid-run and starves the last name (IWM — the decision
# line) below --min-quotes. Tests set CHAIN_SLEEP_S to 0.
MAX_EXPIRIES_PER_NAME = 4
CHAIN_SLEEP_S = 3.5
STRIKE_EPS = 0.01         # strike float matching tolerance

# IWM decision-line parameters (M1-1 spec): $5-wide vertical, evaluated BOTH
# at the gate's floor credit (width/4) and at the credit the chain actually
# pays (measured credit/width median). Whitelist bar friction_r <= 0.06R.
DECISION_WIDTH = 5.0
DECISION_CREDIT = DECISION_WIDTH / 4.0
DECISION_BAR_R = 0.06

_WIDTHS_DESC = "/".join(f"${w:g}" for w in WIDTHS)
ZONE_DESC = (f"PUTS {DTE_MIN}-{DTE_MAX} DTE; SHORT zone |delta| "
             f"{SHORT_DELTA_MIN}-{SHORT_DELTA_MAX}; WING zone = the "
             f"{_WIDTHS_DESC}-wide wing strikes of those shorts")


# ---------------------------------------------------------------------------
# Quote sources
# ---------------------------------------------------------------------------


def _load_moomoo():
    """Import the in-repo moomoo quote functions. Separated for test mocking."""
    from trading_agent.mcp_servers.moomoo.server import (
        get_option_chain,
        get_quote,
        list_option_expiries,
    )
    return get_quote, list_option_expiries, get_option_chain


def _quote(underlying: str, day: str, expiry: str, dte, strike, bid, ask,
           delta) -> dict | None:
    """Normalize one raw row into a usable quote, or None if unusable."""
    try:
        strike = float(strike)
        bid = float(bid)
        ask = float(ask)
        dte = int(dte)
    except (TypeError, ValueError):
        return None
    if strike <= 0 or bid <= 0 or ask <= bid:
        return None
    mid = (bid + ask) / 2.0
    try:
        d = abs(float(delta)) if delta is not None else None
    except (TypeError, ValueError):
        d = None
    return {
        "underlying": underlying, "day": day, "expiry": expiry, "dte": dte,
        "strike": strike, "bid": bid, "ask": ask, "mid": mid,
        "abs_delta": d, "spread_pct": (ask - bid) / mid,
    }


def collect_live(underlyings: list[str], quote_fns) -> dict[str, list[dict]]:
    """Per-underlying quotes across the whole zone band, sampled from OpenD.

    Samples a WIDE strike band (0.80x spot .. spot) on purpose: zone
    classification happens later, in :func:`analyse`, so the wing strikes of
    an in-zone short are present even though their own delta sits far below
    the short band. Delta-filtering here is exactly the 2026-07-22 zone bug.
    """
    get_quote, list_option_expiries, get_option_chain = quote_fns
    out: dict[str, list[dict]] = {}
    today = date.today()
    day = today.isoformat()
    first_chain_call = True
    for t in underlyings:
        code = f"US.{t}"
        samples: list[dict] = []
        try:
            spot_rows = get_quote([code]).get("rows") or []
            spot = float(spot_rows[0].get("last_price") or 0) if spot_rows else 0.0
            expiries: list[tuple[str, int]] = []
            for row in (list_option_expiries(code).get("rows") or []):
                st = str(row.get("strike_time") or "")[:10]
                try:
                    dte = (date.fromisoformat(st) - today).days
                except ValueError:
                    continue
                if DTE_MIN <= dte <= DTE_MAX:
                    expiries.append((st, dte))
        except Exception as e:
            print(f"  {t}: spot/expiry fetch failed ({e}) — skipped", file=sys.stderr)
            out[t] = samples
            continue
        if spot <= 0:
            print(f"  {t}: no spot price — skipped", file=sys.stderr)
            out[t] = samples
            continue
        for exp, dte in expiries[:MAX_EXPIRIES_PER_NAME]:
            if not first_chain_call and CHAIN_SLEEP_S:
                time.sleep(CHAIN_SLEEP_S)  # futu chain-request rate limit
            first_chain_call = False
            try:
                chain = get_option_chain(code, exp, option_type="PUT").get("rows") or []
            except Exception as e:
                print(f"  {t} {exp}: chain failed ({e})", file=sys.stderr)
                continue
            strikes: list[tuple[float, str]] = []
            for c in chain:
                try:
                    k = float(c.get("strike_price") or 0)
                except (TypeError, ValueError):
                    continue
                if c.get("code") and STRIKE_MIN_FRAC_OF_SPOT * spot <= k <= spot:
                    strikes.append((k, c["code"]))
            strikes.sort(reverse=True)  # nearest-the-money first
            strike_of_code = {c: k for k, c in strikes[:MAX_CODES_PER_EXPIRY]}
            codes = list(strike_of_code)
            for i in range(0, len(codes), QUOTE_BATCH):
                try:
                    quotes = get_quote(codes[i:i + QUOTE_BATCH]).get("rows") or []
                except Exception as e:
                    print(f"  {t} {exp}: option quotes failed ({e})", file=sys.stderr)
                    continue
                for r in quotes:
                    # Real moomoo get_market_snapshot rows carry the greek as
                    # ``option_delta``; ``delta`` is the legacy fallback. The
                    # option quote row may not echo the strike, so fall back
                    # to the chain's own strike for that code.
                    strike = r.get("strike_price")
                    if strike in (None, ""):
                        strike = strike_of_code.get(str(r.get("code")))
                    bid = r.get("bid_price")
                    ask = r.get("ask_price")
                    dlt = r.get("option_delta")
                    q = _quote(
                        t, day, exp, dte, strike,
                        r.get("bid") if bid is None else bid,
                        r.get("ask") if ask is None else ask,
                        r.get("delta") if dlt is None else dlt,
                    )
                    if q:
                        samples.append(q)
        out[t] = samples
    return out


_SNAPSHOT_SQL = """
    SELECT underlying,
           snapshot_date::text AS day,
           expiry::text        AS expiry,
           (expiry - snapshot_date) AS dte,
           strike, bid, ask, delta
    FROM option_chain_snapshots
    WHERE underlying = ANY(%(names)s)
      AND option_type = 'PUT'
      AND expiry IS NOT NULL
      AND bid > 0 AND ask > bid
      AND (expiry - snapshot_date) BETWEEN %(dte_min)s AND %(dte_max)s
      AND snapshot_date >= %(since)s
    ORDER BY underlying, snapshot_date, expiry, strike
"""


def _fetch_snapshot_rows(names: list[str], since: date) -> list[tuple]:
    """Read-only SELECT against option_chain_snapshots. Test seam."""
    from trading_agent.store.postgres import cursor
    with cursor() as cur:
        cur.execute(_SNAPSHOT_SQL, {"names": names, "dte_min": DTE_MIN,
                                    "dte_max": DTE_MAX, "since": since})
        return list(cur.fetchall())


def collect_snapshots(
    underlyings: list[str], since: date, fetch=None
) -> dict[str, list[dict]]:
    """Per-underlying zone-band quotes read from the stored EOD snapshots.

    Same shape as :func:`collect_live` — the analysis downstream is identical,
    which is the point: the stored feed is the auditable, market-hours-free
    source for the very same measurement.
    """
    out: dict[str, list[dict]] = {t: [] for t in underlyings}
    rows = (fetch or _fetch_snapshot_rows)(underlyings, since)
    for row in rows:
        underlying, day, expiry, dte, strike, bid, ask, delta = tuple(row)[:8]
        t = str(underlying).upper()
        if t not in out:
            continue
        q = _quote(t, str(day), str(expiry), dte, strike, bid, ask, delta)
        if q:
            out[t].append(q)
    return out


# ---------------------------------------------------------------------------
# Zone classification + statistics
# ---------------------------------------------------------------------------


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def _p75(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[int(0.75 * (len(s) - 1))]


def build_pairs(quotes: list[dict], width: float) -> list[dict]:
    """Vertical pairs at ``width``: an in-zone short leg + its wing strike.

    A pair needs both legs quoted in the SAME chain (same snapshot day and
    expiry) and ``short_mid > long_mid`` — otherwise it is not a credit
    vertical but a crossed/stale quote.
    """
    chains: dict[tuple[str, str], dict[float, dict]] = {}
    for q in quotes:
        chains.setdefault((q["day"], q["expiry"]), {})[round(q["strike"], 2)] = q
    pairs: list[dict] = []
    for legs in chains.values():
        for short in legs.values():
            d = short["abs_delta"]
            if d is None or not (SHORT_DELTA_MIN <= d <= SHORT_DELTA_MAX):
                continue
            want = short["strike"] - width
            long_leg = legs.get(round(want, 2))
            if long_leg is None:      # tolerate float noise in live strikes
                for k, cand in legs.items():
                    if abs(k - want) <= STRIKE_EPS:
                        long_leg = cand
                        break
            if long_leg is None or long_leg["mid"] >= short["mid"]:
                continue
            credit = short["mid"] - long_leg["mid"]
            pairs.append({
                "day": short["day"], "expiry": short["expiry"],
                "dte": short["dte"], "width": width,
                "short": short, "long": long_leg, "credit": credit,
                "wing_ratio": long_leg["mid"] / short["mid"],
                "credit_frac": credit / width,
            })
    return pairs


def analyse(quotes: list[dict]) -> dict:
    """Zone-separated statistics for one underlying.

    Medians are None where a zone is empty; the caller decides whether that
    is enough to calibrate on. The SHORT and WING zones are reported
    SEPARATELY because they measure materially differently (4.24% vs 5.04%
    on IWM $5-wides) and a pooled median hides exactly the wing cost that
    kills a narrow index vertical.
    """
    short_zone = [q for q in quotes
                  if q["abs_delta"] is not None
                  and SHORT_DELTA_MIN <= q["abs_delta"] <= SHORT_DELTA_MAX]
    no_delta = sum(1 for q in quotes if q["abs_delta"] is None)
    by_width: dict[float, dict] = {}
    wing_keys: set[tuple] = set()
    wing_quotes: list[dict] = []
    total_pairs = 0
    for w in WIDTHS:
        pairs = build_pairs(quotes, w)
        total_pairs += len(pairs)
        w_wings = []
        for p in pairs:
            lo = p["long"]
            w_wings.append(lo)
            key = (lo["day"], lo["expiry"], round(lo["strike"], 2))
            if key not in wing_keys:
                wing_keys.add(key)
                wing_quotes.append(lo)
        by_width[w] = {
            "n_pairs": len(pairs),
            "n_chains": len({(p["day"], p["expiry"]) for p in pairs}),
            "wing_ratio": _median([p["wing_ratio"] for p in pairs]),
            "credit_frac_median": _median([p["credit_frac"] for p in pairs]),
            "legsum_over_credit": _median(
                [(p["short"]["mid"] + p["long"]["mid"]) / p["credit"]
                 for p in pairs]),
            "short_zone_spread_pct": _median(
                [p["short"]["spread_pct"] for p in pairs]),
            "wing_zone_spread_pct": _median([q["spread_pct"] for q in w_wings]),
            "dte_observed": (min((p["dte"] for p in pairs), default=None),
                             max((p["dte"] for p in pairs), default=None)),
        }
    pooled = [q["spread_pct"] for q in short_zone + wing_quotes]
    days = sorted({q["day"] for q in quotes})
    return {
        "n_raw_quotes": len(quotes),
        "n_no_delta": no_delta,
        "n_short_zone": len(short_zone),
        "n_wing_zone": len(wing_quotes),
        "n_quotes": len(pooled),
        "n_pairs": total_pairs,
        "short_zone_spread_pct": _median([q["spread_pct"] for q in short_zone]),
        "wing_zone_spread_pct": _median([q["spread_pct"] for q in wing_quotes]),
        "median_spread_pct_mid": _median(pooled),
        "spread_pct_of_mid_p75": _p75(pooled),
        "days": days,
        "by_width": by_width,
    }


def _pick_wing_ratio(stats: dict) -> float | None:
    """One scalar wing ratio for the persisted entry.

    The MAX of the per-width medians: ``legsum = cr(1+r)/(1-r)`` is
    increasing in r, so the largest measured ratio is the conservative
    (friction-overstating) choice for a consumer that knows only
    (width, credit). Per-width figures are preserved under ``by_width``.
    """
    rs = [v["wing_ratio"] for v in stats["by_width"].values()
          if v["wing_ratio"] is not None]
    return max(rs) if rs else None


def _pick_credit_frac(stats: dict) -> float | None:
    """Credit/width median at the DECISION width when measured, else the
    smallest measured width median (the pessimistic end)."""
    bw = stats["by_width"]
    at_decision = bw.get(DECISION_WIDTH, {}).get("credit_frac_median")
    if at_decision is not None:
        return at_decision
    cs = [v["credit_frac_median"] for v in bw.values()
          if v["credit_frac_median"] is not None]
    return min(cs) if cs else None


# ---------------------------------------------------------------------------
# Decision line
# ---------------------------------------------------------------------------


def _decision_friction_r(
    *,
    short_zone_spread_pct: float,
    wing_zone_spread_pct: float,
    wing_ratio: float,
    net_credit: float = DECISION_CREDIT,
    width: float = DECISION_WIDTH,
) -> dict:
    """Honest ``friction_r`` for one vertical AT THE MEASURED figures.

    Mirrors ``execution_costs.friction_r`` — fees on 4 fills plus half of
    EACH leg's own spread at EACH leg's own mark, both directions — but takes
    the freshly measured numbers directly, so the printed decision is
    identical with or without --dry-run (the calibration file may not exist
    yet). Cross-checked against ``execution_costs.friction_r`` in
    tests/test_calibrate_index_options.py.

    Returns the whole arithmetic, not just the ratio, so the printed verdict
    can show the numbers behind it.
    """
    from trading_agent.execution_costs import (
        fees_per_side,
        leg_marks_from_wing_ratio,
    )
    short_mark, long_mark = leg_marks_from_wing_ratio(net_credit, wing_ratio)
    fees = 4.0 * fees_per_side(1, "OPT")
    spread = 2.0 * 100.0 * (short_zone_spread_pct / 2.0 * short_mark
                            + wing_zone_spread_pct / 2.0 * long_mark)
    risk = (width - net_credit) * 100.0
    return {
        "friction_r": (fees + spread) / risk,
        "short_mark": short_mark, "long_mark": long_mark,
        "legsum": short_mark + long_mark, "fees": fees, "spread": spread,
        "risk": risk, "net_credit": net_credit, "width": width,
    }


def _zone_inputs(entry: dict) -> tuple[float, float, float] | None:
    r = entry.get("wing_ratio")
    sp = entry.get("short_zone_spread_pct")
    wp = entry.get("wing_zone_spread_pct")
    if r is None or sp is None or wp is None:
        return None
    return float(sp), float(wp), float(r)


def _decision_scenarios(entry: dict) -> list[dict]:
    """Every (width, credit) the verdict is evaluated at, with ITS OWN
    measured zone spreads and wing ratio.

    Width matters twice over: the wing ratio and the credit fraction both
    move with it (IWM measures r 0.731 / credit 0.191x at $5 and r 0.530 /
    0.176x at $10), and R = width − credit scales with it too, so a $10-wide
    can clear the bar while a $5-wide fails. Each width is evaluated at the
    gate-floor credit (width/4) AND at the credit the chain actually pays.
    Falls back to the scalar entry when no per-width block exists.
    """
    out: list[dict] = []
    by_width = entry.get("by_width") or {}
    seen = False
    for key, v in sorted(by_width.items(), key=lambda kv: float(kv[0])):
        if not v.get("n_pairs"):
            continue
        zi = _zone_inputs(v)
        if zi is None:
            continue
        seen = True
        width = float(key)
        sp, wp, r = zi
        credits = [("gate-floor credit width/4", width / 4.0)]
        cf = v.get("credit_frac_median")
        if cf:
            credits.append((f"MEASURED credit {cf:.3f}xwidth", cf * width))
        for label, credit in credits:
            out.append({"width": width, "label": label, "net_credit": credit,
                        "short_zone_spread_pct": sp, "wing_zone_spread_pct": wp,
                        "wing_ratio": r})
    if seen:
        return out
    zi = _zone_inputs(entry)
    if zi is None:
        return []
    sp, wp, r = zi
    credits = [("gate-floor credit width/4", DECISION_CREDIT)]
    cf = entry.get("credit_frac_median")
    if cf:
        credits.append((f"MEASURED credit {cf:.3f}xwidth", cf * DECISION_WIDTH))
    return [{"width": DECISION_WIDTH, "label": label, "net_credit": credit,
             "short_zone_spread_pct": sp, "wing_zone_spread_pct": wp,
             "wing_ratio": r} for label, credit in credits]


def _print_decision_block(t: str, entry: dict) -> None:
    """Per-underlying, per-width friction verdict with the arithmetic."""
    scenarios = _decision_scenarios(entry)
    if not scenarios:
        print(f"  {t}: NOT ASSESSED — zone measurement incomplete "
              f"(wing_ratio={entry.get('wing_ratio')}, "
              f"short={entry.get('short_zone_spread_pct')}, "
              f"wing={entry.get('wing_zone_spread_pct')})")
        return
    for s in scenarios:
        d = _decision_friction_r(
            short_zone_spread_pct=s["short_zone_spread_pct"],
            wing_zone_spread_pct=s["wing_zone_spread_pct"],
            wing_ratio=s["wing_ratio"], net_credit=s["net_credit"],
            width=s["width"])
        verdict = "EXCLUDED" if d["friction_r"] > DECISION_BAR_R else "OK"
        print(f"  {t} ${s['width']:g}-wide @ {s['label']}: friction_r "
              f"{d['friction_r']:.3f}R ({verdict} vs {DECISION_BAR_R}R bar)")
        print(f"      credit {s['net_credit']:.3f} -> legs "
              f"{d['short_mark']:.3f}/{d['long_mark']:.3f} "
              f"(r={s['wing_ratio']:.3f}, legsum/credit "
              f"{d['legsum'] / s['net_credit']:.2f}), spread "
              f"{s['short_zone_spread_pct']:.2%}/"
              f"{s['wing_zone_spread_pct']:.2%} of mid "
              f"-> ${d['spread']:.2f} + fees ${d['fees']:.2f} "
              f"on R ${d['risk']:.2f}")


def _worst_friction_r(entry: dict, width: float | None = DECISION_WIDTH
                      ) -> float | None:
    """Worst (largest) modeled friction across the credit scenarios.

    ``width=None`` pools every measured width; the default restricts to the
    plan's decision width, because that is the structure the 0.06R bar was
    written about.
    """
    frs = [
        _decision_friction_r(
            short_zone_spread_pct=s["short_zone_spread_pct"],
            wing_zone_spread_pct=s["wing_zone_spread_pct"],
            wing_ratio=s["wing_ratio"], net_credit=s["net_credit"],
            width=s["width"])["friction_r"]
        for s in _decision_scenarios(entry or {})
        if width is None or abs(s["width"] - width) < 1e-9
    ]
    return max(frs) if frs else None


def _best_friction_r(entry: dict, width: float | None = None) -> float | None:
    """Best (smallest) modeled friction across widths/credits — used only to
    tell the operator when SOME measured structure clears the bar even
    though the decision-width one does not."""
    frs = [
        _decision_friction_r(
            short_zone_spread_pct=s["short_zone_spread_pct"],
            wing_zone_spread_pct=s["wing_zone_spread_pct"],
            wing_ratio=s["wing_ratio"], net_credit=s["net_credit"],
            width=s["width"])["friction_r"]
        for s in _decision_scenarios(entry or {})
        if width is None or abs(s["width"] - width) < 1e-9
    ]
    return min(frs) if frs else None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _write_atomic(path: Path, payload: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".execution_costs.",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _emit_calibration_event(per_underlying: dict) -> None:
    """Best-effort agent_event — only when a Postgres env is present, so a
    laptop dry-run never spawns the ~/agent_events.fallback.jsonl file."""
    has_pg = bool(os.environ.get("POSTGRES_DSN")) or (
        Path.home() / ".env.postgres").exists()
    if not has_pg:
        return
    try:
        from trading_agent.events import emit, new_run_id
        emit(run_id=new_run_id("calib"), trigger="manual",
             agent="calibrate_index_options", event_type="calibration_updated",
             payload={"per_underlying": per_underlying, "zone": ZONE_DESC})
    except Exception as e:
        print(f"agent_event emit failed (non-fatal): {e}", file=sys.stderr)


def _round(v, n=5):
    return None if v is None else round(float(v), n)


def build_entry(stats: dict, *, source: str, sample_window: str,
                now: str) -> dict:
    """The persisted ``per_underlying`` record for one calibrated name."""
    return {
        # Pooled short+wing median — what a zone-unaware consumer gets.
        "median_spread_pct_mid": _round(stats["median_spread_pct_mid"]),
        "spread_pct_of_mid_p75": _round(stats["spread_pct_of_mid_p75"]),
        "short_zone_spread_pct": _round(stats["short_zone_spread_pct"]),
        "wing_zone_spread_pct": _round(stats["wing_zone_spread_pct"]),
        "wing_ratio": _round(_pick_wing_ratio(stats), 4),
        "credit_frac_median": _round(_pick_credit_frac(stats), 4),
        "n_quotes": stats["n_quotes"],
        "n_pairs": stats["n_pairs"],
        "n_short_zone": stats["n_short_zone"],
        "n_wing_zone": stats["n_wing_zone"],
        "source": source,
        "sample_window": sample_window,
        "zone": ZONE_DESC,
        "calibrated_at": now,
        "by_width": {
            f"{w:g}": {
                "n_pairs": v["n_pairs"],
                "n_chains": v["n_chains"],
                "wing_ratio": _round(v["wing_ratio"], 4),
                "credit_frac_median": _round(v["credit_frac_median"], 4),
                "legsum_over_credit": _round(v["legsum_over_credit"], 3),
                "short_zone_spread_pct": _round(v["short_zone_spread_pct"]),
                "wing_zone_spread_pct": _round(v["wing_zone_spread_pct"]),
                "dte_observed": list(v["dte_observed"]),
            }
            for w, v in stats["by_width"].items()
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_source(arg: str) -> str:
    """``auto`` → live in-session, stored snapshots when the market is shut."""
    if arg != "auto":
        return arg
    from trading_agent.market_calendar import is_us_market_open
    return "live" if is_us_market_open() else "snapshots"


def _print_underlying_stats(t: str, stats: dict) -> None:
    days = stats["days"]
    span = f" [{days[0]}..{days[-1]}]" if days else ""
    nd = f" ({stats['n_no_delta']} without delta)" if stats["n_no_delta"] else ""
    print(f"  {t}: raw {stats['n_raw_quotes']} quotes{nd} over "
          f"{len(days)} day(s){span} -> short-zone {stats['n_short_zone']}, "
          f"wing-zone {stats['n_wing_zone']}, pairs {stats['n_pairs']}")
    for w, v in stats["by_width"].items():
        if not v["n_pairs"]:
            print(f"      ${w:g}-wide: no pairs")
            continue
        print(f"      ${w:g}-wide: n={v['n_pairs']} pairs / {v['n_chains']} "
              f"chains, DTE {v['dte_observed'][0]}-{v['dte_observed'][1]}, "
              f"short {v['short_zone_spread_pct']:.2%} wing "
              f"{v['wing_zone_spread_pct']:.2%}, wing_ratio "
              f"{v['wing_ratio']:.3f}, legsum/credit "
              f"{v['legsum_over_credit']:.2f}, credit/width "
              f"{v['credit_frac_median']:.3f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--underlyings", default="SPY,QQQ,IWM",
                   help="comma-separated, e.g. SPY,QQQ,IWM")
    p.add_argument("--source", default="auto",
                   choices=("auto", "snapshots", "live"),
                   help="auto = live in-session, stored snapshots when closed")
    p.add_argument("--since-days", type=int, default=120,
                   help="snapshots mode: look back this many calendar days")
    p.add_argument("--min-quotes", type=int, default=40,
                   help="minimum usable ZONE quotes per underlying to calibrate")
    p.add_argument("--min-pairs", type=int, default=10,
                   help="minimum vertical pairs per underlying to calibrate")
    p.add_argument("--dry-run", action="store_true",
                   help="report medians + decision line only; no file write")
    args = p.parse_args(argv)
    underlyings = [u.strip().upper() for u in args.underlyings.split(",") if u.strip()]
    if not underlyings:
        print("no underlyings given", file=sys.stderr)
        return 1

    from trading_agent.config import CONFIG
    source = _resolve_source(args.source)

    if source == "live":
        # --- fail fast when OpenD is unreachable (this runs on EC2 next to
        # trading-agent-opend.service; anywhere else it should die loudly,
        # not write a garbage calibration). -------------------------------
        try:
            quote_fns = _load_moomoo()
            preflight = quote_fns[0]([f"US.{underlyings[0]}"]).get("rows") or []
            if not preflight:
                raise RuntimeError("empty snapshot for preflight symbol")
        except Exception as e:
            print(
                f"FATAL: MoomooOpenD unreachable or not serving quotes at "
                f"{CONFIG.opend_host}:{CONFIG.opend_port} ({e}).\n"
                f"Run this on EC2 with trading-agent-opend.service active "
                f"during US market hours, or use --source snapshots.",
                file=sys.stderr,
            )
            return 2
        print(f"sampling {', '.join(underlyings)} LIVE — zone: {ZONE_DESC}")
        by_name = collect_live(underlyings, quote_fns)
        window = f"live {date.today().isoformat()}"
    else:
        since = date.today() - timedelta(days=max(1, args.since_days))
        print(f"reading option_chain_snapshots for {', '.join(underlyings)} "
              f"since {since} — zone: {ZONE_DESC}")
        try:
            by_name = collect_snapshots(underlyings, since)
        except Exception as e:
            print(f"FATAL: option_chain_snapshots read failed ({e}). Needs a "
                  f"reachable Postgres (POSTGRES_DSN or ~/.env.postgres).",
                  file=sys.stderr)
            return 2
        window = f"snapshots {since.isoformat()}..{date.today().isoformat()}"

    now = datetime.now(UTC).isoformat()
    calibrated: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for t in underlyings:
        stats = analyse(by_name.get(t) or [])
        _print_underlying_stats(t, stats)
        if stats["n_quotes"] < args.min_quotes or stats["n_pairs"] < args.min_pairs:
            why = (f"INSUFFICIENT DATA — {stats['n_quotes']} zone quotes "
                   f"(need {args.min_quotes}), {stats['n_pairs']} pairs "
                   f"(need {args.min_pairs})")
            skipped[t] = why
            print(f"  {t}: {why} — NOT calibrated")
            continue
        entry = build_entry(stats, source=source, sample_window=window, now=now)
        calibrated[t] = entry
        print(f"  {t}: CALIBRATED  short-zone "
              f"{entry['short_zone_spread_pct']:.2%}  wing-zone "
              f"{entry['wing_zone_spread_pct']:.2%}  pooled "
              f"{entry['median_spread_pct_mid']:.2%}  wing_ratio "
              f"{entry['wing_ratio']:.3f}  credit/width "
              f"{entry['credit_frac_median']:.3f}")

    # --- friction verdict per underlying ---------------------------------
    print(f"\ndecision line — modeled round-trip friction_r vs the "
          f"{DECISION_BAR_R}R whitelist bar:")
    for t in underlyings:
        if t in calibrated:
            _print_decision_block(t, calibrated[t])
        else:
            print(f"  {t}: NOT ASSESSED — {skipped.get(t, 'no data')}")
    if "IWM" in underlyings:
        entry = calibrated.get("IWM", {})
        fr = _worst_friction_r(entry)          # the plan's $5-wide decision line
        if fr is None:
            print("IWM NOT ASSESSED — insufficient zone data this run")
        elif fr > DECISION_BAR_R:
            print(f"IWM EXCLUDED (friction_r {fr:.3f}R > {DECISION_BAR_R}R on "
                  f"the plan's ${DECISION_WIDTH:g}-wide decision line) — the "
                  f"honest per-leg-mark model, measured in the zone the sleeve "
                  f"actually trades, puts IWM ABOVE the bar. The 0.031R "
                  f"printed on 2026-07-22 came from two errors: the net_credit "
                  f"leg-mark proxy (leg-mid-sum 2x credit vs a measured 6.4x) "
                  f"and a |delta| 0.03-0.40 sample zone.")
            best = _best_friction_r(entry)
            if best is not None and best <= DECISION_BAR_R:
                print(f"      NOTE: some measured width/credit combination "
                      f"does clear the bar ({best:.3f}R) — see the per-width "
                      f"lines above. Whether the sleeve may trade a WIDER "
                      f"structure instead is a spec decision, not something "
                      f"this report grants.")
        else:
            print(f"IWM OK (friction_r {fr:.3f}R <= {DECISION_BAR_R}R)")
        print("(report only — the whitelist change in the spec is the "
              "operator's call)")

    if not calibrated:
        print("no underlying reached --min-quotes/--min-pairs; nothing written.")
        return 1
    if args.dry_run:
        print("(dry-run — data/execution_costs.json untouched)")
        return 0

    # --- merge into data/execution_costs.json, GLOBAL keys untouched -----
    out_path = Path(CONFIG.data_dir) / "execution_costs.json"
    existing: dict = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"FATAL: refusing to overwrite unparseable {out_path}: {e}",
                  file=sys.stderr)
            return 1
        if not isinstance(existing, dict):
            print(f"FATAL: refusing to overwrite non-object JSON in {out_path} "
                  f"(top-level {type(existing).__name__})", file=sys.stderr)
            return 1
    per = dict(existing.get("per_underlying") or {})
    per.update(calibrated)
    existing["per_underlying"] = per
    _write_atomic(out_path, existing)
    print(f"written: {out_path} (per_underlying: {', '.join(sorted(per))})")

    from trading_agent import execution_costs as ec
    ec.reset_calibration_cache()  # this process sees the new figures at once
    _emit_calibration_event(calibrated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
