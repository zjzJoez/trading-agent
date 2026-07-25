"""Execution cost model — the single source of truth for trading friction.

Phase 0 measurement fix: paper P&L was recorded mid/last-to-last with zero
costs, so every paper statistic (journal pnl, realized_r, replay scores, the
soak Sharpe gate) overstated expectancy by the full cost stack. For the
system's own trade geometry (premium stop −50% / target +65%, R:R 1.3) a
realistic 8–13% round-trip friction moves the breakeven win rate from 43.5%
to 50–57% — large enough to flip a marginal edge negative. Every consumer of
trade economics must charge costs through THIS module so paper results stay
admissible evidence for real-money decisions.

Two cost components:

1. **Fees** — commission + regulatory, charged per side always, even when the
   fill price is a real dealt price.
2. **Half-spread penalty** — charged ONLY when a price is a mark (mid/last)
   rather than a broker dealt price: virtual fills, fallback closes written
   at the last mark, replay counterfactuals. A real dealt price already
   embeds the spread, so penalizing it again would double-count.

Calibration: defaults below are deliberately conservative for single-name US
equity options. ``scripts/calibrate_execution_costs.py`` measures live quoted
spreads for the traded universe and writes ``data/execution_costs.json``,
which overrides the defaults at import time (re-read lazily once per process).
``scripts/calibrate_index_options.py`` adds the per-underlying / per-ZONE
figures (short-leg zone vs long-wing zone spreads, wing ratio, credit/width)
under that file's ``per_underlying`` key, from live quotes or from the stored
``option_chain_snapshots`` feed.

A vertical's friction is NOT two single-leg frictions at the net credit — see
:func:`friction_r`, whose net_credit-as-leg-mark proxy was measured wrong by
~3x and fixed 2026-07-25.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# --- defaults (overridable via data/execution_costs.json) -------------------
# Moomoo US pricing posture: options ≈ $0.65/contract commission + platform &
# regulatory fees ≈ $0.35 → ~$1.00/contract/side. Stocks are commission-free
# but pay SEC/TAF on sells; ~$0.005/share is a conservative blended figure.
_DEFAULTS = {
    "opt_fee_per_contract_per_side": 1.00,   # USD
    "stk_fee_per_share_per_side": 0.005,     # USD
    # Half-spread as a fraction of mark, used only for non-dealt prices.
    # 4% half-spread ≈ 8% full quoted spread — the low end of what
    # single-name 14–60 DTE options actually quote (5–20%).
    "opt_half_spread_pct_of_mark": 0.04,
    "stk_half_spread_pct_of_mark": 0.0005,   # 10 bps full spread
}

_CALIBRATION_FILENAME = "execution_costs.json"
_calibrated: dict | None = None

# --- measured wing ratios (see friction_r) ----------------------------------
# r = long_mark / short_mark for a real credit vertical. MEASURED 2026-07-25
# from option_chain_snapshots (EC2 Postgres): IWM PUTS, 30–37 DTE inside the
# 30–45 band, short-leg |delta| 0.20–0.35, bid>0 & ask>bid, mid = (bid+ask)/2,
# each width from 3 distinct (snapshot_date, expiry) chains:
#   $5-wide  : r median 0.731, p75 0.734  (n=32 pairs, 3 chains)
#   $10-wide : r median 0.530, p75 0.536  (n=25 pairs, 3 chains)
# The textbook "r ≈ 1/3" is simply false for a narrow vertical on a
# high-priced index: at $220 spot a $5 width is 2.3% of spot, so the wing is
# only a couple of vol-points cheaper than the short leg.
#
# WHY TWO DICTS, AND WHY THE p75 IS THE FALLBACK (fixed 2026-07-25)
# -----------------------------------------------------------------
# The modeled spread bill is  legsum(r) = cr·(1+r)/(1−r),  whose derivative
#   d/dr [cr·(1+r)/(1−r)] = 2·cr/(1−r)²  >  0  for all r ∈ [0, 1),
# so the bill is STRICTLY INCREASING in r on the whole admissible range. A
# MEDIAN r is therefore not a conservative constant at all: by construction
# half of all real pairs sit above it, and for every one of those the median
# UNDERSTATES the friction. Using the median as the blind fallback (as this
# module did until 2026-07-25) errs in exactly the wrong direction — the same
# direction that produced the fictitious 0.031R IWM verdict.
# The blind fallback is therefore an UPPER quantile (p75), which understates
# only for the top quartile and overstates for the other three.
MEASURED_WING_RATIO = {5.0: 0.731, 10.0: 0.530}          # medians (provenance)
MEASURED_WING_RATIO_P75 = {5.0: 0.734, 10.0: 0.536}      # upper quantile
# Fallback when no leg marks and no calibrated ratio are available: the LARGEST
# measured p75, i.e. maximally conservative in both dimensions (quantile and
# width). Overstating friction can only keep marginal trades out; understating
# it lets negative-EV trades through, which is how this bug cost the plan its
# IWM exclusion.
DEFAULT_WING_RATIO = max(MEASURED_WING_RATIO_P75.values())
# Credibility floor for any ratio resolved from the calibration FILE or from a
# default — never below the smallest ratio ever measured on a real chain. A
# file value of 0 (or 0.01) would delete the wing's spread bill: measured, IWM
# $5-wide at credit 0.955 reports 0.0207R with wing_ratio 0 against an honest
# 0.0794R, i.e. one bad write buys a 74% discount on the modeled bill and a
# fictitious pass of the 0.06R whitelist bar. Width-BLIND on purpose: these
# medians are measured on IWM only, so a per-width floor would silently
# overwrite a legitimately calibrated figure for a name whose chain really does
# quote a cheaper wing. Explicit caller-supplied ratios are NOT floored — those
# are a caller asserting a known structure, not untrusted persisted data.
MIN_WING_RATIO = min(MEASURED_WING_RATIO.values())


def _norm_underlying(underlying: str | None) -> str | None:
    """'US.SPY' / 'spy' → 'SPY'; None stays None."""
    if not underlying:
        return None
    u = str(underlying).strip().upper()
    if u.startswith("US."):
        u = u[3:]
    return u or None


def _clean_ratios(d: dict) -> None:
    """In-place: keep only wing-ratio fields that are floats in ``(0, 1)``.

    Strictly ``> 0``: a ratio of exactly 0 claims the long wing is free and so
    contributes nothing to the spread bill, which is never true of a quoted
    option and is the shape a truncated/zeroed calibration write takes.
    """
    for fld in ("wing_ratio", "wing_ratio_p75", "wing_ratio_median"):
        v = d.get(fld)
        if isinstance(v, (int, float)) and 0 < float(v) < 1:
            d[fld] = float(v)
        else:
            d.pop(fld, None)


def _load_calibration() -> dict:
    """Merge calibrated values over defaults. Lazy, once per process."""
    global _calibrated
    if _calibrated is not None:
        return _calibrated
    merged = dict(_DEFAULTS)
    merged["per_underlying"] = {}
    try:
        from trading_agent.config import CONFIG
        path = Path(CONFIG.data_dir) / _CALIBRATION_FILENAME
        if path.exists():
            raw = json.loads(path.read_text())
            for k in _DEFAULTS:
                v = raw.get(k)
                if isinstance(v, (int, float)) and v >= 0:
                    merged[k] = float(v)
            # M1-0.3 per-underlying spread calibration. Shape:
            #   {"SPY": {"median_spread_pct_mid": 0.008, "n_quotes": 62,
            #            "short_zone_spread_pct": 0.0042,   # short-leg zone
            #            "wing_zone_spread_pct": 0.0050,    # long-wing zone
            #            "wing_ratio": 0.731, "credit_frac_median": 0.191,
            #            "calibrated_at": "...", "zone": "..."}, ...}
            # Every numeric figure is validated independently: a fraction of
            # mid must be 0 < x < 1 (a percentage like 7.3 is a unit mistake)
            # and a wing ratio must be 0 < r < 1 — r >= 1 means the wing is
            # worth more than the short leg (not a credit vertical) and r <= 0
            # means the wing is FREE, which zeroes its half of the spread bill
            # outright (measured: IWM $5 at credit 0.955 reports 0.0207R with
            # wing_ratio 0 vs the honest 0.0794R). Since file values win over
            # the measured default, accepting 0 here was a way for one bad
            # calibration write to silently delete 74% of the modeled bill.
            # A malformed field is dropped so the caller falls back to the next
            # source, rather than poisoning costs with it.
            pu = raw.get("per_underlying")
            if isinstance(pu, dict):
                for sym, entry in pu.items():
                    key = _norm_underlying(sym)
                    if not key or not isinstance(entry, dict):
                        continue
                    clean = dict(entry)
                    for fld in ("median_spread_pct_mid",
                                "short_zone_spread_pct",
                                "wing_zone_spread_pct"):
                        v = clean.get(fld)
                        if isinstance(v, (int, float)) and 0 < v < 1:
                            clean[fld] = float(v)
                        else:
                            clean.pop(fld, None)
                    _clean_ratios(clean)
                    # Per-width ratio blocks (written by
                    # scripts/calibrate_index_options.py, already gated on
                    # --min-chains there) get the same field validation, and a
                    # block left with no usable ratio is dropped entirely so it
                    # cannot shadow the name-level figure with metadata.
                    bw = clean.get("by_width")
                    clean_bw: dict[float, dict] = {}
                    if isinstance(bw, dict):
                        for wkey, wval in bw.items():
                            try:
                                w = float(wkey)
                            except (TypeError, ValueError):
                                continue
                            if w <= 0 or not isinstance(wval, dict):
                                continue
                            wclean = dict(wval)
                            _clean_ratios(wclean)
                            if wclean.keys() & {"wing_ratio", "wing_ratio_p75"}:
                                clean_bw[w] = wclean
                    clean["by_width"] = clean_bw
                    # An entry with no usable spread figure at all would only
                    # shadow the global fallback with metadata — drop it,
                    # UNLESS it still carries a usable wing ratio (which
                    # friction_r consults independently of the spread).
                    if not (clean.keys() & {"median_spread_pct_mid",
                                            "short_zone_spread_pct",
                                            "wing_zone_spread_pct",
                                            "wing_ratio",
                                            "wing_ratio_p75"}) and not clean_bw:
                        continue
                    merged["per_underlying"][key] = clean
            log.info("execution_costs: calibration loaded from %s", path)
    except Exception as e:  # never let cost loading break a trade path
        log.warning("execution_costs: calibration load failed (%s) — defaults", e)
    _calibrated = merged
    return merged


@dataclass(frozen=True)
class CostBreakdown:
    """Round-trip friction for one position, in account dollars (≥ 0)."""

    entry_fees: float
    exit_fees: float
    entry_spread_cost: float   # 0.0 when the entry price was a real dealt price
    exit_spread_cost: float    # 0.0 when the exit price was a real dealt price

    @property
    def total(self) -> float:
        return round(
            self.entry_fees + self.exit_fees
            + self.entry_spread_cost + self.exit_spread_cost, 4)


def fees_per_side(qty: float, asset_type: str) -> float:
    """Commission + regulatory fees for one side of a trade, in dollars."""
    c = _load_calibration()
    q = abs(float(qty or 0.0))
    if str(asset_type).upper() == "OPT":
        return round(q * c["opt_fee_per_contract_per_side"], 4)
    return round(q * c["stk_fee_per_share_per_side"], 4)


def half_spread_cost(
    price: float,
    qty: float,
    asset_type: str,
    *,
    bid: float | None = None,
    ask: float | None = None,
    underlying: str | None = None,
    zone: str | None = None,
) -> float:
    """Dollar cost of crossing half the spread at ``price`` for ``qty`` units.

    Uses the live quoted spread when bid/ask are both present and sane;
    otherwise falls back to the calibrated percent-of-mark estimate. Only
    call this for NON-dealt prices (marks, mids, virtual fills).

    ``underlying`` (optional, options only): prefer that name's M1-0.3
    per-underlying calibrated spread over the global figure — index chains
    (SPY/QQQ) quote far tighter than the single-name watchlist median the
    global number was calibrated on. Unknown/uncalibrated names fall back
    to the global percent-of-mark; live bid/ask still wins over both.

    ``zone`` (optional, options only): ``"short"`` or ``"wing"``. The two
    legs of a vertical do NOT quote the same percent spread — MEASURED on
    IWM $5-wides, 4.24% at the short leg vs 5.04% at the wing (the wing is
    cheaper in dollars, so the same penny spread is a bigger fraction of
    its mid). When the calibration carries the zone figure it is preferred
    over that name's pooled median; unknown zone falls back to the pooled
    median, then to the global percent-of-mark.
    """
    c = _load_calibration()
    is_opt = str(asset_type).upper() == "OPT"
    mult = 100.0 if is_opt else 1.0
    q = abs(float(qty or 0.0))
    p = float(price or 0.0)
    if p <= 0 or q <= 0:
        return 0.0
    if bid is not None and ask is not None and 0 < float(bid) <= float(ask):
        half = (float(ask) - float(bid)) / 2.0
    else:
        pct = c["opt_half_spread_pct_of_mark"] if is_opt else c["stk_half_spread_pct_of_mark"]
        if is_opt:
            full = _calibrated_spread_pct(underlying, zone)
            if full is not None:
                # stored as FULL spread as fraction of mid → half-spread pct
                pct = full / 2.0
        half = p * pct
    return round(half * q * mult, 4)


def _calibrated_spread_pct(
    underlying: str | None, zone: str | None = None
) -> float | None:
    """Per-underlying FULL quoted spread as a fraction of mid, or None.

    Preference: the requested ``zone``'s measured figure → that name's
    pooled median → None (caller uses the global default).
    """
    entry = _load_calibration().get("per_underlying", {}).get(
        _norm_underlying(underlying) or "")
    if not entry:
        return None
    keys = []
    if zone == "short":
        keys.append("short_zone_spread_pct")
    elif zone == "wing":
        keys.append("wing_zone_spread_pct")
    keys.append("median_spread_pct_mid")
    for k in keys:
        v = entry.get(k)
        # Only _load_calibration-sanitized values are floats; anything the
        # validator rejected was popped, so a leftover non-float means "not
        # measured" and must fall through, never into the cost.
        if isinstance(v, float):
            return v
    return None


def round_trip_costs(
    qty: float,
    asset_type: str,
    *,
    entry_price: float,
    exit_price: float,
    entry_is_dealt: bool,
    exit_is_dealt: bool,
    entry_bid: float | None = None,
    entry_ask: float | None = None,
    exit_bid: float | None = None,
    exit_ask: float | None = None,
    underlying: str | None = None,
) -> CostBreakdown:
    """Full friction stack for one round trip.

    ``*_is_dealt=True`` means the price is a broker-confirmed fill — the
    spread is already paid inside the price, so only fees apply for that side.
    """
    return CostBreakdown(
        entry_fees=fees_per_side(qty, asset_type),
        exit_fees=fees_per_side(qty, asset_type),
        entry_spread_cost=0.0 if entry_is_dealt else half_spread_cost(
            entry_price, qty, asset_type, bid=entry_bid, ask=entry_ask,
            underlying=underlying),
        exit_spread_cost=0.0 if exit_is_dealt else half_spread_cost(
            exit_price, qty, asset_type, bid=exit_bid, ask=exit_ask,
            underlying=underlying),
    )


def net_pnl(
    gross_pnl: float,
    qty: float,
    asset_type: str,
    *,
    entry_price: float,
    exit_price: float,
    entry_is_dealt: bool,
    exit_is_dealt: bool,
    entry_bid: float | None = None,
    entry_ask: float | None = None,
    exit_bid: float | None = None,
    exit_ask: float | None = None,
    underlying: str | None = None,
) -> tuple[float, CostBreakdown]:
    """``(gross − friction, breakdown)`` for one closed position."""
    costs = round_trip_costs(
        qty, asset_type,
        entry_price=entry_price, exit_price=exit_price,
        entry_is_dealt=entry_is_dealt, exit_is_dealt=exit_is_dealt,
        entry_bid=entry_bid, entry_ask=entry_ask,
        exit_bid=exit_bid, exit_ask=exit_ask,
        underlying=underlying,
    )
    return round(float(gross_pnl) - costs.total, 2), costs


def round_trip_fee_per_unit(asset_type: str) -> float:
    """Round-trip fees expressed per PRICE UNIT (per share / per option point).

    Lets realized-R math subtract fees in price space:
    ``r_net = (exit − entry − round_trip_fee_per_unit) / risk_per_unit``.
    For options $1/contract/side → $2 round trip → 0.02 price points.
    """
    c = _load_calibration()
    if str(asset_type).upper() == "OPT":
        return round(2.0 * c["opt_fee_per_contract_per_side"] / 100.0, 6)
    return round(2.0 * c["stk_fee_per_share_per_side"], 6)


def leg_marks_from_wing_ratio(
    net_credit: float, wing_ratio: float
) -> tuple[float, float]:
    """``(short_mark, long_mark)`` implied by a credit and a wing ratio.

    Solving ``short − long = net_credit`` with ``long = r · short`` gives
    ``short = cr/(1−r)``, ``long = cr·r/(1−r)``, so the leg-mid-sum is
    ``cr·(1+r)/(1−r)``. Raises on r outside ``[0, 1)`` — a caller that can't
    guarantee the range should go through :func:`friction_r`, which degrades
    to the measured default instead of raising in a trade path.
    """
    cr = float(net_credit)
    r = float(wing_ratio)
    if cr <= 0:
        raise ValueError(f"net_credit must be > 0, got {net_credit}")
    if not 0.0 <= r < 1.0:
        raise ValueError(f"wing_ratio must be in [0, 1), got {wing_ratio}")
    short = cr / (1.0 - r)
    return short, short * r


def _ratio_from_width_map(
    by_width: dict[float, float], width: float
) -> tuple[float, float] | None:
    """``(ratio, measured_width)`` for ``width``, conservatively, or None.

    r falls as width grows (measured 0.731 at $5 vs 0.530 at $10 — a $10 wing
    sits further OTM relative to the same short leg), and the spread bill is
    strictly increasing in r, so:

    * an EXACT width match is the honest figure and always wins;
    * otherwise take the MAX over measured widths **<= the traded width** — a
      narrower measured width yields a LARGER r, which OVERSTATES the bill;
    * a measured width WIDER than the traded one would understate r, so it is
      never used here; the caller falls through to the name-level scalar (which
      is itself a max over widths) or to the conservative default.
    """
    if not by_width or not (width > 0):
        return None
    for w, r in by_width.items():
        if abs(w - width) < 1e-9:
            return float(r), float(w)
    below = [(float(r), float(w)) for w, r in by_width.items() if w <= width]
    return max(below) if below else None


def _calibrated_wing_ratio(
    underlying: str | None, width: float
) -> tuple[float, str] | None:
    """``(ratio, provenance)`` from data/execution_costs.json, or None.

    Width-aware (the ratio is strongly width-dependent) and p75-preferring
    (the bill is increasing in r, so the median is the wrong tail).
    """
    entry = _load_calibration().get("per_underlying", {}).get(
        _norm_underlying(underlying) or "")
    if not entry:
        return None
    by_width = {}
    for w, blk in (entry.get("by_width") or {}).items():
        r = blk.get("wing_ratio_p75")
        if r is None:
            r = blk.get("wing_ratio")
        if r is not None:
            by_width[w] = r
    hit = _ratio_from_width_map(by_width, width)
    if hit is not None:
        r, mw = hit
        exact = abs(mw - width) < 1e-9
        return r, ("wing_ratio_calibrated_width" if exact
                   else f"wing_ratio_calibrated_width_{mw:g}_le")
    r = entry.get("wing_ratio_p75")
    if r is None:
        r = entry.get("wing_ratio")
    if r is None:
        return None
    return float(r), "wing_ratio_calibrated"


def _resolve_leg_marks(
    underlying: str | None,
    width: float,
    net_credit: float,
    short_mark: float | None,
    long_mark: float | None,
    wing_ratio: float | None,
) -> tuple[float, float, str]:
    """``(short_mark, long_mark, provenance)`` for the spread bill.

    Never raises: a nonsensical input is logged and degraded to the next
    source, ending at the measured (conservative) default ratio — a cost
    model must not be able to abort a trade-decision path.

    Ratio preference, in order:

    1. an explicit ``wing_ratio`` argument (a caller asserting the structure);
    2. that name's calibrated ratio for THIS width — p75 first, exact width
       first, else the max over narrower measured widths;
    3. that name's calibrated name-level p75/ratio;
    4. the measured p75 constants, again width-aware;
    5. :data:`DEFAULT_WING_RATIO`.

    Everything from source 2 downwards is floored at :data:`MIN_WING_RATIO`;
    source 1 is not (see that constant's note).
    """
    cr = float(net_credit)
    if short_mark is not None and long_mark is not None:
        s, lo = float(short_mark), float(long_mark)
        if s > 0 and lo >= 0 and s > lo:
            implied = s - lo
            if abs(implied - cr) > 0.20 * max(cr, 0.01):
                # The marks and the credit describe different structures
                # (stale quote, wrong leg, mixed mid/fill). Keep the marks —
                # they are still the only per-leg information available —
                # but make the mismatch visible instead of silent.
                log.warning(
                    "friction_r: leg marks imply credit %.4f but net_credit "
                    "is %.4f (%s) — using the marks for the spread bill",
                    implied, cr, underlying)
            return s, lo, "leg_marks"
        log.warning(
            "friction_r: unusable leg marks short=%r long=%r (%s) — falling "
            "back to the wing-ratio model", short_mark, long_mark, underlying)
    # 1. explicit argument — trusted as given (not floored), only range-checked.
    if wing_ratio is not None:
        if 0.0 < float(wing_ratio) < 1.0:
            s, lo = leg_marks_from_wing_ratio(cr, float(wing_ratio))
            return s, lo, "wing_ratio_arg"
        log.warning("friction_r: ignoring out-of-range wing_ratio %r (%s)",
                    wing_ratio, underlying)

    # 2/3. the calibration file, width-aware and p75-preferring.
    hit = _calibrated_wing_ratio(underlying, width)
    if hit is not None:
        r, src = hit
    else:
        # 4/5. the measured constants, width-aware, else the global max p75.
        const = _ratio_from_width_map(MEASURED_WING_RATIO_P75, float(width))
        if const is not None:
            r, mw = const
            src = ("wing_ratio_measured_width" if abs(mw - float(width)) < 1e-9
                   else f"wing_ratio_measured_width_{mw:g}_le")
        else:
            r, src = DEFAULT_WING_RATIO, "wing_ratio_default"
    # Floor every non-explicit source at the smallest ratio ever measured on a
    # real chain: below that a value is not a measurement but a corruption, and
    # because the bill is increasing in r it would UNDERSTATE friction. The
    # floor is deliberately width-BLIND (the global min, not the per-width
    # median): MEASURED_WING_RATIO comes from IWM only, so a per-width floor
    # would silently overwrite a legitimately calibrated figure for a name
    # whose chain really does quote a cheaper wing.
    if float(r) < MIN_WING_RATIO:
        log.warning(
            "friction_r: %s ratio %.4f below the measured floor %.4f (%s) — "
            "flooring", src, float(r), MIN_WING_RATIO, underlying)
        r, src = MIN_WING_RATIO, f"{src}_floored"
    s, lo = leg_marks_from_wing_ratio(cr, float(r))
    return s, lo, src


def friction_r(
    underlying: str | None,
    width: float,
    net_credit: float,
    contracts: int = 1,
    *,
    short_mark: float | None = None,
    long_mark: float | None = None,
    wing_ratio: float | None = None,
    short_bid: float | None = None,
    short_ask: float | None = None,
    long_bid: float | None = None,
    long_ask: float | None = None,
) -> float:
    """Modeled round-trip friction of a defined-risk credit vertical, in R.

    ``R = max_loss = (width − net_credit)`` per contract-set (× 100 × qty in
    dollars). The friction stack (REVIVAL_PLAN M1-0.3):

    * **fees on 4 fills** — 2 legs × (open + close);
    * **spread on 2 legs × 2 directions** — each leg crosses half of ITS OWN
      spread at entry AND at exit, priced at ITS OWN mark. Per leg the spread
      comes from that leg's LIVE touch (``short_bid``/``short_ask``,
      ``long_bid``/``long_ask``) when supplied, else that name's per-zone
      calibrated percent-of-mid, else the global percent-of-mark. The fallback
      is per leg: a missing wing quote does not discard a good short quote.

    The spread bill therefore needs the two LEG marks, not just the net
    credit. Preference order:

    1. ``short_mark`` + ``long_mark`` — exact. The live sizing path has both
       (``sizing.ProposedCombo`` carries per-leg ``price``); use
       :func:`combo_friction_r` there, and pass ``leg_quotes`` so the touch
       drives the spread too.
    2. ``wing_ratio`` r = long/short → leg-mid-sum ``cr·(1+r)/(1−r)``.
    3. that name's calibrated ratio for THIS ``width`` (p75, from
       data/execution_costs.json ``by_width``), then its name-level ratio.
    4. :data:`MEASURED_WING_RATIO_P75` for this width, then
       :data:`DEFAULT_WING_RATIO` — MEASURED, not assumed.

    ``width`` is load-bearing for the ratio, not just for R: r is strongly
    width-dependent (measured 0.731 at $5 vs 0.530 at $10), so a width-blind
    ratio misprices one of the two structures by ~2× in legsum/credit.

    **The r = 1/3 assumption this function used until 2026-07-25 was wrong.**
    It proxied every leg's mark by ``net_credit`` (leg-mid-sum ≈ 2·cr), which
    is exact only when ``cr(1+r)/(1−r) = 2cr`` ⟺ r = 1/3. Real quotes
    (option_chain_snapshots, IWM PUTS, 30–37 DTE, short |delta| 0.20–0.35):
    r median **0.731** / p75 0.734 on $5-wides (n=32 pairs, 3 date×expiry
    chains) and **0.530** / p75 0.536 on $10-wides (n=25, 3 chains) —
    leg-mid-sum/credit of 6.43 and 3.26 vs the assumed 2.0, i.e. the deployed
    model understated the vertical spread bill by ~3.2× on $5-wides. That
    single error was the difference between "IWM 0.031R, inside the 0.06R bar"
    and the honest ~0.08R that fails it.

    Fallback direction is deliberately CONSERVATIVE, and provably so:
    ``d/dr [cr(1+r)/(1−r)] = 2cr/(1−r)² > 0`` on ``[0, 1)``, so the bill is
    strictly increasing in r. Every blind fallback here is therefore an UPPER
    quantile of the measured ratio (:data:`MEASURED_WING_RATIO_P75`), taken at
    the narrowest applicable width — both choices push the modeled friction UP.
    A median would understate the bill for every structure above it, which is
    the wrong direction: overstating friction only keeps marginal trades out,
    understating it lets negative-expectancy trades in.

    Both the friction stack and R scale linearly in ``contracts``, so the
    result is contracts-invariant; the parameter exists for call-site
    clarity and future non-linear fee schedules.
    """
    w = float(width)
    cr = float(net_credit)
    n = int(contracts)
    if n < 1:
        raise ValueError(f"contracts must be >= 1, got {contracts}")
    if not (0 < cr < w):
        raise ValueError(
            f"need 0 < net_credit < width for a credit vertical, "
            f"got width={width}, net_credit={net_credit}")
    s_mark, l_mark, _src = _resolve_leg_marks(
        underlying, w, cr, short_mark, long_mark, wing_ratio)
    fees = 4.0 * fees_per_side(n, "OPT")            # 2 legs × (open + close)
    spread = 2.0 * (                                # entry + exit
        half_spread_cost(s_mark, n, "OPT", bid=short_bid, ask=short_ask,
                         underlying=underlying, zone="short")
        + half_spread_cost(l_mark, n, "OPT", bid=long_bid, ask=long_ask,
                           underlying=underlying, zone="wing")
    )
    risk = (w - cr) * 100.0 * n
    return round((fees + spread) / risk, 6)


def combo_friction_r(
    combo, contracts: int = 1, *, leg_quotes: dict | None = None
) -> float | None:
    """:func:`friction_r` for a live two-leg combo, using its REAL leg marks.

    Duck-typed on ``sizing.ProposedCombo`` (``ticker``, ``width``,
    ``net_credit``, ``short_leg.price``, ``long_leg.price``) so the cost
    model does not import the sizing layer. Returns None — never raises —
    when the structure is not a credit vertical the model can price
    (missing leg, zero width, non-positive or width-exceeding credit); the
    caller is a guard/report path that must survive garbage input.

    ``leg_quotes`` is the role-keyed touch dict the order guard already builds
    for its liquidity gate — ``{"short": {"bid": .., "ask": ..}, "long": {..}}``
    — and is threaded straight through to :func:`friction_r`. Passing it is the
    difference between pricing friction off the ACTUAL spread the trade will
    cross and pricing it off a calibrated median measured on other days: the
    guard is the one call site that holds fresh quotes, so it must use them.
    A missing/unusable side simply falls back to the calibration for that leg.
    """
    def _touch(role: str, field: str) -> float | None:
        q = (leg_quotes or {}).get(role)
        if not isinstance(q, dict):
            return None
        v = q.get(field)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if v > 0 else None

    try:
        short_leg = getattr(combo, "short_leg", None)
        long_leg = getattr(combo, "long_leg", None)
        if short_leg is None or long_leg is None:
            return None
        return friction_r(
            getattr(combo, "ticker", None),
            float(combo.width),
            float(combo.net_credit),
            contracts,
            short_mark=float(short_leg.price),
            long_mark=float(long_leg.price),
            short_bid=_touch("short", "bid"), short_ask=_touch("short", "ask"),
            long_bid=_touch("long", "bid"), long_ask=_touch("long", "ask"),
        )
    except (AttributeError, TypeError, ValueError) as e:
        log.debug("combo_friction_r: not priceable (%s)", e)
        return None


def reset_calibration_cache() -> None:
    """Test hook — force the next call to re-read data/execution_costs.json."""
    global _calibrated
    _calibrated = None


__all__ = [
    "CostBreakdown",
    "DEFAULT_WING_RATIO",
    "MEASURED_WING_RATIO",
    "MEASURED_WING_RATIO_P75",
    "MIN_WING_RATIO",
    "combo_friction_r",
    "fees_per_side",
    "friction_r",
    "half_spread_cost",
    "leg_marks_from_wing_ratio",
    "net_pnl",
    "reset_calibration_cache",
    "round_trip_costs",
    "round_trip_fee_per_unit",
]
