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


def _norm_underlying(underlying: str | None) -> str | None:
    """'US.SPY' / 'spy' → 'SPY'; None stays None."""
    if not underlying:
        return None
    u = str(underlying).strip().upper()
    if u.startswith("US."):
        u = u[3:]
    return u or None


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
            #            "calibrated_at": "...", "zone": "..."}, ...}
            # Only entries with a sane median (0 < x < 1, i.e. a fraction of
            # mid, not a percentage) are kept — a malformed entry silently
            # falls back to the global figure rather than poisoning costs.
            pu = raw.get("per_underlying")
            if isinstance(pu, dict):
                for sym, entry in pu.items():
                    if not isinstance(entry, dict):
                        continue
                    v = entry.get("median_spread_pct_mid")
                    key = _norm_underlying(sym)
                    if key and isinstance(v, (int, float)) and 0 < v < 1:
                        merged["per_underlying"][key] = {
                            **entry, "median_spread_pct_mid": float(v)}
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
            entry = c.get("per_underlying", {}).get(_norm_underlying(underlying) or "")
            if entry:
                # stored as FULL spread as fraction of mid → half-spread pct
                pct = entry["median_spread_pct_mid"] / 2.0
        half = p * pct
    return round(half * q * mult, 4)


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


def friction_r(
    underlying: str | None,
    width: float,
    net_credit: float,
    contracts: int = 1,
) -> float:
    """Modeled round-trip friction of a defined-risk credit vertical, in R.

    ``R = max_loss = (width − net_credit)`` per contract-set (× 100 × qty in
    dollars). The friction stack (REVIVAL_PLAN M1-0.3):

    * **fees on 4 fills** — 2 legs × (open + close);
    * **spread on 2 legs × 2 directions** — each leg crosses half the quoted
      spread at entry AND at exit, at the per-underlying calibrated spread
      (global percent-of-mark fallback when the name is uncalibrated).

    Per-leg mark proxy: ``net_credit``. The true per-leg spread cost sums
    over both leg mids (short + long); with only (width, net_credit) known we
    charge each of the two legs at the net_credit mark, i.e. leg-mid-sum ≈
    2 × net_credit. For short − long = net_credit and wing ratio
    long/short = r this is exact at r = 1/3 (the typical 30-45 DTE vertical)
    and conservative — overstates friction — for cheaper wings (r < 1/3).

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
    fees = 4.0 * fees_per_side(n, "OPT")                    # 2 legs × 2 dirs
    spread = 4.0 * half_spread_cost(                        # 2 legs × 2 dirs
        cr, n, "OPT", underlying=underlying)
    risk = (w - cr) * 100.0 * n
    return round((fees + spread) / risk, 6)


def reset_calibration_cache() -> None:
    """Test hook — force the next call to re-read data/execution_costs.json."""
    global _calibrated
    _calibrated = None


__all__ = [
    "CostBreakdown",
    "fees_per_side",
    "friction_r",
    "half_spread_cost",
    "net_pnl",
    "reset_calibration_cache",
    "round_trip_costs",
    "round_trip_fee_per_unit",
]
