"""Position-sizing + concentration hard rules (R1-R6).

Pure functions, no I/O, no MCP dependencies. The same logic is called from:
  - pretool_order_guard.py (blocks orders at the hook layer)
  - future position-sizing skill (dry-run "why was this blocked")

Rules (from plan, canonical form):
  R1  single-trade risk       <= 2% equity
  R2  concurrent open positions <= 5
  R3  per-ticker exposure      <= 10% equity   (stock + options combined)
  R4  sector concentration     — max 2 already-open positions in the same GICS sector
  R5  options policy           — long premium only, 14 <= DTE <= 60,
                                0.30 <= |delta| <= 0.55, single-trade notional <= 1% equity
  R6  earnings lock            — within 2 trading days of earnings, only
                                strategy_labels starting with "earnings_" are allowed

Risk calc for R1:
  stock trade: risk = |entry - stop| * qty                      (if stop supplied)
               risk = 0.05 * entry * qty                         (implicit 5% stop fallback;
                                                                 a WARN is also emitted)
  long option: risk = contracts * 100 * entry_price              (max loss = debit paid)

Severity:
  "block" — hook must exit 2 (refuse the order)
  "warn"  — hook should log but allow (missing metadata, best-effort rule)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Public, stable rule codes — referenced in hook audit log and skills.
R1 = "R1_single_risk"
R2 = "R2_max_open"
R3 = "R3_ticker_exposure"
R4 = "R4_sector_concentration"
R5 = "R5_option_policy"
R6 = "R6_earnings_lock"
R_STOP_MISSING = "R1_stop_missing"          # warn-only signal when stop absent
R_SECTOR_UNKNOWN = "R4_sector_unknown"      # warn-only signal when sector lookup empty

MAX_SINGLE_RISK_PCT = 0.02
MAX_CONCURRENT_OPENS = 5
MAX_TICKER_EXPOSURE_PCT = 0.10
MAX_SAME_SECTOR_OPENS = 2
MAX_OPTION_NOTIONAL_PCT = 0.01
OPTION_DTE_MIN = 14
OPTION_DTE_MAX = 60
OPTION_DELTA_MIN = 0.30
OPTION_DELTA_MAX = 0.55
EARNINGS_LOCK_DAYS = 2
IMPLICIT_STOP_FRAC = 0.05  # fallback when caller doesn't provide a stop

# Tolerance for float comparisons (pennies-level).
EPS = 1e-6


@dataclass(frozen=True)
class OpenPosition:
    """A currently-open trade. Matches the `trades` table shape."""
    symbol: str                      # US.AAPL or US.AAPL250117C00200000
    ticker: str                      # bare: "AAPL"
    asset_type: Literal["STK", "OPT"]
    qty: float                       # shares (STK) or contracts (OPT)
    entry_price: float
    stop: float | None = None
    sector: str | None = None        # GICS sector, looked up from sectors.csv
    strategy_label: str | None = None

    @property
    def notional(self) -> float:
        """Dollar exposure at entry price. Options multiplied by contract-size 100."""
        if self.asset_type == "OPT":
            return self.qty * 100.0 * self.entry_price
        return self.qty * self.entry_price


@dataclass(frozen=True)
class ProposedTrade:
    """An order the LLM wants to place. All fields are what the hook can see at PreToolUse time."""
    ticker: str                      # bare symbol, e.g. "AAPL"
    asset_type: Literal["STK", "OPT"]
    side: Literal["BUY", "SELL"]
    qty: float
    entry_price: float               # limit price
    stop: float | None = None        # optional; fallback to 5% implicit stop for R1
    strategy_label: str | None = None
    sector: str | None = None        # resolved from sectors.csv by the hook
    # options-only enrichment (may be None if snapshot not fetched)
    delta: float | None = None
    dte: int | None = None
    # earnings gate — caller supplies trading-days-to-next-earnings if known
    earnings_dte: int | None = None

    @property
    def notional(self) -> float:
        if self.asset_type == "OPT":
            return self.qty * 100.0 * self.entry_price
        return self.qty * self.entry_price

    @property
    def max_loss(self) -> float:
        """Worst-case dollar loss on this trade, used for R1."""
        if self.asset_type == "OPT":
            # Long premium: max loss = debit paid. Short premium is refused by R5.
            return self.qty * 100.0 * self.entry_price
        # Stock:
        if self.stop is not None:
            return abs(self.entry_price - self.stop) * self.qty
        # Fallback: pretend stop is 5% away from entry — a deliberate under-estimate
        # is worse than refusing every stopless trade, so we warn + compute.
        return IMPLICIT_STOP_FRAC * self.entry_price * self.qty


@dataclass(frozen=True)
class SizingContext:
    """What sizing needs from the rest of the world at decision time.

    `sector_lookup_available` lets the hook tell us "I tried to look up sectors
    but sectors.csv is missing"; R4 then degrades to a warning rather than
    silently passing.
    """
    equity: float
    opens: tuple[OpenPosition, ...] = ()
    sector_lookup_available: bool = True


@dataclass(frozen=True)
class SizingViolation:
    rule: str                        # one of the R* codes above
    message: str
    severity: Literal["block", "warn"]


def check(ctx: SizingContext, proposed: ProposedTrade) -> list[SizingViolation]:
    """Run all rules and return every violation. Caller decides whether to
    block on any `severity=='block'` entry or to aggregate them for display.
    """
    violations: list[SizingViolation] = []

    # ---- R1 single-trade risk ----
    if proposed.asset_type == "STK" and proposed.stop is None:
        violations.append(SizingViolation(
            R_STOP_MISSING,
            f"stock order for {proposed.ticker} submitted without stop; "
            f"R1 estimated using implicit {IMPLICIT_STOP_FRAC*100:.0f}% stop",
            "warn",
        ))
    r1_budget = MAX_SINGLE_RISK_PCT * ctx.equity
    if proposed.max_loss > r1_budget + EPS:
        violations.append(SizingViolation(
            R1,
            f"max loss ${proposed.max_loss:,.2f} exceeds "
            f"{MAX_SINGLE_RISK_PCT*100:.0f}% of equity (${r1_budget:,.2f})",
            "block",
        ))

    # ---- R2 concurrent open count ----
    # Closing orders (SELL) shouldn't count toward opening a new slot.
    is_opening = proposed.side == "BUY"
    if is_opening and len(ctx.opens) >= MAX_CONCURRENT_OPENS:
        violations.append(SizingViolation(
            R2,
            f"already {len(ctx.opens)} open positions; "
            f"cap is {MAX_CONCURRENT_OPENS}",
            "block",
        ))

    # ---- R3 per-ticker exposure ----
    ticker_u = proposed.ticker.upper()
    existing = sum(p.notional for p in ctx.opens if p.ticker.upper() == ticker_u)
    proposed_exposure = proposed.notional if is_opening else 0.0
    total_exposure = existing + proposed_exposure
    r3_cap = MAX_TICKER_EXPOSURE_PCT * ctx.equity
    if total_exposure > r3_cap + EPS:
        violations.append(SizingViolation(
            R3,
            f"combined {ticker_u} exposure ${total_exposure:,.2f} "
            f"exceeds {MAX_TICKER_EXPOSURE_PCT*100:.0f}% of equity (${r3_cap:,.2f})",
            "block",
        ))

    # ---- R4 sector concentration ----
    if is_opening:
        if proposed.sector:
            same = [p for p in ctx.opens
                    if p.sector and p.sector == proposed.sector
                    and p.ticker.upper() != ticker_u]
            if len(same) >= MAX_SAME_SECTOR_OPENS:
                violations.append(SizingViolation(
                    R4,
                    f"already {len(same)} open positions in sector "
                    f"{proposed.sector!r}: {[p.ticker for p in same]}; cap "
                    f"is {MAX_SAME_SECTOR_OPENS}",
                    "block",
                ))
        elif not ctx.sector_lookup_available:
            violations.append(SizingViolation(
                R_SECTOR_UNKNOWN,
                "sectors.csv missing — R4 not enforced for this order",
                "warn",
            ))
        else:
            # Sector table loaded but ticker isn't in it — treat as unknown.
            violations.append(SizingViolation(
                R_SECTOR_UNKNOWN,
                f"sector for {ticker_u} not in sectors.csv — R4 not enforced "
                "for this order",
                "warn",
            ))

    # ---- R5 options policy ----
    if proposed.asset_type == "OPT" and is_opening:
        if proposed.side != "BUY":
            # Redundant with is_opening, but guards against future side enums.
            violations.append(SizingViolation(
                R5,
                f"options: long-premium only; got side={proposed.side}",
                "block",
            ))
        if proposed.dte is not None and not (OPTION_DTE_MIN <= proposed.dte <= OPTION_DTE_MAX):
            violations.append(SizingViolation(
                R5,
                f"options: DTE {proposed.dte} outside "
                f"[{OPTION_DTE_MIN},{OPTION_DTE_MAX}]",
                "block",
            ))
        if proposed.delta is not None:
            abs_d = abs(proposed.delta)
            if not (OPTION_DELTA_MIN <= abs_d <= OPTION_DELTA_MAX):
                violations.append(SizingViolation(
                    R5,
                    f"options: |delta| {abs_d:.2f} outside "
                    f"[{OPTION_DELTA_MIN:.2f},{OPTION_DELTA_MAX:.2f}]",
                    "block",
                ))
        r5_cap = MAX_OPTION_NOTIONAL_PCT * ctx.equity
        if proposed.notional > r5_cap + EPS:
            violations.append(SizingViolation(
                R5,
                f"options: single-trade notional ${proposed.notional:,.2f} "
                f"exceeds {MAX_OPTION_NOTIONAL_PCT*100:.0f}% of equity "
                f"(${r5_cap:,.2f})",
                "block",
            ))

    # ---- R6 earnings lock ----
    if (is_opening and proposed.earnings_dte is not None
            and 0 <= proposed.earnings_dte <= EARNINGS_LOCK_DAYS):
        lbl = proposed.strategy_label or ""
        if not lbl.startswith("earnings_"):
            violations.append(SizingViolation(
                R6,
                f"within {proposed.earnings_dte} trading day(s) of earnings; "
                f"only strategy_labels starting with 'earnings_' are allowed, "
                f"got {lbl!r}",
                "block",
            ))

    return violations


def blockers(vs: list[SizingViolation]) -> list[SizingViolation]:
    """Filter helper — hook should exit 2 iff this list is non-empty."""
    return [v for v in vs if v.severity == "block"]


def summarize(vs: list[SizingViolation]) -> str:
    """Multi-line human-readable rendering for hook stderr + audit log."""
    if not vs:
        return "OK (no violations)"
    lines = []
    for v in vs:
        tag = "BLOCK" if v.severity == "block" else "warn "
        lines.append(f"  [{tag}] {v.rule}: {v.message}")
    return "\n".join(lines)
