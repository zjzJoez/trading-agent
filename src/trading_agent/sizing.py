"""Position-sizing + concentration hard rules (R1-R6).

Pure functions, no I/O, no MCP dependencies. The same logic is called from:
  - pretool_order_guard.py (blocks orders at the hook layer)
  - future position-sizing skill (dry-run "why was this blocked")

Rules (from plan, canonical form):
  R1  single-trade risk       <= 2.5% equity
  R2  concurrent open positions <= 6
  R3  per-ticker exposure      <= 12% equity   (stock + options combined)
  R4  sector concentration     — max 2 already-open positions in the same GICS sector
  R5  options policy           — long + short premium allowed; 14 <= DTE <= 60,
                                0.25 <= |delta| <= 0.65, single-trade notional <= 1.5% equity
  R5b cash-secured put         — short put requires cash >= strike * 100 * qty
                                (premium already collected may be subtracted)
  R5c naked short call         — explicit hard stop required; max_loss uses
                                stress-buffered stop distance (1.5× to bound
                                the unbounded theoretical risk via R1)
  R6  earnings lock            — within 2 trading days of earnings, only
                                strategy_labels starting with "earnings_" are allowed
  R7  risk:reward floor (LONG) — every opening LONG trade must have reward/risk >= 1.3.
                                SHORT premium is exempt (R:R structurally capped).
                                Strategy specs may TIGHTEN the floor per label
                                (strategy_specs.spec_for_label); the global 1.3
                                stays the absolute minimum for unmapped labels.

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
R7 = "R7_risk_reward"
R5B = "R5b_csp_collateral"
R5C = "R5c_naked_call_no_stop"
R5E = "R5e_combo_defined_risk"               # multi-leg vertical defined-risk gate
R_STOP_MISSING = "R1_stop_missing"          # warn-only signal when stop absent
R_SECTOR_UNKNOWN = "R4_sector_unknown"      # warn-only signal when sector lookup empty
R_TARGET_MISSING = "R7_target_missing"      # warn-only signal when target absent

# ---- Threshold raises (2026-05-22) ----
# User explicitly requested more aggressive sizing.  Senior risk hat raises:
#   R1: 2%   → 2.5%   (single-trade risk budget)
#   R2: 5    → 6      (concurrent open positions)
#   R3: 10%  → 12%    (per-ticker exposure)
#   R5 notional cap: 1% → 1.5%  (single-trade option notional)
#   R5 delta band: 0.30-0.55 → 0.25-0.65
#   R7 R:R floor: 1.5 → 1.3 (still positive-EV at ~55% baseline win rate)
# Combined max heat raised from 10% (R1 × R2) to 15% (2.5% × 6).
MAX_SINGLE_RISK_PCT = 0.025
MAX_CONCURRENT_OPENS = 6
MAX_TICKER_EXPOSURE_PCT = 0.12
MAX_SAME_SECTOR_OPENS = 2
MAX_OPTION_NOTIONAL_PCT = 0.015
OPTION_DTE_MIN = 14
OPTION_DTE_MAX = 60
OPTION_DELTA_MIN = 0.25
OPTION_DELTA_MAX = 0.65
EARNINGS_LOCK_DAYS = 2
IMPLICIT_STOP_FRAC = 0.05  # fallback when caller doesn't provide a stop

# Naked-call stress multiplier — applied to (stop − entry) so R1 sees
# 1.5× the per-share loss the literal stop implies. Bounds the otherwise-
# unbounded theoretical risk of naked short calls against gap-up moves
# that blow through the stop overnight.
NAKED_CALL_STRESS_MULT = 1.5

# Minimum reward/risk ratio for any opening LONG trade. SHORT premium
# is exempt (see TraderProposal._validate_geometry). Mirrored in
# llm.schemas.MIN_RISK_REWARD — both must move together if changed.
MIN_RISK_REWARD = 1.3

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
    # Exposure override (area A). A defined-risk combo is journaled as one OPT
    # row with entry_price=net_credit (for P&L), but its true capital-at-risk
    # for R3/R4 is max_loss = (width − net_credit) × 100 × contracts. When set,
    # `notional` returns this instead of the credit-based product so a combo
    # counts at its real exposure in opening-rule aggregation.
    risk_notional: float | None = None

    @property
    def notional(self) -> float:
        """Dollar exposure. Options multiplied by contract-size 100, unless an
        explicit risk_notional override is supplied (defined-risk combos)."""
        if self.risk_notional is not None:
            return self.risk_notional
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
    target: float | None = None      # optional; required for R7 R:R check
    strategy_label: str | None = None
    sector: str | None = None        # resolved from sectors.csv by the hook
    # options-only enrichment (may be None if snapshot not fetched)
    delta: float | None = None
    dte: int | None = None
    # earnings gate — caller supplies trading-days-to-next-earnings if known
    earnings_dte: int | None = None
    # Short-options enrichment — extracted from OCC option_symbol by the hook
    # (right ∈ {"C", "P"}, strike in dollars).
    right: Literal["C", "P"] | None = None
    strike: float | None = None
    # "open" = entering a new position (long buy OR short sell).
    # "close" = unwinding an existing position (sell long / buy back short).
    # The hook computes this by checking the journal for a matching OPEN row.
    intent: Literal["open", "close"] = "open"

    @property
    def is_opening_option_short(self) -> bool:
        """Selling premium to OPEN a new short position (CSP / naked call)."""
        return (
            self.asset_type == "OPT"
            and self.side == "SELL"
            and self.intent == "open"
        )

    @property
    def notional(self) -> float:
        if self.asset_type == "OPT":
            return self.qty * 100.0 * self.entry_price
        return self.qty * self.entry_price

    @property
    def max_loss(self) -> float:
        """Worst-case dollar loss on this trade, used for R1.

        Routing:
          * STK LONG: |entry − stop| × qty  (fallback to 5% implicit stop)
          * OPT LONG: full premium paid (capped at debit)
          * OPT SHORT_PUT: strike × 100 × qty − premium_collected
                           (CSP collateral exposure on assignment)
          * OPT SHORT_CALL: NAKED_CALL_STRESS_MULT × |stop − entry| × qty × 100
                           (stop distance with stress buffer for gap risk)
        """
        if self.asset_type == "OPT":
            if self.side == "SELL":
                premium_collected = self.qty * 100.0 * self.entry_price
                if self.right == "P":
                    # CSP: worst case is full assignment at strike.
                    strike_dollar = (self.strike or 0.0) * 100.0 * self.qty
                    return max(0.0, strike_dollar - premium_collected)
                # Naked call: stress-buffered stop distance × qty × 100.
                if self.stop is not None:
                    return (
                        NAKED_CALL_STRESS_MULT
                        * abs(self.stop - self.entry_price)
                        * self.qty
                        * 100.0
                    )
                # No stop on a naked call — return a huge number so R1
                # forces the hook to block it (also R5c blocks separately).
                return float("inf")
            # Long premium (buy options).
            return self.qty * 100.0 * self.entry_price
        # STK:
        if self.stop is not None:
            return abs(self.entry_price - self.stop) * self.qty
        # Fallback: pretend stop is 5% away from entry — a deliberate under-estimate
        # is worse than refusing every stopless trade, so we warn + compute.
        return IMPLICIT_STOP_FRAC * self.entry_price * self.qty


@dataclass(frozen=True)
class SizingContext:
    """What sizing needs from the rest of the world at decision time.

    `cash` is the unencumbered cash available for new collateral
    (needed by R5b for cash-secured-put validation). Defaults to
    equity (assumes 100% cash) when unknown so R5b doesn't false-block
    when the hook can't fetch live cash.

    `sector_lookup_available` lets the hook tell us "I tried to look up sectors
    but sectors.csv is missing"; R4 then degrades to a warning rather than
    silently passing.
    """
    equity: float
    cash: float | None = None
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

    # `intent` distinguishes opens from closes; closing orders reduce risk
    # (selling a long, buying back a short) and must skip the opening-only
    # rules below (R1, R2, R3, R5 notional caps).
    is_opening = proposed.intent == "open"

    # ---- R1 single-trade risk ----
    if proposed.asset_type == "STK" and proposed.stop is None:
        violations.append(SizingViolation(
            R_STOP_MISSING,
            f"stock order for {proposed.ticker} submitted without stop; "
            f"R1 estimated using implicit {IMPLICIT_STOP_FRAC*100:.0f}% stop",
            "warn",
        ))
    r1_budget = MAX_SINGLE_RISK_PCT * ctx.equity
    if is_opening and proposed.max_loss > r1_budget + EPS:
        violations.append(SizingViolation(
            R1,
            f"max loss ${proposed.max_loss:,.2f} exceeds "
            f"{MAX_SINGLE_RISK_PCT*100:.0f}% of equity (${r1_budget:,.2f})",
            "block",
        ))

    # ---- R2 concurrent open count ----
    if is_opening and len(ctx.opens) >= MAX_CONCURRENT_OPENS:
        violations.append(SizingViolation(
            R2,
            f"already {len(ctx.opens)} open positions; "
            f"cap is {MAX_CONCURRENT_OPENS}",
            "block",
        ))

    # ---- R3 per-ticker exposure ----
    # Closes are fully exempt: existing notionals are valued at entry while
    # equity is live, so a drawdown can push existing/equity over the cap —
    # and the close is the order that reduces it.
    ticker_u = proposed.ticker.upper()
    if is_opening:
        existing = sum(p.notional for p in ctx.opens if p.ticker.upper() == ticker_u)
        total_exposure = existing + proposed.notional
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
        # R5: long premium + short premium (CSP + naked call) both allowed.
        # Side must be BUY (opening a long) or SELL (opening a short).
        if proposed.side not in ("BUY", "SELL"):
            violations.append(SizingViolation(
                R5,
                f"options: side must be BUY or SELL; got {proposed.side!r}",
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

    # ---- R5b cash-secured put collateral ----
    # Short puts must be fully cash-collateralized so worst-case assignment
    # doesn't blow the account. Collateral need = strike × 100 × qty minus
    # the premium being collected (which lands in cash on fill).
    if proposed.is_opening_option_short and proposed.right == "P":
        if proposed.strike is None or proposed.strike <= 0:
            violations.append(SizingViolation(
                R5B,
                "short put missing strike — cannot verify CSP collateral",
                "block",
            ))
        else:
            cash_avail = ctx.cash if ctx.cash is not None else ctx.equity
            collateral_need = (
                proposed.strike * 100.0 * proposed.qty - proposed.notional
            )
            if collateral_need > cash_avail + EPS:
                violations.append(SizingViolation(
                    R5B,
                    f"CSP collateral ${collateral_need:,.2f} exceeds "
                    f"available cash ${cash_avail:,.2f} "
                    f"(strike={proposed.strike}, qty={proposed.qty}, "
                    f"premium=${proposed.notional:,.2f})",
                    "block",
                ))

    # ---- R5c naked-call must have explicit stop ----
    # Naked short calls have theoretically unbounded risk; the only thing
    # bounding it in practice is a hard stop + the NAKED_CALL_STRESS_MULT
    # buffer R1 applies. A naked call without a stop is uncapped risk
    # and must be refused outright.
    if proposed.is_opening_option_short and proposed.right == "C":
        if proposed.stop is None or proposed.stop <= proposed.entry_price:
            violations.append(SizingViolation(
                R5C,
                f"naked short call requires explicit stop > entry; "
                f"got stop={proposed.stop}, entry={proposed.entry_price}",
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

    # ---- R7 risk:reward floor (LONG opens only) ----
    # SHORT premium opens are exempt — see TraderProposal._validate_geometry
    # for rationale (R:R structurally capped on premium-selling trades).
    # Mirrors the pydantic validator. target is optional on ProposedTrade
    # because the hook layer may not know it; in that case we warn rather
    # than block.
    if is_opening and not proposed.is_opening_option_short:
        # Per-spec R:R floor (Phase 2 item 9): when the strategy_label maps
        # to a declared StrategySpec, the spec's floor governs. Specs may
        # only TIGHTEN — max() keeps the global 1.3 as the absolute minimum
        # for unmapped/legacy labels. Lazy + guarded import: this code runs
        # inside the pretool hook and the moomoo MCP guard, so a registry
        # bug must degrade to the global floor, never crash validation.
        rr_floor = MIN_RISK_REWARD
        rr_floor_origin = "global floor"
        try:
            from trading_agent.strategy_specs import spec_for_label
            spec = spec_for_label(proposed.strategy_label)
            if (spec is not None and spec.min_risk_reward is not None
                    and spec.min_risk_reward > rr_floor):
                rr_floor = spec.min_risk_reward
                rr_floor_origin = f"spec '{spec.name}'"
        except Exception:
            pass  # registry unavailable — fall back to the global floor

        if proposed.target is None:
            violations.append(SizingViolation(
                R_TARGET_MISSING,
                f"opening trade for {proposed.ticker} has no target; "
                f"R7 R:R floor not enforced",
                "warn",
            ))
        elif proposed.stop is not None:
            risk = abs(proposed.entry_price - proposed.stop)
            reward = abs(proposed.target - proposed.entry_price)
            if risk < 0.01:
                violations.append(SizingViolation(
                    R7,
                    f"stop ({proposed.stop}) too close to entry "
                    f"({proposed.entry_price}); risk=${risk:.4f}",
                    "block",
                ))
            elif reward < 0.01:
                violations.append(SizingViolation(
                    R7,
                    f"target ({proposed.target}) too close to entry "
                    f"({proposed.entry_price}); reward=${reward:.4f}",
                    "block",
                ))
            else:
                rr = reward / risk
                if rr < rr_floor:
                    violations.append(SizingViolation(
                        R7,
                        f"R:R {rr:.2f} below floor {rr_floor} "
                        f"({rr_floor_origin}) "
                        f"(risk=${risk:.2f}, reward=${reward:.2f}). "
                        f"Tighten stop OR widen target.",
                        "block",
                    ))

    return violations


# ---------------------------------------------------------------------------
# Multi-leg defined-risk verticals (R5e) — area A
# ---------------------------------------------------------------------------
#
# The single-leg SELL-to-open hard block (order_guard.R_short_option_open_blocked)
# stays in force forever: a legged-in short evaluated ALONE is a naked short.
# A DEFINED-RISK vertical is different — the long leg caps the short leg's loss,
# so the position's true risk is a finite (width - net_credit). check_combo
# PROVES that property before any sizing, then sizes the combo as ONE position
# off its max_loss. Only CREDIT verticals are unblocked here; naked shorts and
# debit structures stay blocked.


@dataclass(frozen=True)
class ComboLeg:
    """One leg of a two-leg vertical. `price` is the per-contract premium."""
    option_symbol: str
    side: Literal["BUY", "SELL"]
    contracts: float
    price: float
    right: Literal["C", "P"]
    strike: float
    dte: int | None = None
    delta: float | None = None


@dataclass(frozen=True)
class ProposedCombo:
    """A two-leg defined-risk credit vertical to open atomically."""
    ticker: str
    legs: tuple[ComboLeg, ...]
    strategy_label: str | None = None
    sector: str | None = None
    intent: Literal["open"] = "open"

    @property
    def short_leg(self) -> ComboLeg | None:
        shorts = [leg for leg in self.legs if leg.side == "SELL"]
        return shorts[0] if len(shorts) == 1 else None

    @property
    def long_leg(self) -> ComboLeg | None:
        longs = [leg for leg in self.legs if leg.side == "BUY"]
        return longs[0] if len(longs) == 1 else None

    @property
    def contracts(self) -> float:
        return self.legs[0].contracts if self.legs else 0.0

    @property
    def net_credit(self) -> float:
        """Per-contract net credit = short premium − long premium."""
        s, lo = self.short_leg, self.long_leg
        return (s.price - lo.price) if (s and lo) else 0.0

    @property
    def width(self) -> float:
        s, lo = self.short_leg, self.long_leg
        return abs(s.strike - lo.strike) if (s and lo) else 0.0

    @property
    def max_loss(self) -> float:
        """Defined risk = (width − net_credit) × 100 × contracts."""
        return max(0.0, self.width - self.net_credit) * 100.0 * self.contracts

    @property
    def notional(self) -> float:
        # A defined-risk vertical's true exposure (for R3) is its max_loss —
        # capital at risk — not the gross premium of either leg.
        return self.max_loss


def check_combo(ctx: SizingContext, combo: ProposedCombo) -> list[SizingViolation]:
    """Validate a two-leg defined-risk credit vertical, then size it as ONE
    position. Returns all violations; caller blocks on any severity=='block'.

    Defined-risk invariants (R5e) are checked FIRST. If ANY fails the structure
    is not a provable vertical and is blocked outright — the per-leg
    short-open path (R5b/R5c) is intentionally NOT consulted, because the
    width-minus-credit math supersedes it for a bounded short leg and a
    non-vertical must never fall through to a naked-short evaluation.
    """
    legs = combo.legs
    if len(legs) != 2:
        return [SizingViolation(
            R5E, f"combo must have exactly 2 legs, got {len(legs)}", "block")]
    a, b = legs
    s, lo = combo.short_leg, combo.long_leg
    if s is None or lo is None:
        return [SizingViolation(
            R5E, "combo must have exactly one BUY and one SELL leg", "block")]
    if abs(a.contracts - b.contracts) > EPS or combo.contracts <= 0:
        return [SizingViolation(
            R5E, f"combo legs need equal positive contracts; got "
            f"{a.contracts} and {b.contracts}", "block")]
    if a.dte is not None and b.dte is not None and a.dte != b.dte:
        return [SizingViolation(
            R5E, f"combo legs must share one expiry; got dte {a.dte} and "
            f"{b.dte} (calendars not supported)", "block")]
    if a.right != b.right:
        return [SizingViolation(
            R5E, f"combo legs must be the same right (both C or both P); got "
            f"{a.right} and {b.right}", "block")]
    # Long leg must CAP the short leg → defined risk.
    capped = (lo.strike < s.strike) if a.right == "P" else (lo.strike > s.strike)
    if not capped:
        return [SizingViolation(
            R5E, f"long leg does not cap the short (right={a.right}, "
            f"short_strike={s.strike}, long_strike={lo.strike}) — not a "
            f"defined-risk vertical", "block")]
    net_credit, width = combo.net_credit, combo.width
    if net_credit <= 0:
        return [SizingViolation(
            R5E, f"combo is not a credit spread (net_credit ${net_credit:.2f} "
            f"<= 0); debit structures are blocked", "block")]
    if width <= net_credit + EPS:
        return [SizingViolation(
            R5E, f"combo max_loss <= 0 (width ${width:.2f} <= net_credit "
            f"${net_credit:.2f}) — mispriced or not defined-risk", "block")]

    # ---- size the combo as ONE position ----
    v: list[SizingViolation] = []
    max_loss = combo.max_loss
    is_opening = combo.intent == "open"

    r1_budget = MAX_SINGLE_RISK_PCT * ctx.equity
    if is_opening and max_loss > r1_budget + EPS:
        v.append(SizingViolation(
            R1, f"combo max loss ${max_loss:,.2f} exceeds "
            f"{MAX_SINGLE_RISK_PCT*100:.0f}% of equity (${r1_budget:,.2f})",
            "block"))

    if is_opening and len(ctx.opens) >= MAX_CONCURRENT_OPENS:
        v.append(SizingViolation(
            R2, f"already {len(ctx.opens)} open positions; cap is "
            f"{MAX_CONCURRENT_OPENS}", "block"))

    ticker_u = combo.ticker.upper()
    if is_opening:
        existing = sum(p.notional for p in ctx.opens if p.ticker.upper() == ticker_u)
        total = existing + combo.notional
        r3_cap = MAX_TICKER_EXPOSURE_PCT * ctx.equity
        if total > r3_cap + EPS:
            v.append(SizingViolation(
                R3, f"combined {ticker_u} exposure ${total:,.2f} exceeds "
                f"{MAX_TICKER_EXPOSURE_PCT*100:.0f}% of equity (${r3_cap:,.2f})",
                "block"))

    if is_opening:
        if combo.sector:
            same = [p for p in ctx.opens
                    if p.sector and p.sector == combo.sector
                    and p.ticker.upper() != ticker_u]
            if len(same) >= MAX_SAME_SECTOR_OPENS:
                v.append(SizingViolation(
                    R4, f"already {len(same)} open positions in sector "
                    f"{combo.sector!r}: {[p.ticker for p in same]}; cap is "
                    f"{MAX_SAME_SECTOR_OPENS}", "block"))
        elif not ctx.sector_lookup_available:
            v.append(SizingViolation(
                R_SECTOR_UNKNOWN,
                "sectors.csv missing — R4 not enforced for this combo", "warn"))
        else:
            v.append(SizingViolation(
                R_SECTOR_UNKNOWN, f"sector for {ticker_u} not in sectors.csv — "
                "R4 not enforced for this combo", "warn"))

    dte = legs[0].dte
    if is_opening and dte is not None and not (OPTION_DTE_MIN <= dte <= OPTION_DTE_MAX):
        v.append(SizingViolation(
            R5, f"combo DTE {dte} outside [{OPTION_DTE_MIN},{OPTION_DTE_MAX}]",
            "block"))

    return v


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
