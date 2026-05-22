"""Deterministic exit-rule executor.

Replaces the LLM exit_monitor on the hot path. Every intraday tick, for
every open position, ``hard_exit_decision`` consumes the ``ExitPlan``
that was stored at entry time and returns either an ``ExitDecision`` or
``None`` (= HOLD). No LLM calls. No network. Pure function.

Priority order (highest first):
  0. Regime kill switch (CRISIS → exit)
  1. Hard stop (mark <= hard_stop)
  2. DTE rules (options only)
       - DTE <= force_exit_at_dte         → EXIT_DTE_HARD
       - DTE <= 5 & |Δ| < threshold       → EXIT_DTE_OTM
       - DTE <= intrinsic-floor & |Δ| OK  → upgrade stop to intrinsic - buffer
  3. Hard target / scale-out ladder
  4. Trailing stop (after engagement, using mfe_so_far as high-water mark)
  5. Age (time_in_trade_max_days)

See docs/position_management.md for the rationale behind each threshold.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from trading_agent.llm.schemas import ExitPlan


# ---------------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExitDecision:
    """The single output of one position-tick evaluation."""

    action: Literal[
        "EXIT_REGIME",
        "EXIT_STOP",
        "EXIT_TARGET",
        "EXIT_TRAIL",
        "EXIT_DTE_HARD",
        "EXIT_DTE_OTM",
        "EXIT_INTRINSIC_FLOOR",
        "EXIT_AGE",
    ]
    exit_qty_factor: float            # 1.0 = full close, 0<x<1 = partial
    reason: str                       # one-line audit string

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "exit_qty_factor": float(self.exit_qty_factor),
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Option-symbol parsing helpers (OCC + moomoo prefix)
# ---------------------------------------------------------------------------


# moomoo format: "US.SPY260529C00742000" or "US.SPY260529C742000"
# OCC parts: <TICKER><YYMMDD><C|P><strike*1000 padded to 8 digits>
_OPT_RE = re.compile(r"^(?:US\.)?([A-Z]+)(\d{6})([CP])(\d{4,8})$")


def compute_dte_from_symbol(symbol: str, now_utc: datetime) -> int | None:
    """Days to expiry from an option symbol. Returns None for non-options."""
    if not symbol:
        return None
    m = _OPT_RE.match(symbol.upper())
    if not m:
        return None
    _ticker, ymd, _cp, _strike = m.groups()
    try:
        year = 2000 + int(ymd[:2])
        month = int(ymd[2:4])
        day = int(ymd[4:6])
        # Options expire at end of trading day; treat expiry as 23:59 UTC of
        # the expiry date so the DTE flips to 0 only after the calendar day
        # has passed.
        expiry = datetime(year, month, day, 23, 59, tzinfo=timezone.utc)
    except ValueError:
        return None
    delta = (expiry - now_utc).total_seconds() / 86400.0
    # Floor so 0.4 days reads as DTE=0 (last trading day) rather than 1.
    return max(0, int(delta))


def compute_intrinsic(symbol: str, underlying_mark: float | None) -> float | None:
    """Per-contract intrinsic value in dollars. None if not parseable.

    For calls: max(0, underlying - strike).
    For puts:  max(0, strike - underlying).
    """
    if not symbol or underlying_mark is None or underlying_mark <= 0:
        return None
    m = _OPT_RE.match(symbol.upper())
    if not m:
        return None
    _ticker, _ymd, cp, strike_str = m.groups()
    try:
        # OCC strike is strike-price * 1000 (so "00742000" = $742.000).
        strike = int(strike_str) / 1000.0
    except ValueError:
        return None
    if cp == "C":
        return max(0.0, underlying_mark - strike)
    return max(0.0, strike - underlying_mark)


# ---------------------------------------------------------------------------
# Plan loading + back-compat
# ---------------------------------------------------------------------------


def load_exit_plan(raw: dict | ExitPlan | None,
                   fallback_stop: float | None,
                   fallback_target: float | None) -> ExitPlan | None:
    """Build a usable ExitPlan from journal_trades.

    * ``raw`` (the JSONB column or already-parsed dict) wins.
    * Otherwise fall back to the legacy stop/target columns + plan defaults.
    * Returns None if neither path can produce a stop+target pair.
    """
    if isinstance(raw, ExitPlan):
        return raw
    if isinstance(raw, dict) and raw.get("hard_stop") and raw.get("hard_target"):
        try:
            return ExitPlan.model_validate(raw)
        except Exception:
            pass  # fall through to fallback
    if fallback_stop is None or fallback_target is None:
        return None
    if fallback_stop <= 0 or fallback_target <= 0:
        return None
    # Legacy plan — no scale-out, no trail, default DTE rules.
    return ExitPlan(
        hard_stop=float(fallback_stop),
        hard_target=float(fallback_target),
    )


# ---------------------------------------------------------------------------
# Main decision function
# ---------------------------------------------------------------------------


def _trail_floor(plan: ExitPlan, entry_price: float) -> float:
    """Resolve `never_below` to a concrete dollar floor for the trail."""
    if plan.trail_stop is None:
        return 0.0
    nb = plan.trail_stop.never_below
    if nb in ("break_even", "entry"):
        return entry_price
    if nb == "original_stop":
        return plan.hard_stop
    return 0.0


def hard_exit_decision(
    *,
    pos: dict,
    plan: ExitPlan,
    regime_label: str,
    now_utc: datetime,
    mfe_so_far: float | None,
    underlying_mark: float | None = None,
    scale_rungs_taken: int = 0,
) -> ExitDecision | None:
    """Return an exit decision for one position, or None to HOLD.

    Args:
        pos: position dict with keys: symbol, asset_type, entry_price, mark,
             qty, opened_at, delta.
        plan: the ExitPlan stored at entry time (see ExitPlan schema).
        regime_label: current HMM-derived regime label (e.g. "RANGE_LOW_VOL").
        now_utc: current UTC time (injectable for tests).
        mfe_so_far: latest Maximum Favorable Excursion in R-multiples
             (journal_trades.mfe_so_far). Used to derive the trail
             high-water mark. None = treat as no high-water info available.
        underlying_mark: spot price of the option's underlying (needed for
             intrinsic floor at DTE <= 5).
        scale_rungs_taken: how many scale-out rungs have already fired
             on this trade (so we don't fire the same rung twice).
    """
    mark = float(pos.get("mark") or 0.0)
    entry = float(pos.get("entry_price") or 0.0)
    asset_type = pos.get("asset_type") or "STK"
    symbol = pos.get("symbol") or ""
    is_short = plan.direction == "SHORT"

    # ---- P0 — Regime kill switch ----
    if regime_label in plan.regime_rules.exit_on_labels:
        return ExitDecision(
            action="EXIT_REGIME",
            exit_qty_factor=1.0,
            reason=f"regime={regime_label}",
        )

    # ---- P1 — Hard stop ----
    # LONG:  mark <= stop triggers (option price falling against us)
    # SHORT: mark >= stop triggers (option price rising against us)
    if mark > 0:
        stop_hit = (mark >= plan.hard_stop) if is_short else (mark <= plan.hard_stop)
        if stop_hit:
            cmp = ">=" if is_short else "<="
            return ExitDecision(
                action="EXIT_STOP",
                exit_qty_factor=1.0,
                reason=f"mark={mark:.2f} {cmp} hard_stop={plan.hard_stop:.2f}",
            )

    # ---- P2 — DTE rules (options only) ----
    if asset_type == "OPT":
        dte = compute_dte_from_symbol(symbol, now_utc)
        if dte is not None:
            delta_abs = abs(float(pos.get("delta") or 0.0))
            rules = plan.dte_rules

            if is_short:
                # Short premium near expiry: theta decay is YOUR friend.
                # Only hard rule is to avoid assignment / pin risk on the
                # final day. force_exit_at_dte is still respected (default
                # 2) — bump down to 1 in the plan if you want to ride it
                # closer to the wire.
                if dte <= rules.force_exit_at_dte:
                    return ExitDecision(
                        action="EXIT_DTE_HARD",
                        exit_qty_factor=1.0,
                        reason=(
                            f"DTE={dte} <= force_exit_at_dte="
                            f"{rules.force_exit_at_dte} (short assignment risk)"
                        ),
                    )
                # No OTM theta rule and no intrinsic floor for shorts —
                # the same conditions that hurt longs benefit shorts.
            else:
                # LONG option DTE ladder.
                # P2a — hard DTE floor (assignment risk)
                if dte <= rules.force_exit_at_dte:
                    return ExitDecision(
                        action="EXIT_DTE_HARD",
                        exit_qty_factor=1.0,
                        reason=f"DTE={dte} <= force_exit_at_dte={rules.force_exit_at_dte}",
                    )

                # P2b — OTM near expiry → theta vampire, force exit
                if (dte <= 5
                        and delta_abs < rules.force_exit_at_dte_5_if_delta_below):
                    return ExitDecision(
                        action="EXIT_DTE_OTM",
                        exit_qty_factor=1.0,
                        reason=(
                            f"DTE={dte}, |Δ|={delta_abs:.2f} < "
                            f"{rules.force_exit_at_dte_5_if_delta_below:.2f} "
                            f"(theta vampire window)"
                        ),
                    )

                # P2c — ITM/ATM near expiry → intrinsic floor stop instead of exit
                if dte <= rules.switch_to_intrinsic_floor_at_dte and delta_abs >= 0.40:
                    intrinsic = compute_intrinsic(symbol, underlying_mark)
                    if intrinsic is not None and intrinsic > 0:
                        floor = max(
                            plan.hard_stop,
                            intrinsic - rules.intrinsic_floor_buffer,
                        )
                        if mark <= floor:
                            return ExitDecision(
                                action="EXIT_INTRINSIC_FLOOR",
                                exit_qty_factor=1.0,
                                reason=(
                                    f"mark={mark:.2f} <= intrinsic_floor={floor:.2f} "
                                    f"(intrinsic={intrinsic:.2f}, DTE={dte}, |Δ|={delta_abs:.2f})"
                                ),
                            )

    # ---- P3 — Hard target ----
    # LONG:  mark >= target (price rose to take-profit)
    # SHORT: mark <= target (premium decayed to buyback level)
    target_hit = (mark <= plan.hard_target) if is_short else (mark >= plan.hard_target)
    if target_hit:
        cmp = "<=" if is_short else ">="
        return ExitDecision(
            action="EXIT_TARGET",
            exit_qty_factor=1.0,
            reason=f"mark={mark:.2f} {cmp} hard_target={plan.hard_target:.2f}",
        )

    # ---- P3b — Scale-out ladder ----
    # LONG: rung fires when mark RISES to at_mark
    # SHORT: rung fires when mark FALLS to at_mark
    for idx, rung in enumerate(plan.scale_out_ladder):
        if idx < scale_rungs_taken:
            continue
        rung_hit = (mark <= rung.at_mark) if is_short else (mark >= rung.at_mark)
        if rung_hit:
            cmp = "<=" if is_short else ">="
            return ExitDecision(
                action="EXIT_TARGET",
                exit_qty_factor=float(rung.exit_factor),
                reason=(
                    f"scale-out rung#{idx + 1}: mark={mark:.2f} {cmp} {rung.at_mark:.2f}, "
                    f"close {int(rung.exit_factor * 100)}%"
                ),
            )

    # ---- P4 — Trailing stop ----
    # LONG: trail tracks the HIGH-water mark; exit when mark falls back
    #       to high_water × (1 − distance_pct).
    # SHORT: trail tracks the LOW-water mark; exit when mark rises back
    #        to low_water × (1 + distance_pct). For shorts we use MAE
    #        instead of MFE (most-negative R = price moving in our favor).
    if plan.trail_stop is not None and entry > 0:
        r_unit = abs(entry - plan.hard_stop)
        if r_unit > 0:
            if is_short:
                # mae_so_far is most-negative R from entry's perspective —
                # but for a short, "negative R" means LONG-style loss = mark > entry.
                # The price moving FAVORABLY for a short = mark < entry, which
                # appears as MAE (most-adverse from a long's POV).
                # So: low_water = entry + mae × r_unit (mae is negative ≤ 0)
                low_water_R = pos.get("mae_so_far")
                if low_water_R is not None:
                    low_water = entry + float(low_water_R) * r_unit
                    # Engaged when mark has DROPPED to engage_at_mark
                    if low_water <= plan.trail_stop.engage_at_mark:
                        trail_dist = low_water * plan.trail_stop.distance_pct
                        # never_below for shorts means "never_ABOVE" the floor
                        floor = _trail_floor(plan, entry)
                        trailed_stop = min(low_water + trail_dist, floor)
                        if mark >= trailed_stop:
                            return ExitDecision(
                                action="EXIT_TRAIL",
                                exit_qty_factor=1.0,
                                reason=(
                                    f"trail: mark={mark:.2f} >= "
                                    f"trailed_stop={trailed_stop:.2f} "
                                    f"(low_water={low_water:.2f}, "
                                    f"dist={plan.trail_stop.distance_pct:.0%})"
                                ),
                            )
            elif mfe_so_far is not None:
                high_water = entry + float(mfe_so_far) * r_unit
                if high_water >= plan.trail_stop.engage_at_mark:
                    trail_dist = high_water * plan.trail_stop.distance_pct
                    trailed_stop = max(
                        high_water - trail_dist,
                        _trail_floor(plan, entry),
                    )
                    if mark <= trailed_stop:
                        return ExitDecision(
                            action="EXIT_TRAIL",
                            exit_qty_factor=1.0,
                            reason=(
                                f"trail: mark={mark:.2f} <= "
                                f"trailed_stop={trailed_stop:.2f} "
                                f"(high_water={high_water:.2f}, "
                                f"dist={plan.trail_stop.distance_pct:.0%})"
                            ),
                        )

    # ---- P5 — Max age ----
    opened_at = pos.get("opened_at")
    if opened_at is not None and plan.time_in_trade_max_days > 0:
        try:
            if isinstance(opened_at, str):
                opened_dt = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
            else:
                opened_dt = opened_at
            if opened_dt.tzinfo is None:
                opened_dt = opened_dt.replace(tzinfo=timezone.utc)
            age = now_utc - opened_dt
            if age >= timedelta(days=plan.time_in_trade_max_days):
                return ExitDecision(
                    action="EXIT_AGE",
                    exit_qty_factor=1.0,
                    reason=(
                        f"open {age.days}d >= max "
                        f"{plan.time_in_trade_max_days}d"
                    ),
                )
        except (TypeError, ValueError):
            pass  # malformed timestamp — fail safe (HOLD)

    return None  # HOLD
