"""Strategy spec registry — first-class, falsifiable strategy declarations.

Phase 2 item 9. Until now "strategy" existed only as free-text
strategy_label on proposals, so nothing was falsifiable: the post-mortem
graded every label against an UNCITED ~55% baseline win rate (the R7
comment in llm/schemas.py) and a "<40% WR ⇒ moratorium" rule — which
structurally kills a convexity book whose DECLARED shape is 30–45% WR
with fat winners, while letting a high-WR negative-expectancy book pass.
Each spec here states its expected shape up front (WR range, payoff
target, friction-adjusted breakeven) and the single observable result
that kills it. The journal post-mortem compares declared vs realized
(``spec_comparison``); sizing R7 enforces the spec's R:R floor per label.

Breakeven win rates are COMPUTED from the spec's min_risk_reward plus the
execution-cost model's friction (execution_costs.py) — never hard-coded —
so recalibrating data/execution_costs.json moves the declared breakevens
with it. This is the same Phase-0 cost honesty that moved the global
R7-1.3 breakeven from 43.5% gross to 50–57% net.

docs/STRATEGY_SPECS.md mirrors this module; THIS module is canonical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from trading_agent.execution_costs import fees_per_side, half_spread_cost

# ---------------------------------------------------------------------------
# Breakeven arithmetic — functions, not constants
# ---------------------------------------------------------------------------


def breakeven_wr_gross(min_rr: float) -> float:
    """Win rate where ``p·rr − (1−p)·1 = 0`` → ``1/(1+rr)``. No friction.

    Sanity anchor: at the global R7 floor 1.3 this is 1/2.3 ≈ 43.5%, the
    figure cited in the execution_costs module docstring.
    """
    if min_rr <= 0:
        raise ValueError(f"min_rr must be > 0, got {min_rr}")
    return 1.0 / (1.0 + min_rr)


def round_trip_friction_r(
    *,
    typical_premium: float,
    risk_per_unit: float,
    n_option_legs: int = 1,
) -> float:
    """Round-trip friction expressed in R units (fraction of one risk unit).

    Fees both sides per leg, plus half-spread both sides per leg at the
    typical option mark — journal prices are mids, so entry AND exit each
    cross half the quoted spread (the same convention the Phase-0 cost
    model charges on non-dealt prices). ``risk_per_unit`` is the dollar
    value of 1R for one contract(-set): premium × stop-fraction × 100 for
    single-leg long premium, (width − credit) × 100 for a vertical.
    """
    if typical_premium <= 0 or risk_per_unit <= 0 or n_option_legs < 1:
        raise ValueError(
            f"invalid friction inputs: premium={typical_premium}, "
            f"risk={risk_per_unit}, legs={n_option_legs}"
        )
    fees = 2.0 * n_option_legs * fees_per_side(1, "OPT")
    spread = 2.0 * n_option_legs * half_spread_cost(typical_premium, 1, "OPT")
    return (fees + spread) / risk_per_unit


def breakeven_wr_net(min_rr: float, friction_r: float) -> float:
    """Breakeven WR when every trade also pays ``friction_r`` R of friction.

    A win nets ``rr − f`` R, a loss nets ``−(1 + f)`` R; solving
    ``p(rr − f) = (1 − p)(1 + f)`` gives ``p = (1 + f)/(1 + rr)``.
    Strictly above ``breakeven_wr_gross`` for any f > 0 — friction only
    ever raises the bar. Anchor: rr 1.3 with the system's typical ~0.18R
    friction → 1.18/2.3 ≈ 51%, inside the 50–57% band the Phase-0 cost
    audit measured.
    """
    if friction_r < 0:
        raise ValueError(f"friction_r must be >= 0, got {friction_r}")
    return (1.0 + friction_r) / (1.0 + min_rr)


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectancyProfile:
    """The declared economic shape of a strategy.

    ``expected_wr_range`` is the envelope the strategy is SUPPOSED to live
    in — being below it is failure, but being above it is also suspicious
    (the structure isn't doing what was declared). ``breakeven_wr_*`` are
    computed via the functions above, never typed in by hand.
    """

    expected_wr_range: tuple[float, float]
    avg_win_to_avg_loss_target: float
    breakeven_wr_gross: float
    breakeven_wr_net: float


@dataclass(frozen=True)
class StrategySpec:
    """One falsifiable strategy declaration.

    ``falsification`` is the contract: the observable result that kills
    the strategy. If a spec can't state one, it isn't a strategy — it's
    a vibe.
    """

    name: str
    status: Literal["active", "blocked"]
    structure: str
    entry_gates: Mapping[str, Any] = field(default_factory=dict)
    allowed_regimes: tuple[str, ...] = ()
    expectancy_profile: ExpectancyProfile | None = None
    min_trades_for_eval: int = 30
    falsification: str = ""

    @property
    def min_risk_reward(self) -> float | None:
        v = self.entry_gates.get("min_risk_reward")
        return float(v) if v is not None else None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Friction inputs for single-leg long premium: $2.00/contract is the middle
# of what the system actually buys (journal fills cluster $1.5–$3), and the
# standard exit geometry stops at −50% of premium (see execution_costs
# docstring) — so 1R = 0.5 × premium × 100 per contract.
_TYPICAL_PREMIUM = 2.00
_PREMIUM_STOP_FRAC = 0.50


def _convexity_spec() -> StrategySpec:
    min_rr = 2.0
    risk_per_unit = _PREMIUM_STOP_FRAC * _TYPICAL_PREMIUM * 100.0
    friction_r = round_trip_friction_r(
        typical_premium=_TYPICAL_PREMIUM,
        risk_per_unit=risk_per_unit,
        n_option_legs=1,
    )
    return StrategySpec(
        name="convexity_long_premium",
        status="active",
        structure=(
            "Single-leg long premium (calls or puts), debit-defined risk. "
            "The edge lives in the tail of the winners, not the hit rate — "
            "losing more often than winning is the declared shape."
        ),
        entry_gates={
            # Tighter than the R5 policy band (14–60 DTE, 0.25–0.65 |delta|):
            # 21–45 DTE keeps theta bleed survivable while staying clear of
            # the gamma cliff; 0.30–0.55 |delta| avoids both lottery tickets
            # and stock-replacement deltas.
            "dte_range": (21, 45),
            "abs_delta_range": (0.30, 0.55),
            # Deliberately ABOVE the global R7 1.3 floor: at a declared
            # 30–45% WR, 1.3:1 winners cannot pay for the losers — the
            # re-specced convexity track demands 2:1 minimum.
            "min_risk_reward": min_rr,
            "catalysts_required": False,
            # RANGE_LOW_VOL long premium needs an explicit catalyst —
            # regime/gates.py blocks with no_catalyst_in_range_regime.
            "catalyst_required_in_regimes": ("RANGE_LOW_VOL",),
        },
        allowed_regimes=("BULL_TREND", "RANGE_LOW_VOL"),
        expectancy_profile=ExpectancyProfile(
            expected_wr_range=(0.30, 0.45),
            avg_win_to_avg_loss_target=2.5,
            breakeven_wr_gross=breakeven_wr_gross(min_rr),
            breakeven_wr_net=breakeven_wr_net(min_rr, friction_r),
        ),
        min_trades_for_eval=30,
        falsification="LB95(mean R) < 0 after 30 closed trades",
    )


def _credit_put_spread_spec() -> StrategySpec:
    # Defined-risk vertical: reward = credit, risk = width − credit. At
    # short-leg |delta| 0.20–0.30 a vertical collects ~28–35% of width,
    # i.e. R:R ≈ 0.40 — structurally capped, which is WHY R7 exempts
    # premium-selling structures from the long floor.
    min_rr = 0.40
    width = 5.0
    credit = width * min_rr / (1.0 + min_rr)
    risk_per_unit = (width - credit) * 100.0
    friction_r = round_trip_friction_r(
        typical_premium=1.50,   # average per-leg mark on a 0.25Δ vertical
        risk_per_unit=risk_per_unit,
        n_option_legs=2,        # two contracts per side, fees+spread on both
    )
    return StrategySpec(
        name="credit_put_spread_30_45",
        # ACTIVE (area A): the atomic multi-leg path landed. A defined-risk
        # vertical now opens via place_paper_option_combo, which sizes it as
        # ONE position off max_loss = (width − net_credit) and is governed by
        # R5e (defined-risk proof) + R1-R5/R5d. The single-leg SELL-to-open
        # hard block (R_short_option_open_blocked) remains in force — only
        # provable verticals are unblocked, never a legged-in naked short.
        status="active",
        structure=(
            "Short put vertical, 30–45 DTE: sell a 0.20–0.30 |delta| put, "
            "buy a further-OTM put; max loss defined by the width."
        ),
        entry_gates={
            "dte_range": (30, 45),
            "abs_delta_range": (0.20, 0.30),   # short leg
            "min_risk_reward": min_rr,
            "catalysts_required": False,
            "width_defined_risk": True,
        },
        allowed_regimes=("BULL_TREND", "RANGE_LOW_VOL"),
        expectancy_profile=ExpectancyProfile(
            expected_wr_range=(0.70, 0.80),
            # Expectancy stays positive only if avg loss < ~2.5× avg win
            # net of friction (1/0.40). Credit spreads die by letting the
            # rare max-loss trades run past the width math, not by WR.
            avg_win_to_avg_loss_target=min_rr,
            breakeven_wr_gross=breakeven_wr_gross(min_rr),
            breakeven_wr_net=breakeven_wr_net(min_rr, friction_r),
        ),
        min_trades_for_eval=30,
        falsification=(
            "LB95(mean R) < 0 after 30 closed trades, or realized WR below "
            "breakeven_wr_net over the same window"
        ),
    )


REGISTRY: dict[str, StrategySpec] = {
    s.name: s for s in (_convexity_spec(), _credit_put_spread_spec())
}


# Free-text strategy_label prefixes → spec name. The journal's historical
# labels ("directional_long_call", "pullback_reversal", "breakout_squeeze",
# "earnings_iv_drop_*") are all single-leg long premium, so they are graded
# against the convexity spec. Anything else is legacy and maps to None —
# we don't grade a trade against a spec it never declared.
_LABEL_PREFIX_TO_SPEC: tuple[tuple[str, str], ...] = (
    ("directional_", "convexity_long_premium"),
    ("earnings_iv_drop", "convexity_long_premium"),
    ("pullback_", "convexity_long_premium"),
    ("breakout_", "convexity_long_premium"),
    # Defined-risk credit verticals (area A) — placed via the combo path.
    ("credit_put_spread", "credit_put_spread_30_45"),
)


def spec_for_label(label: str | None) -> StrategySpec | None:
    """Map a free-text strategy_label to its governing StrategySpec.

    Exact spec names resolve directly; the prefix table covers the
    historical free-text families; unknown labels return None (legacy).
    """
    if not label:
        return None
    lbl = label.strip().lower()
    if lbl in REGISTRY:
        return REGISTRY[lbl]
    for prefix, name in _LABEL_PREFIX_TO_SPEC:
        if lbl.startswith(prefix):
            return REGISTRY[name]
    return None


# ---------------------------------------------------------------------------
# Schema-level band enforcement — proposal validation, BEFORE sizing
# ---------------------------------------------------------------------------


def spec_band_violations(
    *,
    strategy_label: str | None,
    asset_type: str | None,
    option_dte: int | None,
    option_delta: float | None,
) -> list[dict[str, Any]]:
    """DTE/delta bands an OPT proposal must satisfy at PROPOSAL time.

    Mapped labels enforce their spec's declared ``dte_range`` /
    ``abs_delta_range``, intersected with the global R5 band — a spec can
    only TIGHTEN the policy gates, never relax them (R5's 14–60 DTE /
    0.25–0.65 |delta| remain the outer bound; convexity's 21–45 DTE /
    0.30–0.55 |delta| is the binding inner band for mapped labels).

    UNMAPPED-LABEL POLICY: an unmapped/legacy strategy_label is NOT a free
    pass. An OPT proposal whose label maps to no spec falls back to the
    global R5 band at this same layer, so the 2026-07-08 CRNX shape
    (label 'momentum-continuation-ITM-call', delta 0.815) dies at proposal
    validation regardless of what the label says.

    A missing dte/delta skips only that band — matching sizing's R5
    None-semantics; sizing remains the enforcement of record downstream.
    Non-OPT proposals have no option bands → always [].
    """
    if (asset_type or "").upper() != "OPT":
        return []
    # Lazy import mirrors sizing's own lazy import of this module — keeps
    # the two policy modules free of an import cycle.
    from trading_agent.sizing import (
        OPTION_DELTA_MAX,
        OPTION_DELTA_MIN,
        OPTION_DTE_MAX,
        OPTION_DTE_MIN,
    )

    dte_lo, dte_hi = float(OPTION_DTE_MIN), float(OPTION_DTE_MAX)
    delta_lo, delta_hi = float(OPTION_DELTA_MIN), float(OPTION_DELTA_MAX)
    spec = spec_for_label(strategy_label)
    source = "global_r5"
    if spec is not None:
        source = spec.name
        spec_dte = spec.entry_gates.get("dte_range")
        if spec_dte is not None:
            dte_lo = max(dte_lo, float(spec_dte[0]))
            dte_hi = min(dte_hi, float(spec_dte[1]))
        spec_delta = spec.entry_gates.get("abs_delta_range")
        if spec_delta is not None:
            delta_lo = max(delta_lo, float(spec_delta[0]))
            delta_hi = min(delta_hi, float(spec_delta[1]))

    violations: list[dict[str, Any]] = []
    if option_dte is not None and not (dte_lo <= int(option_dte) <= dte_hi):
        violations.append({
            "band": "dte_range",
            "spec": source,
            "bounds": [dte_lo, dte_hi],
            "value": int(option_dte),
            "message": (
                f"option_dte {int(option_dte)} outside {source} band "
                f"[{dte_lo:g},{dte_hi:g}]"
            ),
        })
    if option_delta is not None:
        abs_d = abs(float(option_delta))
        if not (delta_lo <= abs_d <= delta_hi):
            violations.append({
                "band": "abs_delta_range",
                "spec": source,
                "bounds": [delta_lo, delta_hi],
                "value": abs_d,
                "message": (
                    f"|option_delta| {abs_d:.3f} outside {source} band "
                    f"[{delta_lo:g},{delta_hi:g}]"
                ),
            })
    return violations


__all__ = [
    "ExpectancyProfile",
    "REGISTRY",
    "StrategySpec",
    "breakeven_wr_gross",
    "breakeven_wr_net",
    "round_trip_friction_r",
    "spec_band_violations",
    "spec_for_label",
]
