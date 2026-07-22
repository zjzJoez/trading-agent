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
    underlying: str | None = None,
) -> float:
    """Round-trip friction expressed in R units (fraction of one risk unit).

    Fees both sides per leg, plus half-spread both sides per leg at the
    typical option mark — journal prices are mids, so entry AND exit each
    cross half the quoted spread (the same convention the Phase-0 cost
    model charges on non-dealt prices). ``risk_per_unit`` is the dollar
    value of 1R for one contract(-set): premium × stop-fraction × 100 for
    single-leg long premium, (width − credit) × 100 for a vertical.

    ``underlying`` (optional): use that name's M1-0.3 per-underlying
    calibrated spread instead of the global percent-of-mark — index chains
    (SPY/QQQ) quote far tighter than the pooled single-name median.
    Uncalibrated/None falls back to the global figure unchanged.
    """
    if typical_premium <= 0 or risk_per_unit <= 0 or n_option_legs < 1:
        raise ValueError(
            f"invalid friction inputs: premium={typical_premium}, "
            f"risk={risk_per_unit}, legs={n_option_legs}"
        )
    fees = 2.0 * n_option_legs * fees_per_side(1, "OPT")
    spread = 2.0 * n_option_legs * half_spread_cost(
        typical_premium, 1, "OPT", underlying=underlying)
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

    Status semantics (enforced — see ``spec_trading_block`` below):
      * ``active``          — tradeable; sizing/guards apply normally.
      * ``shadow_only``     — RETIRED from real fills. Proposals mapped to
        this spec are never sized into an order; build_trade_proposal
        records them to the shadow book (shadow_proposals, final_action
        SHADOW_ONLY) for option-level counterfactual replay against
        option_chain_snapshots. CLOSES of existing positions are exempt
        (sizing gates the check on intent=='open') — retirement must
        never strand an open position.
      * ``pending_prereqs`` — declared but its blocking prerequisites are
        not yet green; NOT tradeable through any consumer. Unlike
        shadow_only there is no counterfactual book to feed yet, so a
        proposal reaching build_trade_proposal is finalized
        BLOCKED_SPEC_STATUS rather than SHADOW_ONLY.
      * ``blocked``         — legacy name, kept for compatibility; same
        non-tradeable enforcement as pending_prereqs. Historical note:
        before area A, ``blocked`` carried NO enforcement anywhere in the
        proposal→order path (its only consumer was the journal
        post-mortem's spec_status display) — the actual block was
        order_guard's single-leg SELL-to-open hard block. The statuses
        above exist precisely because 'blocked' the *name* promised an
        enforcement that never existed; new code should prefer the two
        explicit statuses.
    """

    name: str
    status: Literal["active", "shadow_only", "pending_prereqs", "blocked"]
    structure: str
    # Machine-readable shape of ``structure`` (free text is for humans).
    # spec_trading_block uses it to refuse a SINGLE-LEG order whose label
    # maps to a vertical spec: with convexity retired, relabeling a long
    # single-leg as an active credit spec would otherwise reopen the exact
    # structure the retirement closed (the retry loop feeds validation
    # errors back to the LLM, which makes label-shopping a live path,
    # not a hypothetical).
    structure_kind: Literal["single_leg", "vertical"] = "single_leg"
    entry_gates: Mapping[str, Any] = field(default_factory=dict)
    allowed_regimes: tuple[str, ...] = ()
    expectancy_profile: ExpectancyProfile | None = None
    min_trades_for_eval: int = 30
    falsification: str = ""

    @property
    def min_risk_reward(self) -> float | None:
        v = self.entry_gates.get("min_risk_reward")
        return float(v) if v is not None else None

    @property
    def is_tradeable(self) -> bool:
        """Only ``active`` specs may open real (paper) positions."""
        return self.status == "active"


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
        # RETIRED to shadow-only (sleeve 3, zero capital): operator-approved
        # 2026-07-20 — docs/REVIVAL_PLAN_2026-07-20.md ("Sleeve 3:
        # convexity_long_premium → shadow-only"). NO NEW real long-premium
        # fills; every in-band proposal is still recorded to the shadow
        # book so revival can be judged on option-level counterfactual P&L
        # (shadow proposals replayed against option_chain_snapshots +
        # execution_costs.py, promote-grade 97.5% gate, per the plan) —
        # never on the 20d underlying-direction proxy. Closing existing
        # long-premium positions remains allowed (intent=='close' is
        # exempt from the status gate).
        status="shadow_only",
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
        structure_kind="vertical",
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


def _credit_vertical_index_spec() -> StrategySpec:
    """Sleeve 1 — index credit verticals (docs/REVIVAL_PLAN_2026-07-20.md,
    M1-1 revised). Registered ``pending_prereqs``: it turns active ONLY when
    every M1-0 blocking prerequisite is green (combo-aware exit path with
    net-of-both-legs marking + atomic two-leg close, combo cost-honest paper
    fills, per-underlying spread calibration, and the M1-0.4 gate-feasibility
    snapshot replay). Until then no consumer may size or place it.

    EVERY number below is a PLACEHOLDER pending the M1-0.4 replay against
    option_chain_snapshots — the plan explicitly forbids shipping a second
    better-documented zero-order funnel: the replay must show qualifying
    verticals exist on >=60% of snapshot days in the allowed regimes, or the
    gates get retuned and the breakevens recomputed before activation.
    """
    return StrategySpec(
        name="credit_vertical_index_30_45",
        status="pending_prereqs",   # M1-0 all green → active (operator flip)
        structure=(
            "Short put vertical (BULL/RANGE); short call vertical (BEAR, on "
            "regime confirmation). 30-45 DTE, short leg |delta| 0.20-0.35, "
            "width $5-10, opened atomically via place_paper_option_combo "
            "(R5e defined-risk proof), managed deterministically by the "
            "hard executor: 50% profit-take or 21-DTE force-close, no "
            "mid-trade stop."
        ),
        structure_kind="vertical",
        entry_gates={
            # M1-1 revised gates — ALL PLACEHOLDERS pending M1-0.4 replay.
            "underlying_whitelist": ("SPY", "QQQ"),  # IWM pending M1-0.3
                                                     # measured combo friction
                                                     # (> 0.06R ⇒ stays out)
            "dte_range": (30, 45),
            "abs_delta_range": (0.20, 0.35),   # short leg; 0.30→0.35 per plan
            # credit >= width/4 (draft's width/3 was unsatisfiable in the
            # allowed low-IV regimes — structural zero-order funnel; final
            # value set by the M1-0.4 replay).
            "min_credit_frac_of_width": 0.25,
            "max_spread_pct_mid": 0.05,
            "news_veto_required": True,        # single LLM call pre-entry
            # NOTE: deliberately NO min_risk_reward — the plan deletes it as
            # mathematically redundant with (and contradictory to) the
            # credit gate; R7 exempts short-premium structures anyway.
        },
        allowed_regimes=("BULL_TREND", "RANGE_LOW_VOL"),
        # No ExpectancyProfile yet — the plan REJECTS the expiry-binary
        # breakeven formula for a 50%-PT / 21-DTE managed book (it is wrong
        # in both directions); the declared profile must come from the
        # M1-0.4 managed-payoff snapshot replay. Post-mortem spec_comparison
        # skips None profiles, so nothing grades against fake numbers.
        expectancy_profile=None,
        min_trades_for_eval=30,
        falsification=(
            "Three-tier contract (docs/REVIVAL_PLAN_2026-07-20.md M1-3; "
            "all thresholds PLACEHOLDER pending M1-0.4 replay): "
            "(1) n=30 mechanical health check — realized costs within 1.5x "
            "the M1-0.3 MEASURED calibration, no single loss > 1.05x the "
            "defined max_loss, observed WR vs the management-aware baseline "
            "P(50%-PT before 21-DTE | short delta) from snapshot replay "
            "(not hold-to-expiry 1-delta); the cost and max-loss checks are "
            "the real teeth at n=30 (binomial power ~37% at true WR 65%). "
            "(2) n=60 kill-gate — moratorium iff LB95(mean R) < -0.10R "
            "('proven harmful', not 'unproven'), CI via entry-week "
            "block-bootstrap (same-week SPY/QQQ verticals correlate >0.9). "
            "(3) promotion/scale-up is a SEPARATE gate — LB95(mean R) > 0 "
            "at 97.5% one-sided or replication across two non-overlapping "
            "windows; at the declared economics that needs n~150-400, so "
            "the account-level daily benchmark window vs 60/40 is the "
            "primary confirm/deny statistic."
        ),
    )


REGISTRY: dict[str, StrategySpec] = {
    s.name: s for s in (
        _convexity_spec(),
        _credit_put_spread_spec(),
        _credit_vertical_index_spec(),
    )
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
    # Sleeve 1 index verticals (M1-1) — pending_prereqs until M1-0 is green.
    ("credit_vertical", "credit_vertical_index_30_45"),
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
# Status enforcement — is this label allowed to OPEN a real position?
# ---------------------------------------------------------------------------


def spec_trading_block(
    strategy_label: str | None,
    *,
    asset_type: str | None = None,
) -> dict[str, Any] | None:
    """Return a structured block descriptor when ``strategy_label`` maps to a
    non-tradeable spec, else None.

    This is the single source of truth consulted by every opening path:
    sizing.check (single-leg, via the pretool hook AND the moomoo MCP
    guard), sizing.check_combo (verticals), and build_trade_proposal (which
    additionally records shadow_only proposals to the shadow book instead
    of silently dropping them).

    ONLY opening paths may call this — closes (intent=='close') must never
    be status-gated: retiring a strategy while stranding its open positions
    is strictly worse than either state.

    ``asset_type`` opts the caller into the STRUCTURE fallback, which fails
    closed for the structure, not the label: strategy_label is free-text
    LLM output (the trader-synthesizer has already invented off-prefix
    labels in production — 2026-07-08 CRNX 'momentum-continuation-ITM-call'),
    so an UNMAPPED (or absent) label must not be a retirement bypass.
    Single-leg SELL-to-open is hard-blocked at the order layer, which makes
    an unmapped-label single-leg OPT open EXACTLY the retired long-premium
    structure — it is therefore governed by the convexity spec's status.
    Unmapped STOCK opens stay governed by the global R1-R8 gates only, and
    the combo path (its own label families + the R5e structural proof)
    passes no asset_type, so both are unaffected.
    """
    spec = spec_for_label(strategy_label)
    if spec is None and (asset_type or "").strip().upper() == "OPT":
        fallback = REGISTRY["convexity_long_premium"]
        if fallback.is_tradeable:
            return None
        return {
            "spec": fallback.name,
            "status": fallback.status,
            "strategy_label": strategy_label,
            "unmapped_label": True,
            "message": (
                f"unmapped strategy_label {strategy_label!r} cannot "
                f"BUY-to-open a single-leg option while '{fallback.name}' is "
                f"{fallback.status} — single-leg SELL-to-open is hard-blocked, "
                f"so an unmapped-label option open is exactly the retired "
                f"long-premium structure; map the label or register a spec"
            ),
        }
    if (
        spec is not None
        and (asset_type or "").strip().upper() == "OPT"
        and spec.structure_kind != "single_leg"
    ):
        # Structure/shape mismatch: the caller is opening a SINGLE-LEG
        # option, but the label maps to a vertical spec. Whatever that
        # spec's status, it cannot authorize this shape — otherwise
        # relabeling a long single-leg as a credit spec reopens the exact
        # structure convexity's retirement closed.
        return {
            "spec": spec.name,
            "status": spec.status,
            "strategy_label": strategy_label,
            "structure_mismatch": True,
            "message": (
                f"strategy_label {strategy_label!r} maps to spec "
                f"'{spec.name}' whose structure is '{spec.structure_kind}' — "
                f"a single-leg option open cannot be authorized by a "
                f"vertical spec; use place_paper_option_combo for verticals, "
                f"or a single-leg label (governed by its own spec status)"
            ),
        }
    if spec is None or spec.is_tradeable:
        return None
    return {
        "spec": spec.name,
        "status": spec.status,
        "strategy_label": strategy_label,
        "message": (
            f"strategy_label {strategy_label!r} maps to spec "
            f"'{spec.name}' with status='{spec.status}' — not tradeable; "
            f"opening orders are refused"
            + (
                " (proposal recorded to the shadow book for counterfactual"
                " replay)" if spec.status == "shadow_only" else ""
            )
        ),
    }


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
    "spec_trading_block",
]
