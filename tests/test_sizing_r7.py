"""R7 + R5b + R5c — short-options sizing rules.

The pydantic validator on TraderProposal is the strict gate at synthesizer
time (tested in tests/llm/test_schemas.py); R7/R5b/R5c in sizing.py are
the same rules callable from any non-schema call path (manual rescue
order, future backtester, etc.) and visible to the pretool hook's audit log.
"""
from __future__ import annotations

from trading_agent.sizing import (
    MIN_RISK_REWARD,
    ProposedTrade,
    R5B,
    R5C,
    R7,
    R_TARGET_MISSING,
    SizingContext,
    blockers,
    check,
)


def _ctx(equity: float = 100_000.0) -> SizingContext:
    return SizingContext(equity=equity, opens=(), sector_lookup_available=True)


def _trade(stop: float | None = 8.0, target: float | None = 13.0) -> ProposedTrade:
    """Default: entry=10, qty=10. Risk=$2/sh × 10 = $20 (well under R1)."""
    return ProposedTrade(
        ticker="AAPL", asset_type="STK", side="BUY",
        qty=10, entry_price=10.0,
        stop=stop, target=target,
        strategy_label="trend",
    )


# ---------------------------------------------------------------------------
# R7 blocks low R:R
# ---------------------------------------------------------------------------


def test_r7_blocks_bad_rr():
    """entry=10, stop=5, target=13 → risk=5, reward=3, R:R 0.6 < 1.5 → block.

    Mirrors the SPY 742C audit: stop too tight, target too modest.
    """
    vs = check(_ctx(), _trade(stop=5.0, target=13.0))
    r7_blocks = [v for v in vs if v.rule == R7 and v.severity == "block"]
    assert len(r7_blocks) == 1
    assert "0.60" in r7_blocks[0].message


def test_r7_accepts_exactly_15():
    """R:R exactly 1.5 passes."""
    vs = check(_ctx(), _trade(stop=8.0, target=13.0))  # R:R = 1.5
    assert not any(v.rule == R7 for v in vs)


def test_r7_blocks_zero_risk_stop_at_entry():
    vs = check(_ctx(), _trade(stop=10.0, target=15.0))  # risk = 0
    r7_blocks = [v for v in vs if v.rule == R7 and v.severity == "block"]
    assert len(r7_blocks) == 1


def test_r7_blocks_zero_reward_target_at_entry():
    vs = check(_ctx(), _trade(stop=5.0, target=10.0))  # reward = 0
    r7_blocks = [v for v in vs if v.rule == R7 and v.severity == "block"]
    assert len(r7_blocks) == 1


# ---------------------------------------------------------------------------
# Warn-only when target is missing
# ---------------------------------------------------------------------------


def test_target_missing_warns_not_blocks():
    """Hook-layer ProposedTrade may not have target; warn rather than
    block (synthesizer validator catches it earlier upstream)."""
    vs = check(_ctx(), _trade(stop=8.0, target=None))
    warns = [v for v in vs if v.rule == R_TARGET_MISSING]
    assert len(warns) == 1
    assert warns[0].severity == "warn"
    # No R7 blocker should appear in this case
    assert not blockers([v for v in vs if v.rule == R7])


def test_close_intent_skips_r7():
    """Closing orders (intent='close') shouldn't be R7-checked — the
    R7 floor only governs new opens. (Pre-shorts the gate was side='BUY';
    now it's intent='open' so we can correctly classify SHORT opens.)"""
    t = ProposedTrade(
        ticker="AAPL", asset_type="STK", side="SELL",
        qty=10, entry_price=10.0, stop=9.0, target=10.5,  # bad R:R, but CLOSE
        intent="close",
    )
    vs = check(_ctx(), t)
    assert not any(v.rule == R7 for v in vs)


# ---------------------------------------------------------------------------
# Sanity: constant matches schema constant
# ---------------------------------------------------------------------------


def test_min_rr_constant_matches_schema():
    """sizing.py and llm/schemas.py both hold a MIN_RISK_REWARD constant;
    they must move together. This guards against drift."""
    from trading_agent.llm.schemas import MIN_RISK_REWARD as schema_min
    assert MIN_RISK_REWARD == schema_min


# ---------------------------------------------------------------------------
# R5b — Cash-secured-put collateral
# ---------------------------------------------------------------------------


def _csp_trade(strike=100.0, premium=2.0, qty=1):
    return ProposedTrade(
        ticker="AAPL", asset_type="OPT", side="SELL",
        qty=qty, entry_price=premium,
        stop=premium * 2,    # short stop is ABOVE entry
        target=premium * 0.2,  # short target is BELOW entry
        right="P", strike=strike,
        intent="open",
        dte=30, delta=-0.30,
    )


def test_r5b_passes_when_cash_sufficient():
    # 1 contract × 100 × $100 strike = $10k collateral; minus $200 premium → $9.8k.
    # ctx with $100k equity, $50k cash → plenty.
    ctx = SizingContext(equity=100_000, cash=50_000)
    vs = check(ctx, _csp_trade())
    assert not any(v.rule == R5B for v in vs)


def test_r5b_blocks_when_cash_insufficient():
    # Need $9,800 collateral, have only $5,000 cash → block
    ctx = SizingContext(equity=100_000, cash=5_000)
    vs = check(ctx, _csp_trade())
    r5b_blocks = [v for v in vs if v.rule == R5B]
    assert len(r5b_blocks) == 1
    assert r5b_blocks[0].severity == "block"
    assert "exceeds available cash" in r5b_blocks[0].message


def test_r5b_falls_back_to_equity_when_cash_unknown():
    """If broker doesn't expose cash, ctx.cash=None → treat equity as cash."""
    ctx = SizingContext(equity=100_000, cash=None)  # cash unknown
    vs = check(ctx, _csp_trade())
    # 9,800 < 100k equity → passes
    assert not any(v.rule == R5B for v in vs)


def test_r5b_skipped_on_buy_long_put():
    # Same put strike + premium but BUYING (long put) → no CSP rule
    t = ProposedTrade(
        ticker="AAPL", asset_type="OPT", side="BUY",
        qty=1, entry_price=2.0, stop=1.0, target=4.0,
        right="P", strike=100.0,
        intent="open", dte=30, delta=-0.30,
    )
    ctx = SizingContext(equity=100_000, cash=100)  # tiny cash
    vs = check(ctx, t)
    assert not any(v.rule == R5B for v in vs)


# ---------------------------------------------------------------------------
# R5c — Naked-call must have explicit stop > entry
# ---------------------------------------------------------------------------


def _naked_call(stop=None, premium=1.5, qty=1):
    return ProposedTrade(
        ticker="AAPL", asset_type="OPT", side="SELL",
        qty=qty, entry_price=premium,
        stop=stop,
        target=premium * 0.2,
        right="C", strike=150.0,
        intent="open", dte=30, delta=0.25,
    )


def test_r5c_blocks_naked_call_without_stop():
    ctx = SizingContext(equity=100_000, cash=100_000)
    vs = check(ctx, _naked_call(stop=None))
    r5c_blocks = [v for v in vs if v.rule == R5C]
    assert len(r5c_blocks) == 1
    assert r5c_blocks[0].severity == "block"


def test_r5c_blocks_naked_call_with_stop_below_entry():
    # premium $1.50, stop $1.00 — that's a LONG-direction stop, not SHORT
    ctx = SizingContext(equity=100_000, cash=100_000)
    vs = check(ctx, _naked_call(stop=1.0, premium=1.5))
    r5c_blocks = [v for v in vs if v.rule == R5C]
    assert len(r5c_blocks) == 1


def test_r5c_passes_with_valid_stop():
    # premium $1.50, stop $3.00 (200% loss) — proper SHORT stop
    ctx = SizingContext(equity=100_000, cash=100_000)
    vs = check(ctx, _naked_call(stop=3.0, premium=1.5))
    assert not any(v.rule == R5C for v in vs)


# ---------------------------------------------------------------------------
# R1 max_loss math for shorts
# ---------------------------------------------------------------------------


def test_csp_max_loss_uses_assignment_collateral():
    """CSP max loss = strike × 100 × qty - premium collected."""
    t = _csp_trade(strike=100.0, premium=2.0, qty=1)
    # Max loss if assigned: 100 × 100 - 200 = $9,800
    assert t.max_loss == 9_800.0


def test_naked_call_max_loss_uses_stress_buffer():
    """Naked call max loss = 1.5 × (stop - entry) × 100 × qty."""
    t = _naked_call(stop=3.0, premium=1.5, qty=1)
    # (3.0 - 1.5) × 100 × 1 × 1.5 = 225
    assert t.max_loss == 225.0


def test_naked_call_without_stop_max_loss_infinite():
    """No stop on naked call → max_loss = inf so R1 blocks even before R5c."""
    t = _naked_call(stop=None, premium=1.5)
    assert t.max_loss == float("inf")


def test_r7_skipped_for_short_opens():
    """SHORT premium opens are not gated by R7 (R:R structurally capped)."""
    t = _csp_trade(strike=100.0, premium=2.0, qty=1)
    # CSP R:R = reward $2 / risk $9800 ≈ 0.0002 — would fail R7 1.3
    ctx = SizingContext(equity=100_000, cash=50_000)
    vs = check(ctx, t)
    assert not any(v.rule == R7 for v in vs)


# ---------------------------------------------------------------------------
# Threshold raise sanity
# ---------------------------------------------------------------------------


def test_thresholds_raised():
    """Verify thresholds match the 2026-05-22 aggressive-mode raises."""
    from trading_agent.sizing import (
        MAX_CONCURRENT_OPENS,
        MAX_OPTION_NOTIONAL_PCT,
        MAX_SINGLE_RISK_PCT,
        MAX_TICKER_EXPOSURE_PCT,
        MIN_RISK_REWARD,
        OPTION_DELTA_MAX,
        OPTION_DELTA_MIN,
    )
    assert MAX_SINGLE_RISK_PCT == 0.025
    assert MAX_CONCURRENT_OPENS == 6
    assert MAX_TICKER_EXPOSURE_PCT == 0.12
    assert MAX_OPTION_NOTIONAL_PCT == 0.015
    assert OPTION_DELTA_MIN == 0.25
    assert OPTION_DELTA_MAX == 0.65
    assert MIN_RISK_REWARD == 1.3
