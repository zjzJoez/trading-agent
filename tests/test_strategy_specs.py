"""Item 9 — strategy spec registry: integrity, breakeven arithmetic, label
mapping, the per-spec R7 floor in sizing, and the journal's declared-vs-
realized spec_comparison block."""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from unittest.mock import patch

import pytest

from trading_agent.execution_costs import fees_per_side, half_spread_cost
from trading_agent.strategy_specs import (
    REGISTRY,
    breakeven_wr_gross,
    breakeven_wr_net,
    round_trip_friction_r,
    spec_for_label,
)

# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------


def test_registry_has_exactly_the_two_declared_specs():
    assert set(REGISTRY) == {"convexity_long_premium", "credit_put_spread_30_45"}
    assert REGISTRY["convexity_long_premium"].status == "active"
    # area A: unblocked once the atomic multi-leg combo path (place_paper_option_
    # combo + R5e) landed; defined-risk verticals are now tradeable.
    assert REGISTRY["credit_put_spread_30_45"].status == "active"


def test_every_spec_breakeven_net_above_gross():
    """Friction only ever raises the bar — a spec whose net breakeven is not
    strictly above its gross breakeven means the cost model was bypassed."""
    for spec in REGISTRY.values():
        p = spec.expectancy_profile
        assert p is not None, spec.name
        assert p.breakeven_wr_net > p.breakeven_wr_gross, spec.name


def test_every_spec_declares_falsification_and_eval_n():
    for spec in REGISTRY.values():
        assert spec.falsification, spec.name
        assert spec.min_trades_for_eval >= 20, spec.name
        lo, hi = spec.expectancy_profile.expected_wr_range
        assert 0.0 < lo < hi < 1.0, spec.name


def test_convexity_min_rr_is_respecced_above_global():
    """The convexity track demands 2:1, NOT the global 1.3 — at a declared
    30-45% WR, 1.3:1 winners cannot pay for the losers."""
    from trading_agent.llm.schemas import MIN_RISK_REWARD
    spec = REGISTRY["convexity_long_premium"]
    assert spec.min_risk_reward == 2.0
    assert spec.min_risk_reward > MIN_RISK_REWARD


def test_convexity_declared_range_contains_profitable_territory():
    """The declared WR envelope must reach above the net breakeven —
    otherwise the spec declares a strategy that cannot make money."""
    p = REGISTRY["convexity_long_premium"].expectancy_profile
    lo, hi = p.expected_wr_range
    assert hi > p.breakeven_wr_net


# ---------------------------------------------------------------------------
# Breakeven arithmetic — functions, not constants
# ---------------------------------------------------------------------------


def test_breakeven_gross_matches_cited_anchor():
    # The R7 1.3 floor's 43.5% gross breakeven, cited in execution_costs.
    assert breakeven_wr_gross(1.3) == pytest.approx(1.0 / 2.3)
    assert breakeven_wr_gross(2.0) == pytest.approx(1.0 / 3.0)
    with pytest.raises(ValueError):
        breakeven_wr_gross(0.0)


def test_breakeven_net_formula_and_monotonicity():
    # p = (1 + f) / (1 + rr); zero friction degenerates to gross.
    assert breakeven_wr_net(1.3, 0.18) == pytest.approx(1.18 / 2.3)
    assert breakeven_wr_net(2.0, 0.0) == pytest.approx(breakeven_wr_gross(2.0))
    assert (breakeven_wr_net(2.0, 0.2) > breakeven_wr_net(2.0, 0.1)
            > breakeven_wr_gross(2.0))


def test_friction_r_is_derived_from_cost_model_not_hardcoded():
    """round_trip_friction_r must agree with the execution-cost model's own
    primitives, whatever calibration is loaded — fees both sides per leg
    plus half-spread both sides per leg, over the 1R denominator."""
    expected = (2.0 * fees_per_side(1, "OPT")
                + 2.0 * half_spread_cost(2.0, 1, "OPT")) / 100.0
    got = round_trip_friction_r(typical_premium=2.0, risk_per_unit=100.0)
    assert got == pytest.approx(expected)
    # two-leg vertical doubles both components
    got2 = round_trip_friction_r(
        typical_premium=2.0, risk_per_unit=100.0, n_option_legs=2)
    assert got2 == pytest.approx(2.0 * expected)
    with pytest.raises(ValueError):
        round_trip_friction_r(typical_premium=2.0, risk_per_unit=0.0)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", [
    "directional_long_call",
    "directional_long_put",
    "earnings_iv_drop",
    "earnings_iv_drop_post_print",
    "pullback_reversal",
    "breakout_squeeze",
    "Directional_Long_Call",   # case-insensitive
])
def test_long_premium_families_map_to_convexity(label):
    spec = spec_for_label(label)
    assert spec is not None and spec.name == "convexity_long_premium"


def test_exact_spec_name_resolves_directly():
    spec = spec_for_label("credit_put_spread_30_45")
    assert spec is not None and spec.name == "credit_put_spread_30_45"


@pytest.mark.parametrize("label", [
    None, "", "mean_reversion_pairs", "trend",
    # earnings_ alone is NOT enough — only the iv_drop family is long
    # premium; an earnings_straddle would be a different (undeclared) spec.
    "earnings_straddle",
])
def test_unknown_labels_are_legacy(label):
    assert spec_for_label(label) is None


# ---------------------------------------------------------------------------
# R7 per-spec floor in sizing
# ---------------------------------------------------------------------------

from trading_agent.sizing import (  # noqa: E402
    ProposedTrade,
    R7,
    SizingContext,
    check,
)


def _ctx() -> SizingContext:
    return SizingContext(equity=100_000.0, opens=(), sector_lookup_available=True)


def _trade(label: str, target: float = 13.0) -> ProposedTrade:
    """entry=10, stop=8 → risk $2/sh; target 13 → R:R 1.5."""
    return ProposedTrade(
        ticker="AAPL", asset_type="STK", side="BUY",
        qty=10, entry_price=10.0, stop=8.0, target=target,
        strategy_label=label,
    )


def test_r7_spec_floor_blocks_mapped_label_at_15():
    """R:R 1.5 clears the global 1.3 but NOT the convexity spec's 2.0 —
    a mapped label is held to its declared floor."""
    vs = [v for v in check(_ctx(), _trade("directional_long_call"))
          if v.rule == R7]
    assert len(vs) == 1 and vs[0].severity == "block"
    assert "convexity_long_premium" in vs[0].message  # names the spec
    assert "2.0" in vs[0].message


def test_r7_global_floor_still_governs_unmapped_label_at_15():
    """Same geometry under a legacy label passes at the global 1.3 floor."""
    vs = check(_ctx(), _trade("trend"))
    assert not any(v.rule == R7 for v in vs)


def test_r7_spec_floor_accepts_exactly_20():
    # entry=10, stop=8, target=14 → R:R exactly 2.0
    vs = check(_ctx(), _trade("directional_long_call", target=14.0))
    assert not any(v.rule == R7 for v in vs)


def test_r7_registry_failure_degrades_to_global_floor():
    """sizing runs inside the pretool hook + moomoo guard: a registry bug
    must degrade to the global floor, never crash or block validation."""
    with patch("trading_agent.strategy_specs.spec_for_label",
               side_effect=RuntimeError("registry exploded")):
        vs = check(_ctx(), _trade("directional_long_call"))
    # 1.5 >= global 1.3 → no R7 violation; and no exception escaped
    assert not any(v.rule == R7 for v in vs)


def test_r7_spec_floor_never_relaxes_below_global():
    """The blocked credit spread spec declares min_rr 0.40 — if that label
    somehow reaches a LONG open, the global 1.3 must still govern."""
    t = _trade("credit_put_spread_30_45", target=12.4)  # R:R 1.2 < 1.3
    vs = [v for v in check(_ctx(), t) if v.rule == R7]
    assert len(vs) == 1 and vs[0].severity == "block"
    assert "global floor" in vs[0].message


# ---------------------------------------------------------------------------
# Journal spec_comparison — declared vs realized
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_journal_db(tmp_path: Path, monkeypatch):
    from trading_agent import config as config_mod
    from trading_agent import db as db_mod

    db_file = tmp_path / "trader_test.db"
    new_cfg = dataclasses.replace(
        config_mod.CONFIG, db_path=db_file, data_dir=tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(db_mod, "CONFIG", new_cfg)
    db_mod.migrate(db_file)
    yield db_file


def _seed_closed_trades(label: str, pnls: list[float], *,
                        provenance: str = "agent") -> None:
    """Closed OPT trades: entry 2.0, stop 1.0, qty 1 → risk $100 → R=pnl/100."""
    from trading_agent.db import connection
    with connection() as conn:
        for pnl in pnls:
            conn.execute(
                "INSERT INTO trades (symbol, asset_type, strategy_label, side,"
                " qty, entry_price, exit_price, stop, target, pnl, outcome,"
                " opened_at, closed_at, provenance, is_paper)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                ("US.AAPL260717C00200000", "OPT", label, "BUY",
                 1, 2.0, 2.0 + pnl / 100.0, 1.0, 4.0, pnl,
                 "WIN" if pnl > 0 else "LOSS",
                 "2026-06-08T14:30:00+00:00", "2026-06-10T19:00:00+00:00",
                 provenance),
            )


def test_spec_comparison_insufficient_n_and_realized_stats(tmp_journal_db):
    _seed_closed_trades("directional_long_call", [250.0, -100.0, -100.0])
    from trading_agent.mcp_servers.journal.server import (
        generate_post_mortem_prompt,
    )
    out = generate_post_mortem_prompt("2026-06-01")
    sc = out["spec_comparison"]["directional_long_call"]
    assert sc["spec"] == "convexity_long_premium"
    assert sc["verdict_hint"] == "INSUFFICIENT_N"  # 3 < min_trades_for_eval 30
    r = sc["realized"]
    assert r["n"] == 3 and r["n_with_r"] == 3
    assert r["win_rate"] == pytest.approx(1 / 3, abs=1e-3)
    # R = pnl/100: (2.5 - 1.0 - 1.0)/3
    assert r["mean_r"] == pytest.approx(0.1667, abs=1e-3)
    assert r["avg_win"] == pytest.approx(250.0)
    assert r["avg_loss"] == pytest.approx(100.0)
    d = sc["declared"]
    assert d["expected_wr_range"] == [0.30, 0.45]
    assert d["min_risk_reward"] == 2.0
    assert 0 < d["breakeven_wr_net"] < 1


def test_spec_comparison_within_and_outside_declared(tmp_journal_db):
    # 30 trades at WR .40 → inside the declared 30-45% envelope
    _seed_closed_trades("pullback_reversal",
                        [250.0] * 12 + [-100.0] * 18)
    # 30 trades at WR .667 → ABOVE the envelope: also flagged (the structure
    # is not doing what was declared, even though it's winning)
    _seed_closed_trades("breakout_squeeze",
                        [250.0] * 20 + [-100.0] * 10)
    from trading_agent.mcp_servers.journal.server import (
        generate_post_mortem_prompt,
    )
    out = generate_post_mortem_prompt("2026-06-01")
    assert out["spec_comparison"]["pullback_reversal"]["verdict_hint"] == \
        "WITHIN_DECLARED"
    assert out["spec_comparison"]["breakout_squeeze"]["verdict_hint"] == \
        "OUTSIDE_DECLARED"


def test_spec_comparison_skips_legacy_labels_and_shadow_rows(tmp_journal_db):
    _seed_closed_trades("old_style_momo", [50.0, -50.0])       # no spec
    _seed_closed_trades("directional_long_call", [100.0],
                        provenance="virtual_backfill")          # shadow bucket
    from trading_agent.mcp_servers.journal.server import (
        generate_post_mortem_prompt,
    )
    out = generate_post_mortem_prompt("2026-06-01")
    assert out["spec_comparison"] == {}


def test_moratorium_instruction_rekeyed_to_expectancy(tmp_journal_db):
    from trading_agent.mcp_servers.journal.server import (
        generate_post_mortem_prompt,
    )
    instr = generate_post_mortem_prompt("2026-06-01")["instructions"]
    assert "win rate alone is NEVER a moratorium reason" in instr
    assert "LB95(mean R) < 0" in instr
    assert "<40% win rate" not in instr  # the old WR moratorium is gone
