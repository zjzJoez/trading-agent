"""Tests for scripts/calibrate_index_options.py — the M1-0.3 per-underlying,
per-ZONE calibration runner.

Hermetic: no OpenD, no Postgres (conftest blocks real Postgres anyway). The
snapshot source is exercised through an injected ``fetch`` seam.

The synthetic chain is deliberately LINEAR in strike (mid = 0.10 + 0.30·(k−75))
so every expectation below can be hand-derived:

    |delta|(k)   = 0.02 + 0.016·(k − 80)  (floored at 0.005)
    spread/mid   = 0.04 for k >= 91          (short-leg zone)
                   0.05 for 81 <= k <= 90    (wing zone)
                   0.01 for k <= 80          (far-OTM junk: TIGHT in percent
                                              terms — the 2026-07-22 zone-bug
                                              bait)
    in-zone shorts (|delta| 0.20-0.35): k = 92..100  (9 strikes)
    $5-wide wings  : 87..95      $10-wide wings : 82..90
"""
from __future__ import annotations

import dataclasses
import json
from datetime import date, timedelta
from decimal import Decimal

import pytest

import scripts.calibrate_index_options as cal
from trading_agent import execution_costs as ec

STRIKES = list(range(75, 101))
DAY = "2026-07-22"
EXPIRY = "2026-08-28"          # 37 DTE from DAY — inside the 30-45 band
DTE = 37


def _mid(k: float) -> float:
    return round(0.10 + 0.30 * (k - 75), 2)


def _abs_delta(k: float) -> float:
    return max(0.005, round(0.02 + 0.016 * (k - 80), 6))


def _spread_pct(k: float) -> float:
    if k >= 91:
        return 0.04
    if k >= 81:
        return 0.05
    return 0.01


def _bid_ask(k: float, pct: float | None = None) -> tuple[float, float]:
    m = _mid(k)
    p = _spread_pct(k) if pct is None else pct
    return m * (1 - p / 2), m * (1 + p / 2)


def _quotes(underlying="IWM", day=DAY, expiry=EXPIRY, dte=DTE,
            strikes=None) -> list[dict]:
    out = []
    for k in (strikes if strikes is not None else STRIKES):
        bid, ask = _bid_ask(k)
        q = cal._quote(underlying, day, expiry, dte, k, bid, ask, -_abs_delta(k))
        assert q is not None
        out.append(q)
    return out


def _snapshot_rows(underlyings=("IWM",), days=(DAY,)) -> list[tuple]:
    """Rows shaped like the real SELECT: Decimal numerics, negative put delta."""
    rows: list[tuple] = []
    for u in underlyings:
        for d in days:
            for k in STRIKES:
                bid, ask = _bid_ask(k)
                rows.append((u, d, EXPIRY, DTE, Decimal(str(k)),
                             Decimal(f"{bid:.4f}"), Decimal(f"{ask:.4f}"),
                             Decimal(f"{-_abs_delta(k):.4f}")))
    return rows


@pytest.fixture(autouse=True)
def _hermetic(tmp_path, monkeypatch):
    """tmp data_dir, never emit agent_events (even the best-effort fallback
    jsonl would land in the real $HOME), and never reach Postgres/OpenD."""
    from trading_agent import config as config_mod
    new_cfg = dataclasses.replace(config_mod.CONFIG, data_dir=tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(cal, "_emit_calibration_event", lambda *_a, **_k: None)
    monkeypatch.setattr(cal, "CHAIN_SLEEP_S", 0)  # no rate-limit pacing
    monkeypatch.setattr(cal, "_fetch_snapshot_rows",
                        lambda *_a, **_k: pytest.fail(
                            "test reached the real Postgres snapshot loader"))
    ec.reset_calibration_cache()
    yield tmp_path
    ec.reset_calibration_cache()


def _run_snapshots(monkeypatch, argv, rows=None):
    monkeypatch.setattr(
        cal, "_fetch_snapshot_rows",
        lambda names, since: _snapshot_rows() if rows is None else rows)
    return cal.main(["--source", "snapshots", *argv])


# ---------------------------------------------------------------------------
# Zone classification + pairing
# ---------------------------------------------------------------------------


def test_pairs_take_only_in_zone_shorts_and_the_exact_wing_strike():
    qs = _quotes()
    for width, wings in ((5.0, range(87, 96)), (10.0, range(82, 91))):
        pairs = cal.build_pairs(qs, width)
        assert len(pairs) == 9, width
        assert sorted(p["short"]["strike"] for p in pairs) == list(range(92, 101))
        assert sorted(p["long"]["strike"] for p in pairs) == list(wings)
        for p in pairs:
            assert (cal.SHORT_DELTA_MIN <= p["short"]["abs_delta"]
                    <= cal.SHORT_DELTA_MAX)
            assert p["short"]["strike"] - p["long"]["strike"] == width
            assert p["credit"] > 0


def test_pairs_never_span_two_chains():
    """A vertical is one expiry on one day — a 'pair' built across chains
    would be a fabricated structure with a meaningless credit."""
    qs = _quotes(strikes=[92]) + _quotes(day="2026-07-23", strikes=[87])
    assert cal.build_pairs(qs, 5.0) == []
    qs2 = _quotes(strikes=[92]) + _quotes(expiry="2026-09-18", strikes=[87])
    assert cal.build_pairs(qs2, 5.0) == []


def test_wing_ratio_credit_frac_and_legsum_hand_computed():
    st = cal.analyse(_quotes())
    w5 = st["by_width"][5.0]
    # median short leg is k=96 (mid 6.40); its wing k=91 (mid 4.90)
    assert _mid(96) == 6.40 and _mid(91) == 4.90
    assert w5["wing_ratio"] == pytest.approx(4.90 / 6.40)            # 0.765625
    # linear mids → every $5 pair collects exactly $1.50 on a $5 width
    assert w5["credit_frac_median"] == pytest.approx(1.50 / 5.0)
    assert w5["legsum_over_credit"] == pytest.approx((6.40 + 4.90) / 1.50)
    w10 = st["by_width"][10.0]
    assert w10["wing_ratio"] == pytest.approx((6.40 - 3.00) / 6.40)  # 0.53125
    assert w10["credit_frac_median"] == pytest.approx(3.00 / 10.0)
    # scalar picks: wing ratio = MAX over widths (conservative — legsum grows
    # in r), credit frac = the decision width's median.
    assert cal._pick_wing_ratio(st) == pytest.approx(4.90 / 6.40)
    assert cal._pick_credit_frac(st) == pytest.approx(0.30)


def test_zone_medians_are_separated_and_ignore_far_otm_junk():
    """THE 2026-07-22 ZONE BUG, pinned.

    Strikes <= 80 quote a 1% percent-spread — tighter than anything in the
    trading zone — because they are cheap, not because they are liquid.
    Pooling them (the old |delta| 0.03-0.40 sample) is what reported IWM at
    3.12% when the traded zone measures 4-5%.
    """
    st = cal.analyse(_quotes())
    assert st["short_zone_spread_pct"] == pytest.approx(0.04)
    assert st["wing_zone_spread_pct"] == pytest.approx(0.05)
    assert st["median_spread_pct_mid"] == pytest.approx(0.04)
    # the tight far-OTM strikes are present in the raw sample and excluded
    assert st["n_raw_quotes"] == len(STRIKES)
    assert st["n_short_zone"] == 9
    assert st["n_wing_zone"] == 14          # 82..95, deduped across widths
    assert st["n_quotes"] == 23             # 9 short + 14 wing
    # per width, the $10 structure's wing zone is entirely the wider band
    assert st["by_width"][10.0]["wing_zone_spread_pct"] == pytest.approx(0.05)
    assert st["by_width"][10.0]["short_zone_spread_pct"] == pytest.approx(0.04)


def test_quotes_without_delta_cannot_enter_the_short_zone():
    """No delta → no zone membership. The old runner fell back to a strike
    prefilter, which cannot tell a 0.25Δ short from a 0.05Δ wing."""
    qs = _quotes()
    for q in qs:
        q["abs_delta"] = None
    st = cal.analyse(qs)
    assert st["n_no_delta"] == len(qs)
    assert st["n_short_zone"] == 0 and st["n_pairs"] == 0
    assert st["short_zone_spread_pct"] is None


def test_crossed_or_unusable_rows_are_dropped():
    assert cal._quote("IWM", DAY, EXPIRY, DTE, 95, 2.0, 2.0, -0.25) is None   # ask==bid
    assert cal._quote("IWM", DAY, EXPIRY, DTE, 95, 2.1, 2.0, -0.25) is None   # crossed
    assert cal._quote("IWM", DAY, EXPIRY, DTE, 95, 0.0, 2.0, -0.25) is None   # no bid
    assert cal._quote("IWM", DAY, EXPIRY, DTE, 0, 1.0, 2.0, -0.25) is None    # no strike
    assert cal._quote("IWM", DAY, EXPIRY, DTE, 95, "x", 2.0, -0.25) is None   # junk
    ok = cal._quote("IWM", DAY, EXPIRY, DTE, 95, 1.0, 2.0, None)
    assert ok is not None and ok["abs_delta"] is None                         # no greek


# ---------------------------------------------------------------------------
# Snapshot loader
# ---------------------------------------------------------------------------


def test_snapshot_loader_normalizes_rows():
    got = cal.collect_snapshots(
        ["IWM", "SPY"], date(2026, 6, 1),
        fetch=lambda names, since: _snapshot_rows(underlyings=("IWM",)) + [
            ("SPY", DAY, EXPIRY, DTE, 700, 5.0, 5.2, -0.30),
            ("QQQ", DAY, EXPIRY, DTE, 700, 5.0, 5.2, -0.30),    # not requested
            ("IWM", DAY, EXPIRY, DTE, 95, 2.0, 1.0, -0.30),     # crossed
        ])
    assert set(got) == {"IWM", "SPY"}                  # QQQ row ignored
    assert len(got["IWM"]) == len(STRIKES)             # crossed row dropped
    q = next(x for x in got["IWM"] if x["strike"] == 95.0)
    assert isinstance(q["mid"], float)                 # Decimal → float
    assert q["abs_delta"] == pytest.approx(_abs_delta(95))   # sign stripped
    assert got["SPY"][0]["spread_pct"] == pytest.approx(0.2 / 5.1)


def test_snapshot_sql_is_read_only_and_zone_bounded():
    sql = cal._SNAPSHOT_SQL.upper()
    assert sql.strip().startswith("SELECT")
    for forbidden in ("INSERT", "UPDATE", "DELETE ", "DROP", "ALTER"):
        assert forbidden not in sql
    assert "OPTION_TYPE = 'PUT'" in sql
    assert "BETWEEN %(DTE_MIN)S AND %(DTE_MAX)S" in sql


# ---------------------------------------------------------------------------
# Decision line arithmetic
# ---------------------------------------------------------------------------


def test_decision_line_matches_execution_costs_friction_r(_hermetic):
    """The script's local arithmetic must equal the cost model's own
    friction_r when the model is fed the same measured figures."""
    (_hermetic / "execution_costs.json").write_text(json.dumps({
        "per_underlying": {"IWM": {
            "median_spread_pct_mid": 0.046,
            "short_zone_spread_pct": 0.0424,
            "wing_zone_spread_pct": 0.0504,
            "wing_ratio": 0.731,
        }}}))
    ec.reset_calibration_cache()
    mine = cal._decision_friction_r(
        short_zone_spread_pct=0.0424, wing_zone_spread_pct=0.0504,
        wing_ratio=0.731, net_credit=cal.DECISION_CREDIT)
    assert mine["friction_r"] == pytest.approx(
        ec.friction_r("IWM", cal.DECISION_WIDTH, cal.DECISION_CREDIT), abs=1e-6)


def test_decision_line_iwm_measured_is_about_eight_percent_of_r():
    """The headline number: at the MEASURED IWM figures a $5-wide vertical
    pays ~0.08R round trip — above the 0.06R whitelist bar. Hand check:
      credit 0.191×5 = 0.955 → legs 3.550/2.595 (r 0.731), legsum 6.145
      spread 2 × 100 × (0.0212×3.550 + 0.0252×2.595) = $28.13
      fees $4 ; R = (5 − 0.955)×100 = $404.50 → 32.13/404.50 = 0.0794R
    """
    d = cal._decision_friction_r(
        short_zone_spread_pct=0.0424, wing_zone_spread_pct=0.0504,
        wing_ratio=0.731, net_credit=0.191 * 5.0)
    assert d["short_mark"] == pytest.approx(0.955 / (1 - 0.731), abs=1e-3)
    assert d["legsum"] == pytest.approx(6.145, abs=0.01)
    assert d["risk"] == pytest.approx(404.5, abs=0.01)
    assert d["friction_r"] == pytest.approx(0.079, abs=0.002)
    assert d["friction_r"] > cal.DECISION_BAR_R
    # ... and the old model's headline 0.031R is no longer reachable
    assert d["friction_r"] > 2 * 0.031


def test_worst_friction_r_takes_the_pessimistic_credit_scenario():
    entry = {"short_zone_spread_pct": 0.0424, "wing_zone_spread_pct": 0.0504,
             "wing_ratio": 0.731, "credit_frac_median": 0.191}
    both = [cal._decision_friction_r(
        short_zone_spread_pct=0.0424, wing_zone_spread_pct=0.0504,
        wing_ratio=0.731, net_credit=c)["friction_r"]
        for c in (cal.DECISION_CREDIT, 0.191 * 5.0)]
    assert cal._worst_friction_r(entry) == pytest.approx(max(both))
    assert cal._worst_friction_r({}) is None            # nothing measured


def test_decision_scenarios_are_per_width_with_that_width_s_own_figures():
    """Width changes the wing ratio, the credit fraction AND R, so each width
    must be judged on its own measurement — the real IWM numbers put the
    $5-wide above the 0.06R bar and the $10-wide below it."""
    entry = {
        "short_zone_spread_pct": 0.0424, "wing_zone_spread_pct": 0.0504,
        "wing_ratio": 0.7308, "credit_frac_median": 0.191,
        "by_width": {
            "5": {"n_pairs": 32, "short_zone_spread_pct": 0.04242,
                  "wing_zone_spread_pct": 0.05038, "wing_ratio": 0.7308,
                  "credit_frac_median": 0.191},
            "10": {"n_pairs": 25, "short_zone_spread_pct": 0.03742,
                   "wing_zone_spread_pct": 0.0574, "wing_ratio": 0.5305,
                   "credit_frac_median": 0.1755},
            "20": {"n_pairs": 0},                       # no data → no scenario
        },
    }
    scen = cal._decision_scenarios(entry)
    assert {s["width"] for s in scen} == {5.0, 10.0}
    at10 = [s for s in scen if s["width"] == 10.0]
    assert {s["label"].split()[0] for s in at10} == {"gate-floor", "MEASURED"}
    assert [s["net_credit"] for s in at10] == pytest.approx([2.5, 1.755])
    assert all(s["wing_ratio"] == 0.5305 for s in at10)   # not the 5-wide ratio
    assert cal._worst_friction_r(entry, width=5.0) > cal.DECISION_BAR_R
    assert cal._best_friction_r(entry, width=10.0) < cal.DECISION_BAR_R
    # the headline verdict stays keyed on the plan's $5-wide decision line
    assert cal._worst_friction_r(entry) == pytest.approx(
        cal._worst_friction_r(entry, width=5.0))


def test_decision_scenarios_fall_back_to_the_scalar_entry():
    entry = {"short_zone_spread_pct": 0.0424, "wing_zone_spread_pct": 0.0504,
             "wing_ratio": 0.731, "credit_frac_median": 0.191}
    scen = cal._decision_scenarios(entry)
    assert {s["width"] for s in scen} == {cal.DECISION_WIDTH}
    assert cal._decision_scenarios({"by_width": {"5": {"n_pairs": 3}}}) == []


# ---------------------------------------------------------------------------
# CLI — snapshots mode
# ---------------------------------------------------------------------------


def test_snapshots_mode_writes_zone_entry_and_preserves_globals(
        _hermetic, monkeypatch, capsys):
    existing_globals = {
        "calibrated_at": "2026-06-12T07:13:59+00:00",
        "n_samples": 160,
        "spread_pct_of_mid_median": 0.07338,
        "opt_half_spread_pct_of_mark": 0.03669,
    }
    (_hermetic / "execution_costs.json").write_text(json.dumps(existing_globals))
    rc = _run_snapshots(monkeypatch,
                        ["--underlyings", "IWM", "--min-quotes", "20",
                         "--min-pairs", "10"],
                        rows=_snapshot_rows(days=(DAY, "2026-07-23")))
    assert rc == 0
    saved = json.loads((_hermetic / "execution_costs.json").read_text())
    for k, v in existing_globals.items():
        assert saved[k] == v                     # global figures byte-identical
    e = saved["per_underlying"]["IWM"]
    assert e["short_zone_spread_pct"] == pytest.approx(0.04)
    assert e["wing_zone_spread_pct"] == pytest.approx(0.05)
    assert e["wing_ratio"] == pytest.approx(0.7656, abs=1e-4)
    assert e["credit_frac_median"] == pytest.approx(0.30)
    assert e["n_pairs"] == 36                        # 9 shorts × 2 widths × 2 days
    assert e["n_short_zone"] == 18 and e["n_wing_zone"] == 28
    assert e["n_quotes"] == 46
    assert e["source"] == "snapshots"
    assert e["sample_window"].startswith("snapshots ")
    assert "SHORT zone |delta| 0.2-0.35" in e["zone"]
    assert e["calibrated_at"]
    assert set(e["by_width"]) == {"5", "10"}
    assert e["by_width"]["10"]["wing_zone_spread_pct"] == pytest.approx(0.05)
    assert e["by_width"]["5"]["n_chains"] == 2
    assert e["by_width"]["5"]["dte_observed"] == [DTE, DTE]
    # the freshly written file flows into the cost model, zone-aware
    ec.reset_calibration_cache()
    assert ec.half_spread_cost(2.0, 1, "OPT", underlying="IWM",
                               zone="wing") == pytest.approx(2.0 * 0.025 * 100)
    assert ec.half_spread_cost(2.0, 1, "OPT", underlying="IWM",
                               zone="short") == pytest.approx(2.0 * 0.02 * 100)
    out = capsys.readouterr().out
    assert "short-zone 4.00%" in out and "wing-zone 5.00%" in out
    assert "report only" in out


def test_snapshots_mode_reports_insufficient_data_without_writing(
        _hermetic, monkeypatch, capsys):
    """SPY/QQQ as of 2026-07-25: a handful of days, nothing in the zone. The
    honest answer is 'insufficient data', never an invented figure."""
    rc = _run_snapshots(
        monkeypatch, ["--underlyings", "SPY,QQQ", "--min-quotes", "40"],
        rows=[("SPY", DAY, EXPIRY, DTE, 700, 5.0, 5.2, -0.30)])
    assert rc == 1
    assert not (_hermetic / "execution_costs.json").exists()
    out = capsys.readouterr().out
    assert "SPY: INSUFFICIENT DATA" in out
    assert "QQQ: INSUFFICIENT DATA" in out
    assert "NOT ASSESSED" in out
    assert "nothing written" in out


def test_iwm_verdict_excluded_at_measured_spreads(_hermetic, monkeypatch, capsys):
    """A chain whose zone spreads match the real IWM measurement must print
    the EXCLUDED verdict with the honest friction, not the old 0.031R."""
    rows = []
    for k in STRIKES:
        pct = 0.0424 if k >= 91 else 0.0504
        bid, ask = _bid_ask(k, pct)
        rows.append(("IWM", DAY, EXPIRY, DTE, k, bid, ask, -_abs_delta(k)))
    rc = _run_snapshots(monkeypatch,
                        ["--underlyings", "IWM", "--min-quotes", "20",
                         "--dry-run"], rows=rows)
    assert rc == 0
    out = capsys.readouterr().out
    assert "IWM EXCLUDED (friction_r" in out
    assert "> 0.06R" in out
    assert "gate-floor credit width/4" in out
    assert "MEASURED credit" in out
    assert "legsum/credit" in out               # the arithmetic is shown
    assert "dry-run" in out
    assert not (_hermetic / "execution_costs.json").exists()


def test_dry_run_touches_nothing(_hermetic, monkeypatch, capsys):
    rc = _run_snapshots(monkeypatch, ["--underlyings", "IWM", "--min-quotes",
                                      "20", "--dry-run"])
    assert rc == 0
    assert not (_hermetic / "execution_costs.json").exists()
    assert "dry-run" in capsys.readouterr().out


def test_merge_preserves_other_per_underlying_entries(_hermetic, monkeypatch):
    (_hermetic / "execution_costs.json").write_text(json.dumps({
        "opt_half_spread_pct_of_mark": 0.03669,
        "per_underlying": {"QQQ": {"median_spread_pct_mid": 0.012,
                                   "n_quotes": 50}},
    }))
    rc = _run_snapshots(monkeypatch, ["--underlyings", "IWM", "--min-quotes", "20"])
    assert rc == 0
    saved = json.loads((_hermetic / "execution_costs.json").read_text())
    assert saved["per_underlying"]["QQQ"]["median_spread_pct_mid"] == 0.012
    assert saved["per_underlying"]["IWM"]["short_zone_spread_pct"] == pytest.approx(0.04)
    assert saved["opt_half_spread_pct_of_mark"] == 0.03669


def test_non_dict_json_refused_without_overwrite(_hermetic, monkeypatch, capsys):
    (_hermetic / "execution_costs.json").write_text("[1, 2, 3]")
    rc = _run_snapshots(monkeypatch, ["--underlyings", "IWM", "--min-quotes", "20"])
    assert rc == 1
    assert "refusing to overwrite non-object JSON" in capsys.readouterr().err
    assert (_hermetic / "execution_costs.json").read_text() == "[1, 2, 3]"


def test_snapshot_read_failure_is_fatal_not_a_garbage_calibration(
        _hermetic, monkeypatch, capsys):
    def boom(*_a, **_k):
        raise RuntimeError("no POSTGRES_DSN")

    monkeypatch.setattr(cal, "_fetch_snapshot_rows", boom)
    rc = cal.main(["--source", "snapshots", "--underlyings", "IWM"])
    assert rc == 2
    assert "option_chain_snapshots read failed" in capsys.readouterr().err
    assert not (_hermetic / "execution_costs.json").exists()


def test_source_auto_follows_the_market_clock(monkeypatch):
    import trading_agent.market_calendar as mc
    monkeypatch.setattr(mc, "is_us_market_open", lambda *_a, **_k: False)
    assert cal._resolve_source("auto") == "snapshots"
    monkeypatch.setattr(mc, "is_us_market_open", lambda *_a, **_k: True)
    assert cal._resolve_source("auto") == "live"
    # explicit always wins over the clock
    assert cal._resolve_source("snapshots") == "snapshots"
    assert cal._resolve_source("live") == "live"


# ---------------------------------------------------------------------------
# CLI — live mode
# ---------------------------------------------------------------------------


def _mk_quote_fns(spot=100.0, n_expiries=2):
    """Fake (get_quote, list_option_expiries, get_option_chain) triple over
    the same synthetic chain, addressed by option code."""
    in_window = [(date.today() + timedelta(days=d)).isoformat()
                 for d in (32, 40)][:n_expiries]
    out_window = (date.today() + timedelta(days=5)).isoformat()

    def get_quote(codes):
        rows = []
        for i, c in enumerate(codes):
            if "PUT" not in c:
                rows.append({"code": c, "last_price": spot})
                continue
            k = float(c.split("PUT")[1])
            bid, ask = _bid_ask(k)
            row = {"code": c, "bid_price": bid, "ask_price": ask,
                   "strike_price": k}
            # Real snapshot rows carry ``option_delta``; keep the first row of
            # each batch on legacy ``delta`` to pin the fallback.
            row["delta" if i == 0 else "option_delta"] = -_abs_delta(k)
            rows.append(row)
        return {"rows": rows}

    def list_option_expiries(code):
        return {"rows": [{"strike_time": s} for s in in_window + [out_window]]}

    def get_option_chain(code, expiry, option_type="ALL"):
        assert option_type == "PUT"           # sleeve zone is puts only
        assert expiry in in_window            # the 5-DTE expiry must not be pulled
        return {"rows": [{"code": f"US.IWM{expiry}PUT{k:g}", "strike_price": k}
                         for k in STRIKES]}

    return get_quote, list_option_expiries, get_option_chain


def _run_live(monkeypatch, fns, argv):
    monkeypatch.setattr(cal, "_load_moomoo", lambda: fns)
    return cal.main(["--source", "live", *argv])


def test_live_mode_samples_wide_then_classifies_zones(
        _hermetic, monkeypatch, capsys):
    """Live sampling must reach BELOW the short zone — the old runner's
    |delta| filter cut the wings off, so no pair could be formed at all."""
    rc = _run_live(monkeypatch, _mk_quote_fns(),
                   ["--underlyings", "IWM", "--min-quotes", "20"])
    assert rc == 0
    saved = json.loads((_hermetic / "execution_costs.json").read_text())
    e = saved["per_underlying"]["IWM"]
    assert e["source"] == "live"
    assert e["sample_window"].startswith("live ")
    assert e["n_pairs"] == 36                  # 9 shorts × 2 widths × 2 expiries
    assert e["short_zone_spread_pct"] == pytest.approx(0.04)
    assert e["wing_zone_spread_pct"] == pytest.approx(0.05)
    assert "IWM EXCLUDED (friction_r" in capsys.readouterr().out


def test_live_expiry_cap_limits_chain_calls(_hermetic, monkeypatch):
    """Futu rate limit: at most MAX_EXPIRIES_PER_NAME chain calls per name."""
    get_quote, _, _ = _mk_quote_fns()
    many = [(date.today() + timedelta(days=d)).isoformat() for d in range(31, 40)]
    calls: list[str] = []

    def list_exp(code):
        return {"rows": [{"strike_time": s} for s in many]}

    def counting_chain(code, expiry, option_type="ALL"):
        calls.append(expiry)
        return {"rows": [{"code": f"US.IWM{expiry}PUT{k:g}", "strike_price": k}
                         for k in STRIKES]}

    rc = _run_live(monkeypatch, (get_quote, list_exp, counting_chain),
                   ["--underlyings", "IWM", "--min-quotes", "20"])
    assert rc == 0
    assert calls == many[:cal.MAX_EXPIRIES_PER_NAME]


def test_live_opend_unreachable_fails_fast(_hermetic, monkeypatch, capsys):
    def boom():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(cal, "_load_moomoo", boom)
    rc = cal.main(["--source", "live", "--underlyings", "SPY,QQQ,IWM"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "MoomooOpenD unreachable" in err
    assert "--source snapshots" in err
    assert not (_hermetic / "execution_costs.json").exists()
