"""Tests for scripts/calibrate_index_options.py (M1-0.3 per-underlying
calibration runner) — mocked quote source; no OpenD, no Postgres."""
from __future__ import annotations

import dataclasses
import json
from datetime import date, timedelta

import pytest

import scripts.calibrate_index_options as cal
from trading_agent import execution_costs as ec

SPOT = 100.0


@pytest.fixture(autouse=True)
def _hermetic(tmp_path, monkeypatch):
    """tmp data_dir + never emit agent_events (even best-effort fallback
    jsonl would land in the real $HOME)."""
    from trading_agent import config as config_mod
    new_cfg = dataclasses.replace(config_mod.CONFIG, data_dir=tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    emitted: list[dict] = []
    monkeypatch.setattr(cal, "_emit_calibration_event", emitted.append)
    monkeypatch.setattr(cal, "CHAIN_SLEEP_S", 0)  # no rate-limit pacing in tests
    ec.reset_calibration_cache()
    yield tmp_path
    ec.reset_calibration_cache()


def _mk_quote_fns(spread_by_name: dict[str, float], *, n_strikes=40,
                  delta=0.25, mid=2.0):
    """Fake (get_quote, list_option_expiries, get_option_chain) triple.

    Every option quote for name N carries a full spread of exactly
    ``spread_by_name[N]`` × mid, so the measured median equals it.
    """
    in_window = [(date.today() + timedelta(days=d)).isoformat() for d in (32, 40)]
    out_window = (date.today() + timedelta(days=5)).isoformat()

    def get_quote(codes):
        rows = []
        for i, c in enumerate(codes):
            if "PUT" in c:
                name = c[3:c.index("PUT")]
                pct = spread_by_name[name]
                row = {
                    "code": c,
                    "bid_price": mid * (1 - pct / 2),
                    "ask_price": mid * (1 + pct / 2),
                }
                # Real snapshot rows carry ``option_delta``; keep the first
                # row of each batch on legacy ``delta`` to pin the fallback.
                row["delta" if i == 0 else "option_delta"] = -delta
                rows.append(row)
            else:
                rows.append({"code": c, "last_price": SPOT})
        return {"rows": rows}

    def list_option_expiries(code):
        return {"rows": [{"strike_time": s} for s in in_window + [out_window]]}

    def get_option_chain(code, expiry, option_type="ALL"):
        assert option_type == "PUT"          # sleeve zone is puts only
        assert expiry in in_window           # 5-DTE expiry must not be pulled
        name = code[3:]
        return {"rows": [
            {"code": f"US.{name}PUT{99.5 - 0.5 * i:.1f}",
             "strike_price": 99.5 - 0.5 * i}
            for i in range(n_strikes)
        ]}

    return get_quote, list_option_expiries, get_option_chain


def _run(monkeypatch, fns, argv):
    monkeypatch.setattr(cal, "_load_moomoo", lambda: fns)
    return cal.main(argv)


def test_happy_path_writes_per_underlying_and_preserves_globals(
        _hermetic, monkeypatch, capsys):
    tmp_path = _hermetic
    existing_globals = {
        "calibrated_at": "2026-06-12T07:13:59+00:00",
        "n_samples": 160,
        "spread_pct_of_mid_median": 0.07338,
        "opt_half_spread_pct_of_mark": 0.03669,
    }
    (tmp_path / "execution_costs.json").write_text(json.dumps(existing_globals))

    fns = _mk_quote_fns({"SPY": 0.010, "QQQ": 0.012, "IWM": 0.030})
    rc = _run(monkeypatch, fns,
              ["--underlyings", "SPY,QQQ,IWM", "--min-quotes", "40"])
    assert rc == 0

    saved = json.loads((tmp_path / "execution_costs.json").read_text())
    # global figures byte-identical
    for k, v in existing_globals.items():
        assert saved[k] == v
    pu = saved["per_underlying"]
    assert pu["SPY"]["median_spread_pct_mid"] == pytest.approx(0.010)
    assert pu["QQQ"]["median_spread_pct_mid"] == pytest.approx(0.012)
    assert pu["IWM"]["median_spread_pct_mid"] == pytest.approx(0.030)
    for entry in pu.values():
        assert entry["n_quotes"] >= 40
        assert entry["calibrated_at"]
        assert "PUTS 30-45 DTE" in entry["zone"]

    # the freshly written file flows into the cost model
    ec.reset_calibration_cache()
    assert ec.half_spread_cost(2.0, 1, "OPT", underlying="SPY") == pytest.approx(
        2.0 * (0.010 / 2) * 100)

    # IWM at 3% spread: fees $4 + spread 4×(0.015×1.25×100)=$7.50 → 11.5/375
    out = capsys.readouterr().out
    assert "IWM OK (friction_r 0.031R <= 0.06R)" in out


def test_iwm_excluded_decision_line(_hermetic, monkeypatch, capsys):
    fns = _mk_quote_fns({"IWM": 0.080})
    rc = _run(monkeypatch, fns, ["--underlyings", "IWM", "--min-quotes", "40"])
    assert rc == 0
    # fees $4 + spread 4×(0.04×1.25×100)=$20 → 24/375 = 0.064R > 0.06R
    assert "IWM EXCLUDED (friction_r 0.064R > 0.06R)" in capsys.readouterr().out


def test_decision_line_matches_execution_costs_friction_r(_hermetic):
    tmp_path = _hermetic
    (tmp_path / "execution_costs.json").write_text(json.dumps({
        "per_underlying": {"IWM": {"median_spread_pct_mid": 0.030}}}))
    ec.reset_calibration_cache()
    assert cal._decision_friction_r(0.030) == pytest.approx(
        ec.friction_r("IWM", cal.DECISION_WIDTH, cal.DECISION_CREDIT),
        abs=1e-6)


def test_min_quotes_skips_thin_name_keeps_others(_hermetic, monkeypatch, capsys):
    tmp_path = _hermetic

    real = _mk_quote_fns({"SPY": 0.010, "QQQ": 0.012})
    get_quote, list_exp, get_chain = real

    def thin_chain(code, expiry, option_type="ALL"):
        if code == "US.QQQ":
            rows = get_chain(code, expiry, option_type)["rows"][:3]
            return {"rows": rows}
        return get_chain(code, expiry, option_type)

    rc = _run(monkeypatch, (get_quote, list_exp, thin_chain),
              ["--underlyings", "SPY,QQQ", "--min-quotes", "40"])
    assert rc == 0
    pu = json.loads((tmp_path / "execution_costs.json").read_text())["per_underlying"]
    assert "SPY" in pu and "QQQ" not in pu
    assert "QQQ: only 6 usable quotes" in capsys.readouterr().out


def test_all_names_thin_writes_nothing_and_fails(_hermetic, monkeypatch):
    tmp_path = _hermetic
    fns = _mk_quote_fns({"SPY": 0.010}, n_strikes=2)
    rc = _run(monkeypatch, fns, ["--underlyings", "SPY", "--min-quotes", "40"])
    assert rc == 1
    assert not (tmp_path / "execution_costs.json").exists()


def test_out_of_zone_delta_filtered_out(_hermetic, monkeypatch):
    tmp_path = _hermetic
    # |delta| 0.60 is outside the 0.03-0.40 sleeve zone → zero usable quotes
    fns = _mk_quote_fns({"SPY": 0.010}, delta=0.60)
    rc = _run(monkeypatch, fns, ["--underlyings", "SPY", "--min-quotes", "40"])
    assert rc == 1
    assert not (tmp_path / "execution_costs.json").exists()


def test_expiry_cap_limits_chain_calls(_hermetic, monkeypatch):
    """Futu rate limit: at most MAX_EXPIRIES_PER_NAME chain calls per name."""
    get_quote, _, _ = _mk_quote_fns({"SPY": 0.010})
    many = [(date.today() + timedelta(days=d)).isoformat() for d in range(31, 40)]

    def list_exp(code):
        return {"rows": [{"strike_time": s} for s in many]}

    calls: list[str] = []

    def counting_chain(code, expiry, option_type="ALL"):
        calls.append(expiry)
        return {"rows": [
            {"code": f"US.SPYPUT{99.5 - 0.5 * i:.1f}",
             "strike_price": 99.5 - 0.5 * i}
            for i in range(40)
        ]}

    rc = _run(monkeypatch, (get_quote, list_exp, counting_chain),
              ["--underlyings", "SPY", "--min-quotes", "40"])
    assert rc == 0
    assert len(calls) == cal.MAX_EXPIRIES_PER_NAME
    assert calls == many[:cal.MAX_EXPIRIES_PER_NAME]


def test_non_dict_json_refused_without_overwrite(_hermetic, monkeypatch, capsys):
    tmp_path = _hermetic
    (tmp_path / "execution_costs.json").write_text("[1, 2, 3]")
    fns = _mk_quote_fns({"SPY": 0.010})
    rc = _run(monkeypatch, fns, ["--underlyings", "SPY", "--min-quotes", "40"])
    assert rc == 1
    assert "refusing to overwrite non-object JSON" in capsys.readouterr().err
    assert (tmp_path / "execution_costs.json").read_text() == "[1, 2, 3]"


def test_opend_unreachable_fails_fast(_hermetic, monkeypatch, capsys):
    tmp_path = _hermetic

    def boom():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(cal, "_load_moomoo", boom)
    rc = cal.main(["--underlyings", "SPY,QQQ,IWM", "--min-quotes", "40"])
    assert rc == 2
    assert "MoomooOpenD unreachable" in capsys.readouterr().err
    assert not (tmp_path / "execution_costs.json").exists()


def test_merge_preserves_other_per_underlying_entries(_hermetic, monkeypatch):
    tmp_path = _hermetic
    (tmp_path / "execution_costs.json").write_text(json.dumps({
        "opt_half_spread_pct_of_mark": 0.03669,
        "per_underlying": {"QQQ": {"median_spread_pct_mid": 0.012,
                                   "n_quotes": 50}},
    }))
    fns = _mk_quote_fns({"SPY": 0.010})
    rc = _run(monkeypatch, fns, ["--underlyings", "SPY", "--min-quotes", "40"])
    assert rc == 0
    saved = json.loads((tmp_path / "execution_costs.json").read_text())
    assert saved["per_underlying"]["QQQ"]["median_spread_pct_mid"] == 0.012
    assert saved["per_underlying"]["SPY"]["median_spread_pct_mid"] == pytest.approx(0.010)
    assert saved["opt_half_spread_pct_of_mark"] == 0.03669


def test_dry_run_touches_nothing(_hermetic, monkeypatch, capsys):
    tmp_path = _hermetic
    fns = _mk_quote_fns({"SPY": 0.010})
    rc = _run(monkeypatch, fns,
              ["--underlyings", "SPY", "--min-quotes", "40", "--dry-run"])
    assert rc == 0
    assert not (tmp_path / "execution_costs.json").exists()
    assert "dry-run" in capsys.readouterr().out
