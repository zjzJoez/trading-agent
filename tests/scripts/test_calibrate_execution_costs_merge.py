"""scripts/calibrate_execution_costs.py must not destroy the other writer's data.

Two scripts write data/execution_costs.json:

* ``calibrate_execution_costs.py`` (this one) owns the GLOBAL single-name
  figures — ``opt_half_spread_pct_of_mark`` and its provenance;
* ``calibrate_index_options.py`` owns ``per_underlying`` (per-name, per-zone
  spreads, wing ratios, credit fractions), which ``execution_costs.friction_r``
  now depends on.

Until 2026-07-25 this script did ``out.write_text(json.dumps(payload))``, so a
routine global re-calibration WIPED per_underlying and silently returned every
index name to the blind global fallback — the cost model would keep working and
keep reporting worse-founded numbers, with nothing in the output saying so.
"""
from __future__ import annotations

import json

import pytest

import scripts.calibrate_execution_costs as cal


def test_merge_write_preserves_per_underlying_and_unknown_keys(tmp_path):
    out = tmp_path / "execution_costs.json"
    out.write_text(json.dumps({
        "opt_half_spread_pct_of_mark": 0.03669,
        "spread_pct_of_mid_median": 0.07338,
        "per_underlying": {"IWM": {"median_spread_pct_mid": 0.0474,
                                   "wing_ratio": 0.734,
                                   "by_width": {"5": {"n_pairs": 32,
                                                      "n_chains": 3}}}},
        "some_future_key": [1, 2, 3],
    }))
    payload = {
        "calibrated_at": "2026-07-25T00:00:00+00:00",
        "n_samples": 120,
        "tickers": ["AAPL", "SPY"],
        "spread_pct_of_mid_median": 0.05,
        "spread_pct_of_mid_p75": 0.08,
        "opt_half_spread_pct_of_mark": 0.025,
    }
    assert cal._merge_write(out, payload) == 0
    saved = json.loads(out.read_text())
    # this script's own keys are updated ...
    assert saved["opt_half_spread_pct_of_mark"] == 0.025
    assert saved["spread_pct_of_mid_median"] == 0.05
    assert saved["n_samples"] == 120
    # ... and nothing else is touched
    assert saved["per_underlying"]["IWM"]["wing_ratio"] == 0.734
    assert saved["per_underlying"]["IWM"]["by_width"]["5"]["n_chains"] == 3
    assert saved["some_future_key"] == [1, 2, 3]


def test_merge_write_creates_the_file_when_absent(tmp_path):
    out = tmp_path / "execution_costs.json"
    assert cal._merge_write(out, {"opt_half_spread_pct_of_mark": 0.02}) == 0
    assert json.loads(out.read_text()) == {"opt_half_spread_pct_of_mark": 0.02}


@pytest.mark.parametrize("body", ["{not json", "[1, 2, 3]", '"a string"'])
def test_unparseable_or_non_object_json_is_refused_without_overwrite(
        tmp_path, capsys, body):
    """Same guard as calibrate_index_options: a file we cannot parse is a file
    whose other writer's data we cannot preserve, so refuse rather than
    clobber."""
    out = tmp_path / "execution_costs.json"
    out.write_text(body)
    assert cal._merge_write(out, {"opt_half_spread_pct_of_mark": 0.02}) == 1
    assert out.read_text() == body
    assert "refusing to overwrite" in capsys.readouterr().err


def test_owned_keys_are_exactly_what_the_payload_declares():
    """Provenance guard: if a new global figure is added to the payload it must
    be added to _OWNED_KEYS too, or it would silently never be written."""
    payload_keys = {
        "calibrated_at", "n_samples", "tickers", "spread_pct_of_mid_median",
        "spread_pct_of_mid_p75", "opt_half_spread_pct_of_mark",
    }
    assert set(cal._OWNED_KEYS) == payload_keys
    assert "per_underlying" not in cal._OWNED_KEYS
