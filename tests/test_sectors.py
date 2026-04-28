"""Sectors lookup — module-level cache + ticker normalization."""
from __future__ import annotations

from pathlib import Path

import pytest

from trading_agent import sectors


@pytest.fixture(autouse=True)
def reset_sectors_module():
    """Force re-load on each test so cached state from prior tests doesn't leak."""
    sectors._table = None  # type: ignore[attr-defined]
    sectors._source = None  # type: ignore[attr-defined]
    yield
    sectors._table = None  # type: ignore[attr-defined]
    sectors._source = None  # type: ignore[attr-defined]


def test_lookup_known_ticker():
    assert sectors.lookup("AAPL") == "Information Technology"


def test_lookup_strips_us_prefix():
    assert sectors.lookup("US.AAPL") == "Information Technology"


def test_lookup_lowercase_normalizes():
    assert sectors.lookup("aapl") == "Information Technology"


def test_lookup_unknown_returns_none():
    assert sectors.lookup("__UNKNOWN_TICKER__") is None


def test_lookup_empty_string_returns_none():
    assert sectors.lookup("") is None


def test_known_count_positive():
    assert sectors.known_count() > 50


def test_load_handles_missing_file(tmp_path):
    bogus = tmp_path / "does_not_exist.csv"
    table = sectors.load(bogus)
    assert table == {}
    assert sectors.known_count() == 0


def test_load_custom_csv(tmp_path):
    p = tmp_path / "tiny.csv"
    p.write_text("ticker,sector\nFOO,Energy\nBAR,Health Care\n")
    table = sectors.load(p)
    assert table == {"FOO": "Energy", "BAR": "Health Care"}
    assert sectors.lookup("foo") == "Energy"
