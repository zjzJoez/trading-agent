"""GICS-sector lookup for sizing R4.

Backed by the static `data/sectors.csv` file (`ticker,sector`).  Loaded once
on first import — the file is small (≤ 100 rows) and rarely changes.

Why this is a stand-alone module: ``deterministic_sizing`` and
``load_open_positions`` both need it, and the moomoo SDK doesn't surface
GICS classifications.  The CSV is checked into the repo so the lookup
works offline + deterministically.

When a ticker is missing, ``lookup`` returns ``None`` and the sizing layer
emits a single ``R4_sector_unknown`` warning (instead of failing closed).
The CI-style `scripts/check_soak_acceptance.py` reports the warning rate so
operators know when to extend the CSV.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from threading import Lock
from typing import Final

log = logging.getLogger(__name__)

# Locate data/sectors.csv relative to the project root regardless of CWD.
# Three parents up from this file: src/trading_agent/sectors.py → repo root.
_DEFAULT_CSV: Final[Path] = (
    Path(__file__).resolve().parents[2] / "data" / "sectors.csv"
)

_lock = Lock()
_table: dict[str, str] | None = None
_source: Path | None = None


def _normalize(ticker: str) -> str:
    """Strip exchange prefix + uppercase. ``US.AAPL`` → ``AAPL``."""
    t = (ticker or "").strip().upper()
    for prefix in ("US.", "HK.", "SG.", "."):
        if t.startswith(prefix):
            t = t[len(prefix):]
    return t


def load(path: Path | None = None) -> dict[str, str]:
    """Load (or reload) the lookup table.  Idempotent."""
    global _table, _source
    p = path or _DEFAULT_CSV
    out: dict[str, str] = {}
    try:
        with p.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = (row.get("ticker") or "").strip().upper()
                sector = (row.get("sector") or "").strip()
                if ticker and sector:
                    out[ticker] = sector
    except FileNotFoundError:
        log.warning("sectors.csv not found at %s — R4 will degrade", p)
    except Exception as e:
        log.warning("sectors.csv parse failed (%s); R4 will degrade", e)
    with _lock:
        _table = out
        _source = p
    log.info("loaded %d sectors from %s", len(out), p)
    return out


def _ensure_loaded() -> dict[str, str]:
    if _table is None:
        load()
    return _table or {}


def lookup(ticker: str) -> str | None:
    """Return the GICS sector for ``ticker``, or ``None`` if unknown."""
    if not ticker:
        return None
    table = _ensure_loaded()
    return table.get(_normalize(ticker))


def known_count() -> int:
    return len(_ensure_loaded())


def source_path() -> Path | None:
    _ensure_loaded()
    return _source


__all__ = ["load", "lookup", "known_count", "source_path"]
