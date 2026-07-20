"""Canonical combo detector (trading_agent.combo) — M1-0.1 lane A.

Pins the ONE detection rule both stores share, and its fail-closed contract:
malformed payloads parse to None (the row degrades to "unknown", never to
single-leg management, which would orphan the protective long wing).
"""
from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

SHORT_LEG = "US.SPY260828P00620000"
LONG_LEG = "US.SPY260828P00610000"


def _payload(**over) -> dict:
    p = {
        "combo": True, "short_leg": SHORT_LEG, "long_leg": LONG_LEG,
        "net_credit": 1.55, "width": 10.0, "max_loss": 845.0,
        "contracts": 2.0, "broker_order_ids": ["B77", "B78"],
    }
    p.update(over)
    return p


# ---------------------------------------------------------------------------
# combo_meta
# ---------------------------------------------------------------------------


def test_combo_meta_round_trips_canonical_dict():
    from trading_agent.combo import combo_meta
    m = combo_meta(_payload())
    assert m is not None
    assert m.short_leg == SHORT_LEG
    assert m.long_leg == LONG_LEG
    assert m.net_credit == pytest.approx(1.55)
    assert m.width == pytest.approx(10.0)
    assert m.max_loss == pytest.approx(845.0)
    assert m.contracts == pytest.approx(2.0)
    assert m.broker_order_ids == ("B77", "B78")


def test_combo_meta_accepts_sqlite_json_string_shape():
    """SQLite market_snapshots stores the payload as a JSON string."""
    from trading_agent.combo import combo_meta
    m = combo_meta(json.dumps(_payload()))
    assert m is not None and m.short_leg == SHORT_LEG


def test_combo_meta_optional_fields_tolerated():
    from trading_agent.combo import combo_meta
    m = combo_meta(_payload(width=None, max_loss=None, broker_order_ids=None))
    assert m is not None
    assert m.width is None and m.max_loss is None
    assert m.broker_order_ids == ()


@pytest.mark.parametrize("bad", [
    None,
    "",
    "not json {{{",
    42,
    ["combo"],
    {},                                        # no combo flag
    _payload(combo=False),                     # combo falsy
    _payload(short_leg=""),                    # missing short leg
    _payload(long_leg=None),                   # missing long leg
    _payload(net_credit=None),                 # net_credit unparseable
    _payload(net_credit="not-a-number"),
])
def test_combo_meta_malformed_is_none(bad):
    """Fail-closed: anything malformed → None, never a best-guess meta."""
    from trading_agent.combo import combo_meta
    assert combo_meta(bad) is None


# ---------------------------------------------------------------------------
# sqlite_combo_payloads + open_combo_long_legs
# ---------------------------------------------------------------------------


@pytest.fixture()
def combo_db(tmp_path: Path, monkeypatch):
    from trading_agent import config as config_mod
    from trading_agent import db as db_mod

    db_file = tmp_path / "trader_test.db"
    new_cfg = dataclasses.replace(config_mod.CONFIG, db_path=db_file,
                                  data_dir=tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(db_mod, "CONFIG", new_cfg)
    db_mod.migrate(db_file)
    yield db_file


def _insert_open_combo_sqlite(outcome: str = "OPEN") -> int:
    from trading_agent.db import connection
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO trades (symbol, asset_type, side, qty, entry_price, "
            "opened_at, outcome, is_paper) VALUES (?, 'OPT', 'SELL', 2, 1.55, "
            "?, ?, 1)",
            (SHORT_LEG, datetime.now(UTC).isoformat(), outcome),
        )
        tid = int(cur.lastrowid)
        conn.execute(
            "INSERT INTO market_snapshots (trade_id, taken_at, payload) "
            "VALUES (?, ?, ?)",
            (tid, datetime.now(UTC).isoformat(),
             json.dumps(_payload())),
        )
        conn.commit()
    return tid


def test_sqlite_combo_payloads_latest_wins(combo_db):
    from trading_agent.combo import sqlite_combo_payloads
    from trading_agent.db import connection
    tid = _insert_open_combo_sqlite()
    # A later snapshot for the same trade must shadow the earlier one.
    with connection() as conn:
        conn.execute(
            "INSERT INTO market_snapshots (trade_id, taken_at, payload) "
            "VALUES (?, ?, ?)",
            (tid, datetime.now(UTC).isoformat(),
             json.dumps(_payload(net_credit=1.60))),
        )
        conn.commit()
    out = sqlite_combo_payloads([tid])
    assert out[tid]["net_credit"] == 1.60


def test_sqlite_combo_payloads_skips_non_combo_and_garbage(combo_db):
    from trading_agent.combo import sqlite_combo_payloads
    from trading_agent.db import connection
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO trades (symbol, asset_type, side, qty, entry_price, "
            "opened_at, outcome, is_paper) VALUES ('US.AAPL', 'STK', 'BUY', "
            "10, 200, ?, 'OPEN', 1)",
            (datetime.now(UTC).isoformat(),),
        )
        tid = int(cur.lastrowid)
        conn.execute(
            "INSERT INTO market_snapshots (trade_id, taken_at, payload) "
            "VALUES (?, ?, 'not json')", (tid, "2026-01-01"))
        conn.execute(
            "INSERT INTO market_snapshots (trade_id, taken_at, payload) "
            "VALUES (?, ?, ?)",
            (tid, "2026-01-01", json.dumps({"quote": {"last": 1}})))
        conn.commit()
    assert sqlite_combo_payloads([tid]) == {}
    assert sqlite_combo_payloads([]) == {}


class _PgCtx:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **kw):
        pass

    def fetchall(self):
        return self._rows


def test_open_combo_long_legs_reads_both_stores(combo_db, monkeypatch):
    from trading_agent import combo as combo_mod
    _insert_open_combo_sqlite()
    pg_long = "US.QQQ260828P00500000"
    pg_payload = _payload(short_leg="US.QQQ260828P00510000",
                          long_leg=pg_long)
    monkeypatch.setattr("trading_agent.store.postgres.cursor",
                        lambda: _PgCtx([(pg_payload,)]))
    legs = combo_mod.open_combo_long_legs()
    assert LONG_LEG in legs      # from SQLite
    assert pg_long in legs       # from Postgres


def test_open_combo_long_legs_ignores_closed_rows(combo_db, monkeypatch):
    from trading_agent import combo as combo_mod
    _insert_open_combo_sqlite(outcome="WIN")
    monkeypatch.setattr("trading_agent.store.postgres.cursor",
                        lambda: _PgCtx([]))
    assert LONG_LEG not in combo_mod.open_combo_long_legs()


def test_open_combo_long_legs_store_error_degrades_not_raises(combo_db, monkeypatch):
    """A dark store contributes nothing; the other still answers."""
    from trading_agent import combo as combo_mod
    _insert_open_combo_sqlite()

    def _boom():
        raise ConnectionError("no postgres on this box")
    monkeypatch.setattr("trading_agent.store.postgres.cursor", _boom)
    legs = combo_mod.open_combo_long_legs()
    assert LONG_LEG in legs  # SQLite still contributes


def test_open_combo_long_legs_both_stores_dark_is_empty(tmp_path, monkeypatch):
    import dataclasses as dc

    from trading_agent import combo as combo_mod
    from trading_agent import config as config_mod
    from trading_agent import db as db_mod
    bad = tmp_path / "not_a_db"
    bad.mkdir()
    new_cfg = dc.replace(config_mod.CONFIG, db_path=bad)
    monkeypatch.setattr(config_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(db_mod, "CONFIG", new_cfg)

    def _boom():
        raise ConnectionError("down")
    monkeypatch.setattr("trading_agent.store.postgres.cursor", _boom)
    assert combo_mod.open_combo_long_legs() == set()
