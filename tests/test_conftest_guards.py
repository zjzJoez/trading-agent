"""The conftest autouse guards themselves — a leak here reaches production.

_block_real_ntfy grew out of the 2026-06-02 phone-spam incident;
_block_real_postgres out of the 2026-07-20 journal-leak incident (the
deploy gate ran the suite with the production DSN in env and a fixture
row landed in the live journal_trades). These tests pin the guarantees.
"""
from __future__ import annotations

import pytest


def test_unit_tests_cannot_resolve_a_postgres_dsn(monkeypatch):
    """Even with a DSN in env (the deploy-gate environment), unit tests
    must not be able to resolve it."""
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://prod@example.com/prod")
    from trading_agent.store.postgres import _resolve_dsn
    with pytest.raises(RuntimeError, match="mock the store"):
        _resolve_dsn()


def test_checkpointer_binding_is_also_blocked():
    """checkpointer.py binds _resolve_dsn by name at import — patching
    only the source module would leave this path open."""
    from trading_agent.graph import checkpointer
    with pytest.raises(RuntimeError, match="mock the store"):
        checkpointer._resolve_dsn()


def test_pool_singleton_is_reset_for_unit_tests():
    from trading_agent.store import postgres as pg
    assert pg._pool is None
