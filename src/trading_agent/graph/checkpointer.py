"""Postgres checkpointer for LangGraph.

Uses langgraph-checkpoint-postgres. Schema is auto-created on first call
to `setup()`. Connection comes from the Phase 2 Postgres pool, but we
hold our own connection pool here because LangGraph's checkpointer
prefers a dedicated pool with autocommit.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from trading_agent.store.postgres import _resolve_dsn

log = logging.getLogger(__name__)

_pool: ConnectionPool | None = None
_saver: PostgresSaver | None = None


def get_pool() -> ConnectionPool:
    """Lazy singleton — LangGraph wants autocommit on its own pool."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=_resolve_dsn(),
            min_size=1,
            max_size=4,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=True,
        )
    return _pool


def get_saver() -> PostgresSaver:
    """Return a singleton PostgresSaver. Schema is set up on first call."""
    global _saver
    if _saver is None:
        saver = PostgresSaver(get_pool())
        saver.setup()  # idempotent; creates checkpoint tables if missing
        _saver = saver
    return _saver


@contextmanager
def saver_context() -> Iterator[PostgresSaver]:
    """Convenience: yield the saver. Exists for parity with the docs idiom."""
    yield get_saver()
