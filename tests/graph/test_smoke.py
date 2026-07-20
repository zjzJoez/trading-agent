"""Graph smoke tests — compilation + integration.

Two tiers:
  1. Compile-only (no external deps) — runs always, patches Postgres with MemorySaver.
  2. Integration (requires POSTGRES_DSN) — skipped in environments without Postgres.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Tier 1 — compile-only, MemorySaver substitution (always runs)
# ---------------------------------------------------------------------------

def _memory_saver():
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


def test_all_graphs_compile_with_memory_saver():
    """All 6 subgraphs must compile; no Postgres needed."""
    with patch("trading_agent.graph.builder.get_saver", _memory_saver):
        from trading_agent.graph.builder import GRAPH_BUILDERS
        for name, builder in GRAPH_BUILDERS.items():
            g = builder()
            assert g is not None, f"{name} failed to compile"


def test_graph_builder_registry_has_all_triggers():
    from trading_agent.graph.builder import GRAPH_BUILDERS
    expected = {
        "premarket_scan", "candidate_entry", "intraday_monitor",
        "eod_review", "weekly_learning", "healthcheck",
    }
    assert set(GRAPH_BUILDERS.keys()) == expected


def test_builder_no_longer_imports_stubs():
    """builder.py must not import the stubs module (all nodes are real)."""
    import ast
    import importlib
    import pathlib
    src = pathlib.Path("src/trading_agent/graph/builder.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in getattr(node, "names", [])]
            module = getattr(node, "module", "") or ""
            assert "stubs" not in module and all("stubs" not in n for n in names), \
                f"builder.py still imports stubs: {ast.dump(node)}"


def test_run_once_records_nodes_visited(monkeypatch):
    """run_once must report the nodes that actually executed.

    Regression: the summary derived nodes_visited from state["notifications"],
    which only stub nodes ever appended to — so production runs always
    printed "nodes_visited": [] even though nodes ran.
    """
    from langgraph.graph import END, START, StateGraph

    import trading_agent.orchestrator as orch
    from trading_agent.graph.state import TradingGraphState

    g = StateGraph(TradingGraphState)
    g.add_node("node_a", lambda state: {})
    g.add_node("node_b", lambda state: {})
    g.add_edge(START, "node_a")
    g.add_edge("node_a", "node_b")
    g.add_edge("node_b", END)
    compiled = g.compile(checkpointer=_memory_saver())

    events = []
    monkeypatch.setattr(orch, "GRAPH_BUILDERS", {"healthcheck": lambda: compiled})
    monkeypatch.setattr(orch, "build", lambda trigger: compiled)
    monkeypatch.setattr(orch, "emit", lambda **kw: events.append(kw))

    final = orch.run_once("healthcheck")
    assert final["nodes_visited"] == ["node_a", "node_b"]
    finished = [e for e in events if e["event_type"] == "run_finished"]
    assert len(finished) == 1
    assert finished[0]["payload"]["nodes_visited"] == ["node_a", "node_b"]


# ---------------------------------------------------------------------------
# Tier 2 — integration tests (Postgres required)
# ---------------------------------------------------------------------------

_POSTGRES_AVAILABLE = bool(
    os.environ.get("POSTGRES_DSN")
    or os.path.exists(os.path.expanduser("~/.env.postgres"))
)
pytestmark_pg = pytest.mark.skipif(
    not _POSTGRES_AVAILABLE,
    reason="Postgres DSN not configured (set POSTGRES_DSN or ~/.env.postgres)",
)


# These tests touch LIVE infrastructure: real Postgres writes, real ntfy
# pushes, real LLM calls (eod_review runs generate_eod_digest which is an
# LLM-backed node). Marked `integration` so the pre-deploy pytest gate can
# skip them via `-m 'not integration'`. Otherwise every deploy retry
# spam-fires EOD digests + heartbeats to production (5/22 incident).
@pytest.mark.integration
@pytest.mark.skipif(not _POSTGRES_AVAILABLE, reason="requires Postgres")
def test_healthcheck_graph_completes():
    from trading_agent.orchestrator import run_once
    final = run_once("healthcheck")
    assert final["run_id"]
    assert final["trigger"] == "healthcheck"


@pytest.mark.integration
@pytest.mark.skipif(not _POSTGRES_AVAILABLE, reason="requires Postgres")
def test_eod_review_graph_completes():
    from trading_agent.orchestrator import run_once
    final = run_once("eod_review")
    assert final["run_id"]
    assert final["trigger"] == "eod_review"
