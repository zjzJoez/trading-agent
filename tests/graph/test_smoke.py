"""Phase 2.1 smoke tests for the LangGraph skeleton.

Asserts that:
1. Each subgraph compiles without error
2. healthcheck_graph runs end-to-end (no Postgres / OpenD dependency)
3. premarket_scan_graph runs end-to-end (stubs only)

These tests require a live Postgres pointed at by POSTGRES_DSN. CI/dev
should run them inside the EC2 environment OR with a local docker
postgres. Skipped if Postgres unreachable.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("POSTGRES_DSN") and not (os.path.expanduser("~/.env.postgres") and os.path.exists(os.path.expanduser("~/.env.postgres"))),
    reason="Postgres DSN not configured (set POSTGRES_DSN or ~/.env.postgres)",
)


def test_all_graphs_compile():
    from trading_agent.graph.builder import GRAPH_BUILDERS

    for name, builder in GRAPH_BUILDERS.items():
        graph = builder()
        assert graph is not None, f"{name} failed to compile"


def test_healthcheck_graph_runs():
    from trading_agent.orchestrator import run_once

    final = run_once("healthcheck")
    assert final["run_id"], "run_id should be populated"
    assert final["trigger"] == "healthcheck"
    nodes = [n.get("node") for n in final.get("notifications", [])]
    assert "opend_health" in nodes
    assert "postgres_health" in nodes
    assert "ntfy_health" in nodes


def test_premarket_scan_graph_runs():
    from trading_agent.orchestrator import run_once

    final = run_once("premarket_scan", watchlist=["US.SPY", "US.QQQ"])
    assert final["watchlist"] == ["US.SPY", "US.QQQ"]
    nodes = [n.get("node") for n in final.get("notifications", [])]
    # Each premarket scan node should appear in the notifications log
    for expected in [
        "collect_macro_market_data",
        "compute_regime_features",
        "classify_regime",
        "rank_candidates",
        "ntfy_scan_digest",
    ]:
        assert expected in nodes, f"missing node: {expected}"


def test_eod_review_graph_runs():
    from trading_agent.orchestrator import run_once

    final = run_once("eod_review")
    nodes = [n.get("node") for n in final.get("notifications", [])]
    assert "reconcile_journal" in nodes
    assert "ntfy_daily_summary" in nodes
