"""Main entry point for the autonomous brain.

Dispatches subgraph runs based on trigger (premarket / monitor / eod /
healthcheck). Each run has a `thread_id` so PostgresSaver can checkpoint
its state and resume after a crash.

Usage:
    ta-brain --trigger=healthcheck
    ta-brain --trigger=premarket_scan
    ta-brain --serve   # placeholder for future systemd-driven loop

For Phase 2.1 the orchestrator is a one-shot CLI runner. Phase 2.5 wires
it into systemd + cron so the loop fires autonomously.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

from trading_agent.events import emit, new_run_id
from trading_agent.graph.builder import GRAPH_BUILDERS, build
from trading_agent.graph.state import empty_state

log = logging.getLogger(__name__)


def run_once(
    trigger: str,
    *,
    watchlist: list[str] | None = None,
    ticker: str | None = None,
) -> dict:
    """Compile + invoke a subgraph one time.

    Returns the final state dict. Side effects: agent_events rows + ntfy
    heartbeats from each stub node.

    `ticker` is used by candidate_entry_graph to drive a single per-ticker run.
    """
    if trigger not in GRAPH_BUILDERS:
        raise KeyError(f"Unknown trigger: {trigger}")

    run_id = new_run_id(trigger)
    ts = datetime.now(timezone.utc).isoformat()
    state = empty_state(run_id, trigger, ts)  # type: ignore[arg-type]
    if watchlist:
        state["watchlist"] = watchlist
    if ticker:
        state["candidates"] = [{"ticker": ticker, "score": 1.0, "reason": "cli"}]
        state["research"] = {"target_ticker": ticker}

    emit(
        run_id=run_id,
        trigger=trigger,  # type: ignore[arg-type]
        agent="orchestrator",
        event_type="run_start",
        payload={"thread_id": _thread_id(trigger, run_id, ts)},
    )

    graph = build(trigger)
    config = {"configurable": {"thread_id": _thread_id(trigger, run_id, ts)}}

    try:
        final_state = graph.invoke(state, config=config)
    except Exception as e:
        emit(
            run_id=run_id,
            trigger=trigger,  # type: ignore[arg-type]
            agent="orchestrator",
            event_type="run_failed",
            payload={"error": str(e)},
            severity=2,
        )
        raise
    else:
        emit(
            run_id=run_id,
            trigger=trigger,  # type: ignore[arg-type]
            agent="orchestrator",
            event_type="run_finished",
            payload={"nodes_visited": len(final_state.get("notifications", []))},
        )
        return final_state


def _thread_id(trigger: str, run_id: str, ts: str) -> str:
    """Build a LangGraph thread_id. Per the plan §5.1:
    - per-ticker runs:  '{ticker}:{date}:{trigger}'
    - portfolio runs:   'global:{date}:{trigger}'
    For Phase 2.1 stubs we use run_id directly so each invocation gets
    its own checkpoint stream (avoids state collisions during testing).
    """
    return f"{trigger}:{run_id}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser("ta-brain")
    p.add_argument(
        "--trigger",
        choices=list(GRAPH_BUILDERS.keys()),
        default="healthcheck",
        help="Which subgraph to run once",
    )
    p.add_argument(
        "--watchlist",
        default="",
        help="Comma-separated tickers (e.g. 'US.AAPL,US.SPY')",
    )
    p.add_argument(
        "--ticker",
        default="",
        help="Single ticker for candidate_entry runs (e.g. 'SPY')",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="DEBUG logging",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    watchlist = [t.strip() for t in args.watchlist.split(",") if t.strip()]
    ticker = args.ticker.strip() or None
    final = run_once(args.trigger, watchlist=watchlist, ticker=ticker)
    print(json.dumps({
        "trigger": args.trigger,
        "run_id": final.get("run_id"),
        "nodes_visited": [n.get("node") for n in final.get("notifications", [])],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
