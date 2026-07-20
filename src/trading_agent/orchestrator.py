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
import os
import sys
from datetime import datetime, timezone

from trading_agent.events import emit, new_run_id
from trading_agent.graph.builder import GRAPH_BUILDERS, build
from trading_agent.graph.state import empty_state

log = logging.getLogger(__name__)


def _shutdown_resources() -> None:
    """Close PostgresSaver pool, store pool, and moomoo SDK contexts.

    Each of these spawns non-daemon background threads that block the
    interpreter on exit. Without explicit close, oneshot CLI runs hang in
    ``threading._shutdown()`` until systemd's TimeoutStartSec SIGTERMs them
    (we observed ~5 min wall-clock for ~3 s of real CPU work via py-spy).

    Errors are logged and swallowed — the goal is fast exit, not perfect
    cleanup. After this call the caller should ``os._exit(0)`` as a final
    belt-and-braces in case some transitively-imported library has its own
    non-daemon thread we don't know about.
    """
    try:
        from trading_agent.graph.checkpointer import shutdown as _ck_shutdown
        _ck_shutdown()
    except Exception as e:
        log.warning("checkpointer shutdown failed: %s", e)
    try:
        from trading_agent.store.postgres import shutdown as _pg_shutdown
        _pg_shutdown()
    except Exception as e:
        log.warning("store shutdown failed: %s", e)
    try:
        from trading_agent.mcp_servers.moomoo.server import shutdown as _mm_shutdown
        _mm_shutdown()
    except Exception as e:
        log.warning("moomoo shutdown failed: %s", e)


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

    # Lifecycle payload — `ticker` is the watchdog correlation key for
    # candidate_entry runs. If a candidate_entry_dispatched event is in
    # agent_events but no matching `run_start` for the same ticker shows up
    # within 5 min, the watchdog in healthcheck flags it as a silent die.
    # (This is what would have caught the 5/13 SPY systemd-cgroup kill in
    # 5 min instead of 22 hours.)
    lifecycle_payload = {
        "thread_id": _thread_id(trigger, run_id, ts),
        "ticker": ticker,  # None for non-per-ticker triggers
    }

    emit(
        run_id=run_id,
        trigger=trigger,  # type: ignore[arg-type]
        agent="orchestrator",
        event_type="run_start",
        payload=lifecycle_payload,
    )

    graph = build(trigger)
    config = {"configurable": {"thread_id": _thread_id(trigger, run_id, ts)}}

    # Stream instead of invoke so we can record which nodes actually ran.
    # "updates" chunks are keyed by node name; the last "values" chunk is
    # the final state (identical to what invoke() would have returned).
    nodes_visited: list[str] = []
    final_state: dict | None = None
    try:
        for mode, chunk in graph.stream(
            state, config=config, stream_mode=["updates", "values"]
        ):
            if mode == "updates":
                nodes_visited.extend(
                    k for k in chunk if not k.startswith("__")
                )
            else:
                final_state = chunk
    except Exception as e:
        emit(
            run_id=run_id,
            trigger=trigger,  # type: ignore[arg-type]
            agent="orchestrator",
            event_type="run_failed",
            payload={"error": str(e), "ticker": ticker, "nodes_visited": nodes_visited},
            severity=2,
        )
        raise
    else:
        if final_state is None:
            final_state = dict(state)
        final_state["nodes_visited"] = nodes_visited
        emit(
            run_id=run_id,
            trigger=trigger,  # type: ignore[arg-type]
            agent="orchestrator",
            event_type="run_finished",
            payload={
                "nodes_visited": nodes_visited,
                "ticker": ticker,
            },
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

    exit_code = 0
    try:
        final = run_once(args.trigger, watchlist=watchlist, ticker=ticker)
        print(json.dumps({
            "trigger": args.trigger,
            "run_id": final.get("run_id"),
            "nodes_visited": final.get("nodes_visited", []),
        }, indent=2, default=str))
    except SystemExit as e:
        # Convert to plain exit_code so we still reach os._exit() below.
        # Otherwise SystemExit would bypass the os._exit() hammer and we'd
        # be back at the mercy of threading._shutdown() for any unknown
        # non-daemon thread.
        exit_code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:  # noqa: BLE001 — we re-raise after cleanup
        log.exception("orchestrator run failed")
        exit_code = 1
    finally:
        _shutdown_resources()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

    # Bypass `threading._shutdown()` — we've closed every pool/socket we
    # know about, but any transitively-imported library could still be
    # holding a non-daemon thread. ``os._exit`` is the only guarantee that
    # systemd sees a fast clean exit instead of the 5-minute TimeoutStartSec.
    os._exit(exit_code)


if __name__ == "__main__":
    main()
