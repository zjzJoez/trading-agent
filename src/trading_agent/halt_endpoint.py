"""FastAPI /halt endpoint — emergency killswitch for the autonomous loop.

The orchestrator's entry subgraph checks for `data/halt.flag` before
spawning new positions. /halt creates the flag; /resume removes it.
Intended invocation: iOS Shortcut → POST /halt with X-Halt-Token header.

NOT exposed publicly. Either:
- Behind Cloudflare Tunnel + Access (recommended), or
- AWS Security Group allowlists user's mobile IP, or
- Tailscale-only (no public exposure).

Run via: uvicorn trading_agent.halt_endpoint:app --host 0.0.0.0 --port 8443
"""
from __future__ import annotations

import hmac
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException

from trading_agent.events import SEV_CRITICAL, emit, new_run_id
from trading_agent.notify import send as ntfy_send

log = logging.getLogger(__name__)

# Flag file checked by every entry subgraph before spawning new positions.
HALT_FLAG = Path.home() / "trading-agent" / "data" / "halt.flag"

app = FastAPI(title="trading-agent halt endpoint", version="3.0")


def _check_token(provided: str | None) -> None:
    expected = os.environ.get("HALT_TOKEN")
    if not expected:
        # Refuse to operate without a configured token — fail closed.
        raise HTTPException(status_code=503, detail="HALT_TOKEN not configured")
    if not provided or not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=401, detail="invalid token")


@app.get("/health")
def health() -> dict:
    """Public health probe — returns whether halt is currently active."""
    return {
        "service": "halt-endpoint",
        "halted": HALT_FLAG.exists(),
        "halt_flag_path": str(HALT_FLAG),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/halt")
def halt(x_halt_token: str | None = Header(default=None, alias="X-Halt-Token")) -> dict:
    """Activate halt: write data/halt.flag. Entry subgraph refuses to spawn
    until the flag is removed via /resume. Exits + monitor keep running."""
    _check_token(x_halt_token)
    HALT_FLAG.parent.mkdir(parents=True, exist_ok=True)
    already = HALT_FLAG.exists()
    HALT_FLAG.touch()
    halted_at = datetime.now(timezone.utc).isoformat()

    run_id = new_run_id("halt")
    emit(
        run_id=run_id,
        trigger="healthcheck",
        agent="halt_endpoint",
        event_type="halt_activated",
        payload={"already": already, "halted_at": halted_at},
        severity=SEV_CRITICAL,
    )
    ntfy_send(
        "ops",
        title="🛑 Trading halted by user",
        body=f"data/halt.flag created at {halted_at}.\nNew entries blocked. Exits + monitor continue.",
        priority=5,
        tags=["stop_sign", "rotating_light"],
    )
    return {"halted": True, "halted_at": halted_at, "already_halted": already}


@app.post("/resume")
def resume(x_halt_token: str | None = Header(default=None, alias="X-Halt-Token")) -> dict:
    """Deactivate halt: remove data/halt.flag. Entry subgraph resumes."""
    _check_token(x_halt_token)
    existed = HALT_FLAG.exists()
    if existed:
        HALT_FLAG.unlink()
    resumed_at = datetime.now(timezone.utc).isoformat()

    run_id = new_run_id("halt")
    emit(
        run_id=run_id,
        trigger="healthcheck",
        agent="halt_endpoint",
        event_type="halt_resumed",
        payload={"flag_existed": existed, "resumed_at": resumed_at},
        severity=SEV_CRITICAL,
    )
    ntfy_send(
        "ops",
        title="▶️ Trading resumed",
        body=f"halt.flag removed at {resumed_at}. Autonomous loop accepting new entries.",
        priority=4,
        tags=["white_check_mark"],
    )
    return {"halted": False, "resumed_at": resumed_at, "was_halted": existed}
