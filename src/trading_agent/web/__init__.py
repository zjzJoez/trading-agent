"""Dashboard web UI for the trading-agent.

A single-page FastAPI service that exposes:
  * JSON endpoints under /api/* for live state
  * Static assets (HTML/CSS/JS + lightweight chart) for the UI

Run locally:
    uvicorn trading_agent.web.api:app --host 127.0.0.1 --port 8000 --reload

Run in production (systemd unit recommended):
    uvicorn trading_agent.web.api:app --host 0.0.0.0 --port 8001
"""
