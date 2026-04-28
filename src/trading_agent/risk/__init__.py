"""Phase 2.3 — Active Risk Agent.

Portfolio-level decision authority: APPROVE / DOWNSIZE / VETO / DEFER over
every trade proposal. Sees all open positions, current regime, correlation,
factor exposures, aggregate Greeks, heat metric.

Layered architecture (§7 of plan v3):
  1. portfolio.py     — gather open positions + equity + recent quotes
  2. greeks.py        — aggregate net delta/gamma/vega/theta + stress
  3. correlation.py   — pairwise corr matrix from cached klines
  4. factors.py       — sector / rates / risk-on factor exposures
  5. heat.py          — sum of (loss-to-stop / option max loss)
  6. guardrails.py    — per-regime tiered hard limits (Python-enforced)
  7. agent.py         — orchestrate aggregates → decision
  8. llm.py           — Phase-2.4 LLM council scaffolding (stub for now)
  9. persist.py       — risk_snapshots + risk_decisions Postgres writes
"""

from trading_agent.risk.agent import RiskInput, RiskOutput, decide  # noqa: F401
from trading_agent.risk.guardrails import (  # noqa: F401
    Guardrails,
    GuardrailViolation,
    check_guardrails,
)
from trading_agent.risk.portfolio import PortfolioSnapshot, build_snapshot  # noqa: F401
