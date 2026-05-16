"""LLM behavior audit — deterministic checks on parsed LLM outputs.

Wired into `OAuthLLMRouter.call()`. Every LLM call's parsed output flows
through `audit_llm_output()` which dispatches to a role-specific audit
function. Violations are logged to `llm_behavior_audits`.

Design choices (per advisor calibration):
  - INLINE not batch — caught at the moment of the LLM call, not by cron
  - DETERMINISTIC checks only — no cross-LLM grading (own failure modes + cost)
  - START NARROW — 2 audit types (parse_failure, constraint_violation).
    Add role_compliance soft-checks only after we've seen real production
    failures in the audit table.
  - BEST-EFFORT — DB write failures must NEVER block the production LLM
    call. Audit infra is observability, not control.

What we CAN'T catch deterministically:
  - "The thesis is well-reasoned" — needs a grader
  - "The risk analysis is thorough" — same
  - "The trader's entry price is realistic" — needs market context
  These will surface as ACTUAL trade outcomes, not audit findings.

What we DO catch:
  - parse_failure: LLM returned invalid JSON / no parsed output
  - constraint_violation: parsed output violates a HARD role constraint
    (e.g. arbiter tried VETO→APPROVE, trader emitted stop on wrong side)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Role-specific check functions
# ---------------------------------------------------------------------------
#
# Each check function returns a list of violation dicts. Empty list = clean.
# Each violation dict has shape:
#   {"check": "<short_name>", "detail": "<human_readable>", "severity": "info|warn|error"}
#
# The check name should be short (< 40 chars) and stable so we can aggregate
# in queries: SELECT count(*) FROM llm_behavior_audits WHERE
#   violation_detail->>'check' = 'arbiter_tried_to_upgrade_veto'


def _check_risk_arbiter(parsed: Any, ctx: dict | None) -> list[dict]:
    """RiskArbiterOutput hard constraints (see risk/llm.py:145-154).

    Arbiter cannot upgrade VETO/DEFER to APPROVE; cannot exceed
    deterministic_factor. The router enforces these — we just record
    when the LLM TRIED to violate.
    """
    if parsed is None or ctx is None:
        return []
    v = []
    det = ctx.get("deterministic_decision")
    if det == "VETO" and parsed.decision != "VETO":
        v.append({
            "check": "arbiter_tried_to_upgrade_veto",
            "detail": f"deterministic=VETO but arbiter emitted {parsed.decision}",
            "severity": "error",
        })
    if det == "DEFER" and parsed.decision == "APPROVE":
        v.append({
            "check": "arbiter_tried_to_upgrade_defer_to_approve",
            "detail": f"deterministic=DEFER but arbiter emitted APPROVE",
            "severity": "error",
        })
    if det == "DOWNSIZE" and parsed.decision == "APPROVE":
        v.append({
            "check": "arbiter_tried_to_upgrade_downsize",
            "detail": f"deterministic=DOWNSIZE but arbiter emitted APPROVE",
            "severity": "warn",
        })
    # approved_qty must respect deterministic_factor when capped
    det_factor = float(ctx.get("deterministic_factor", 1.0))
    req_qty = float(ctx.get("requested_qty", 0))
    if req_qty > 0 and parsed.approved_qty > det_factor * req_qty * 1.01:  # 1% slop for rounding
        v.append({
            "check": "arbiter_exceeded_deterministic_factor",
            "detail": (f"approved_qty={parsed.approved_qty} > deterministic cap "
                       f"{det_factor * req_qty:.2f} (factor={det_factor})"),
            "severity": "error",
        })
    return v


def _check_trader_synthesizer(parsed: Any, ctx: dict | None) -> list[dict]:
    """TraderSynthesizerOutput hard constraints."""
    if parsed is None:
        return []
    v = []
    # decline_to_trade=True + proposal present is inconsistent (and vice versa)
    if parsed.decline_to_trade and parsed.proposal is not None:
        v.append({
            "check": "trader_decline_but_proposal_present",
            "detail": "decline_to_trade=True but proposal is non-null",
            "severity": "error",
        })
    if (not parsed.decline_to_trade) and parsed.proposal is None:
        v.append({
            "check": "trader_no_decline_no_proposal",
            "detail": "decline_to_trade=False but proposal is null",
            "severity": "error",
        })
    if parsed.proposal is None:
        return v

    p = parsed.proposal
    entry = float(p.entry_price)
    stop = float(p.stop)
    target = float(p.target)

    # Stop must be on the protective side, target on the profit side
    if p.direction in {"LONG", "LONG_CALL"}:
        if stop >= entry:
            v.append({
                "check": "trader_stop_wrong_side_long",
                "detail": f"LONG/LONG_CALL: stop ({stop}) must be < entry ({entry})",
                "severity": "error",
            })
        if target <= entry:
            v.append({
                "check": "trader_target_wrong_side_long",
                "detail": f"LONG/LONG_CALL: target ({target}) must be > entry ({entry})",
                "severity": "error",
            })
    elif p.direction == "LONG_PUT":
        if stop <= entry:
            v.append({
                "check": "trader_stop_wrong_side_long_put",
                "detail": f"LONG_PUT: stop ({stop}) must be > entry ({entry})",
                "severity": "error",
            })
        if target >= entry:
            v.append({
                "check": "trader_target_wrong_side_long_put",
                "detail": f"LONG_PUT: target ({target}) must be < entry ({entry})",
                "severity": "error",
            })

    # qty must be > 0 if trader is proposing
    if p.qty_request <= 0:
        v.append({
            "check": "trader_zero_qty_proposal",
            "detail": f"proposal qty_request={p.qty_request} is non-positive",
            "severity": "error",
        })

    # max_loss_pct sanity — should be positive
    if p.max_loss_pct <= 0 or p.max_loss_pct > 1.0:
        v.append({
            "check": "trader_max_loss_pct_out_of_range",
            "detail": f"max_loss_pct={p.max_loss_pct} (expected 0..1)",
            "severity": "warn",
        })
    return v


def _check_regime_reviewer(parsed: Any, ctx: dict | None) -> list[dict]:
    """RegimeReviewer can DOWNGRADE confidence or DEFER, never UPGRADE.
    """
    if parsed is None or ctx is None:
        return []
    v = []
    prior_conf = float(ctx.get("prior_confidence", 0.5))
    new_conf = getattr(parsed, "review_confidence", None)
    if new_conf is not None and float(new_conf) > prior_conf + 0.01:
        v.append({
            "check": "regime_reviewer_tried_to_upgrade",
            "detail": f"prior_confidence={prior_conf} but reviewer emitted {new_conf}",
            "severity": "error",
        })
    return v


def _check_bull_researcher(parsed: Any, ctx: dict | None) -> list[dict]:
    """Soft check: thesis non-empty, key_drivers non-empty."""
    if parsed is None:
        return []
    v = []
    thesis = (getattr(parsed, "thesis", "") or "").strip()
    if not thesis or len(thesis) < 20:
        v.append({
            "check": "bull_thesis_empty_or_short",
            "detail": f"thesis length = {len(thesis)} chars",
            "severity": "warn",
        })
    drivers = getattr(parsed, "key_drivers", []) or []
    if not drivers:
        v.append({
            "check": "bull_no_key_drivers",
            "detail": "key_drivers list is empty",
            "severity": "warn",
        })
    return v


def _check_bear_researcher(parsed: Any, ctx: dict | None) -> list[dict]:
    if parsed is None:
        return []
    v = []
    thesis = (getattr(parsed, "thesis", "") or "").strip()
    if not thesis or len(thesis) < 20:
        v.append({
            "check": "bear_thesis_empty_or_short",
            "detail": f"thesis length = {len(thesis)} chars",
            "severity": "warn",
        })
    drivers = getattr(parsed, "key_drivers", []) or []
    if not drivers:
        v.append({
            "check": "bear_no_key_drivers",
            "detail": "key_drivers list is empty",
            "severity": "warn",
        })
    return v


def _check_risk_decision(parsed: Any, ctx: dict | None) -> list[dict]:
    """Generic risk-reviewer check (conservative + opportunity)."""
    if parsed is None:
        return []
    v = []
    factor = float(getattr(parsed, "approved_qty_factor", 1.0))
    if factor < 0.0 or factor > 1.0:
        v.append({
            "check": "risk_factor_out_of_range",
            "detail": f"approved_qty_factor={factor} (expected 0..1)",
            "severity": "error",
        })
    # If decision is VETO but factor > 0, that's inconsistent
    decision = getattr(parsed, "decision", None)
    if decision == "VETO" and factor > 0.01:
        v.append({
            "check": "risk_veto_with_nonzero_factor",
            "detail": f"decision=VETO but factor={factor}",
            "severity": "error",
        })
    return v


# Role → check function dispatch
_CHECKS: dict[str, Callable[[Any, dict | None], list[dict]]] = {
    "risk_arbiter": _check_risk_arbiter,
    "risk_conservative": _check_risk_decision,
    "risk_opportunity": _check_risk_decision,
    "trader_synthesizer": _check_trader_synthesizer,
    "regime_reviewer": _check_regime_reviewer,
    "bull_researcher": _check_bull_researcher,
    "bear_researcher": _check_bear_researcher,
}


# ---------------------------------------------------------------------------
# Audit log writer
# ---------------------------------------------------------------------------


def _log_audit_row(
    *,
    role: str,
    audit_type: str,
    severity: str,
    violation_detail: dict,
    raw_output_excerpt: str | None,
    run_id: str | None,
    model_name: str | None,
    cost_usd: float | None,
) -> None:
    try:
        from trading_agent.store.postgres import cursor
        with cursor() as cur:
            cur.execute(
                """
                INSERT INTO llm_behavior_audits
                    (run_id, role, audit_type, severity,
                     violation_detail, raw_output_excerpt, model_name, cost_usd)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                """,
                (run_id, role, audit_type, severity,
                 json.dumps(violation_detail), raw_output_excerpt,
                 model_name, cost_usd),
            )
    except Exception as e:
        # Audit failures must not block the production LLM call
        log.warning("[llm_audit] insert failed (non-fatal): %s", e)


def audit_llm_output(
    *,
    role: str,
    parsed: Any | None,
    raw_text: str | None = None,
    ctx: dict | None = None,
    run_id: str | None = None,
    model_name: str | None = None,
    cost_usd: float | None = None,
) -> None:
    """Run audit checks for a single LLM call.

    `parsed`: the schema-validated Pydantic output (or None on parse failure).
    `raw_text`: the raw text the LLM produced — used as excerpt on
        parse_failure or violation. Trimmed to 500 chars.
    `ctx`: role-specific context (e.g. for risk_arbiter: deterministic
        decision/factor from upstream).
    """
    # Parse failure path
    if parsed is None:
        _log_audit_row(
            role=role,
            audit_type="parse_failure",
            severity="info",  # caller may fall back to stub; not always fatal
            violation_detail={"reason": "router returned parsed=None"},
            raw_output_excerpt=(raw_text or "")[:500] if raw_text else None,
            run_id=run_id,
            model_name=model_name,
            cost_usd=cost_usd,
        )
        return

    # Constraint check path
    check_fn = _CHECKS.get(role)
    if check_fn is None:
        return  # no audit defined for this role yet

    try:
        violations = check_fn(parsed, ctx)
    except Exception as e:
        log.warning("[llm_audit] check for role=%s raised: %s", role, e)
        return

    for v in violations:
        # Audit type: error → constraint_violation, warn/info → role_compliance
        audit_type = "constraint_violation" if v["severity"] == "error" else "role_compliance"
        _log_audit_row(
            role=role,
            audit_type=audit_type,
            severity=v["severity"],
            violation_detail=v,
            raw_output_excerpt=(
                json.dumps(parsed.model_dump(), default=str)[:500]
                if hasattr(parsed, "model_dump") else None
            ),
            run_id=run_id,
            model_name=model_name,
            cost_usd=cost_usd,
        )
        log.info("[llm_audit] role=%s check=%s severity=%s",
                 role, v["check"], v["severity"])
