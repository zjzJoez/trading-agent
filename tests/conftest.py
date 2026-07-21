"""Test-wide fixtures.

The single most important thing here is ``_block_real_ntfy``: an autouse
fixture that patches ``trading_agent.notify.send`` so NO test can
accidentally fire a real ntfy push notification to the operator's phone.

Background: 2026-06-02 incident. A new ``rank_candidates`` fallback path
emitted a sev-2 ntfy alert when scout LLM failed. The existing test
``test_fallback_when_llm_fails`` triggered that path with a mocked
RuntimeError but did NOT mock ``notify.send`` — so every CI run AND
every auto-deploy pytest gate POST'd to ``ntfy.sh`` for real, hitting
the operator's phone. We caught it on the first deploy; not a great
test posture in general.

This fixture autouse-patches the bottom-level helper. Tests that need
to verify ntfy was called (asserting on title, body, priority) can
override by re-patching at their own scope — the outer autouse patch
just guarantees no test will ever leak to real ntfy.sh.
"""
from __future__ import annotations

import pathlib
import sys
from unittest.mock import patch

import pytest

# Make the repo root importable so tests can `import scripts.*` (the scripts/
# dir is a namespace package, not part of the installed trading_agent wheel).
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(autouse=True)
def _block_real_ntfy():
    """Patch trading_agent.notify.send for the duration of every test.

    The patch target is the SOURCE function, so any caller that does
    ``from trading_agent.notify import send as ntfy_send`` and then
    invokes ``ntfy_send(...)`` still hits the mock — Python re-resolves
    the import at call time via the bound name only for the original
    function in trading_agent.notify, which we've replaced.

    Tests that need to assert ntfy behavior can scope their own
    ``with patch("trading_agent.notify.send") as m: ...`` inside;
    that re-patch wins for the duration of the inner scope.
    """
    with patch("trading_agent.notify.send"):
        yield


@pytest.fixture(autouse=True)
def _block_real_postgres(request, monkeypatch):
    """No unit test may reach a real Postgres. Mirrors _block_real_ntfy.

    Background: 2026-07-20 incident. The auto-deploy pytest gate runs on
    the EC2 host with the production POSTGRES_DSN in its environment
    (the same EnvironmentFile that lets the deploy script apply
    migrations). The combo dual-write path exercised by
    test_posttool_combo_capture is best-effort — locally, with no DSN,
    it silently no-ops and the test passes; on the gate it connected to
    the LIVE journal and leaked a fixture row (journal_trades id=9,
    broker_order_id 'L1'), which the nightly reconcile then flagged.

    Stripping env vars is not enough: _resolve_dsn falls back to reading
    ~/.env.postgres off disk, which exists on the EC2 host. So this
    patches _resolve_dsn itself to raise — best-effort callers degrade
    to their intended no-DSN behavior, strict callers fail loudly in
    the test instead of touching production. checkpointer.py binds
    _resolve_dsn by name at import, so both bindings are patched.

    Integration-marked tests are exempt (the deploy gate deselects them
    with -m 'not integration'; running them locally against a real DB
    is a deliberate act).
    """
    if request.node.get_closest_marker("integration"):
        yield
        return
    for var in ("POSTGRES_DSN", "DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)

    def _refuse(*_a, **_k):
        raise RuntimeError(
            "unit test attempted real Postgres access — mock the store "
            "or mark the test @pytest.mark.integration"
        )

    from trading_agent.graph import checkpointer as _ckpt
    from trading_agent.store import postgres as _pg
    monkeypatch.setattr(_pg, "_resolve_dsn", _refuse)
    monkeypatch.setattr(_ckpt, "_resolve_dsn", _refuse)
    # A pool created by an exempt (integration) test earlier in the same
    # session must not be reachable from unit tests either.
    monkeypatch.setattr(_pg, "_pool", None)
    yield
