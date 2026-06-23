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
