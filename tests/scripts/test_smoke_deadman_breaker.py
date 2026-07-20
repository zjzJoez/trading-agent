"""The Week-1 Step-7 acceptance simulation must pass as part of the suite."""
from __future__ import annotations

import pathlib
import subprocess
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_smoke_deadman_breaker_passes():
    r = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "smoke_deadman_breaker.py")],
        capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "SMOKE PASSED" in r.stdout
