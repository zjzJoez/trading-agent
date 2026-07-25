"""Tests for scripts/verify_replay_report.py — the REPORT.md cross-check.

The verifier's job is to fail when a prose claim and its artifact row disagree.
So the tests are (a) it passes against the committed artifacts, and (b) it
actually fails when an artifact number is moved — including the specific
failure mode that got past the first review pass: a value that still EXISTS
somewhere in the artifact but no longer belongs to the row the claim names.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import scripts.verify_replay_report as vr

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports" / "vertical_replay_2026-07"


def test_every_committed_report_claim_matches_its_artifact_row():
    # This is the gate that keeps REPORT.md and the artifacts numerically in
    # sync: regenerate the artifacts, and any prose number left behind fails
    # here rather than shipping.
    c = vr.verify(REPORT_DIR)
    assert c.rows, "the verifier checked nothing at all"
    assert not c.failures, "\n".join(
        f"{r['what']}: report={r['report_says']!r} artifact={r['artifact_says']!r}"
        for r in c.failures)


def test_main_exits_zero_on_the_committed_artifacts(capsys):
    assert vr.main(["--report-dir", str(REPORT_DIR)]) == 0
    out = capsys.readouterr().out
    assert "ALL CLAIMS MATCH THEIR ARTIFACT ROW" in out


def _clone(tmp_path: Path) -> Path:
    dst = tmp_path / "report"
    shutil.copytree(REPORT_DIR, dst,
                    ignore=shutil.ignore_patterns("inputs", "sensitivity"))
    return dst


def test_verifier_fails_when_a_scalar_moves(tmp_path):
    dst = _clone(tmp_path)
    path = dst / "summary.json"
    payload = json.loads(path.read_text())
    payload["availability"]["overall"]["n_qualified"] = 27      # was 26
    path.write_text(json.dumps(payload))
    c = vr.verify(dst)
    assert any("n_qualified" in r["what"] for r in c.failures)


def test_verifier_fails_when_a_value_moves_to_a_different_row(tmp_path):
    # THE regression this file exists for (finding 4). Global value-membership
    # passes here: 1.41 is still present in the artifact, just on another
    # trade. A tuple match must not.
    dst = _clone(tmp_path)
    path = dst / "diagnostics.json"
    payload = json.loads(path.read_text())
    rows = payload["samples"]["as_reported_close_sameday"][
        "exit_trigger_audit"]["all_rows"]
    target = next(r for r in rows
                  if r["entry_date"] == "2025-10-13" and r["symbol"] == "QQQ"
                  and r["short_strike"] == 590.0)
    other = next(r for r in rows if r is not target)
    target["credit"], other["credit"] = other["credit"], 1.41
    path.write_text(json.dumps(payload))

    c = vr.verify(dst)
    assert any(r["what"].endswith("credit") and "example 1" in r["what"]
               for r in c.failures), [r["what"] for r in c.failures]
    # and the value really is still somewhere in the file, i.e. a membership
    # check would have been satisfied
    assert "1.41" in path.read_text()


def test_verifier_fails_when_a_keyed_row_disappears(tmp_path):
    dst = _clone(tmp_path)
    path = dst / "diagnostics.json"
    payload = json.loads(path.read_text())
    bb = payload["samples"]["as_reported_close_sameday"]["band_bracket_audit"]
    bb["rows"] = [r for r in bb["rows"]
                  if not (r["entry_date"] == "2026-01-26"
                          and r["symbol"] == "QQQ")]
    path.write_text(json.dumps(payload))
    c = vr.verify(dst)
    assert any("artifact row exists" in r["what"] for r in c.failures)


def test_find_row_rejects_an_ambiguous_key():
    rows = [{"a": 1, "b": 1}, {"a": 1, "b": 2}]
    assert vr.find_row(rows, a=1) is None          # two hits -> not identified
    assert vr.find_row(rows, a=1, b=2) == rows[1]
    assert vr.find_row(rows, a=9) is None          # zero hits


def test_digk_handles_keys_containing_dots():
    # the credit-floor grid is keyed "0.25"; splitting on "." would return
    # None and silently turn a wrong number into an absent one
    obj = {"grid": {"0.25": {"rate": 0.5}}}
    assert vr.dig(obj, "grid.0.25.rate") is None
    assert vr.digk(obj, "grid", "0.25", "rate") == 0.5


def test_check_treats_a_missing_artifact_value_as_a_failure():
    c = vr.Claims()
    c.check("x", "missing", 1.0, None)
    c.check("x", "present", 1.0, 1.0)
    c.check("x", "expected-none", None, None)
    assert [r["ok"] for r in c.rows] == [False, True, True]
    assert len(c.failures) == 1


def test_check_respects_the_tolerance():
    c = vr.Claims()
    c.check("x", "within", 0.1234, 0.12341)
    c.check("x", "outside", 0.1234, 0.1250)
    assert [r["ok"] for r in c.rows] == [True, False]


def test_check_does_not_coerce_bools_to_numbers():
    c = vr.Claims()
    c.check("x", "true vs 1", True, 1)
    c.check("x", "false vs 0", False, 0)
    c.check("x", "true vs true", True, True)
    assert [r["ok"] for r in c.rows] == [False, False, True]


@pytest.mark.parametrize("section", ["1", "2c", "5g", "5i"])
def test_every_major_section_is_actually_covered(section):
    # a verifier that quietly stops checking a section is worse than none
    c = vr.verify(REPORT_DIR)
    assert any(r["section"] == section for r in c.rows)
