"""Deployed systemd timer files — schedule contract tests.

The premarket timer must fire in America/New_York wall-clock so DST never
shifts the scans relative to the 09:30 ET open (the old fixed-UTC entries
drifted an hour every DST change). systemd's calendar-timezone support has
been verified to parse on the EC2 host (systemd 249).
"""
from __future__ import annotations

import re
from pathlib import Path

_SYSTEMD_DIR = Path(__file__).resolve().parent.parent / "deploy" / "ec2" / "systemd"


def _oncalendar_lines(unit_file: str) -> list[str]:
    text = (_SYSTEMD_DIR / unit_file).read_text()
    return [line.strip() for line in text.splitlines()
            if line.strip().startswith("OnCalendar=")]


class TestPremarketTimer:
    def test_three_fires_all_pinned_to_new_york(self):
        lines = _oncalendar_lines("trading-agent-premarket.timer")
        assert len(lines) == 3, f"expected 3 OnCalendar entries, got {lines}"
        pattern = re.compile(
            r"^OnCalendar=Mon\.\.Fri \*-\*-\* \d{2}:\d{2}:\d{2} America/New_York$"
        )
        for line in lines:
            assert pattern.match(line), f"not ET-pinned weekday schedule: {line}"

    def test_schedule_times(self):
        lines = _oncalendar_lines("trading-agent-premarket.timer")
        times = {re.search(r"(\d{2}:\d{2}:\d{2})", line).group(1) for line in lines}
        assert times == {"08:30:00", "10:15:00", "13:30:00"}, (
            f"digest 08:30 ET + execute 10:15/13:30 ET expected, got {times}"
        )

    def test_no_utc_pinned_entries_remain(self):
        for line in _oncalendar_lines("trading-agent-premarket.timer"):
            assert " UTC" not in line, f"fixed-UTC schedule reintroduced: {line}"

    def test_timer_still_targets_premarket_brain_unit(self):
        text = (_SYSTEMD_DIR / "trading-agent-premarket.timer").read_text()
        assert "Unit=trading-agent-brain@premarket_scan.service" in text
        assert "Persistent=true" in text

    def test_install_script_still_copies_the_timer(self):
        script = (_SYSTEMD_DIR / "install_timers.sh").read_text()
        assert "trading-agent-premarket.timer" in script
