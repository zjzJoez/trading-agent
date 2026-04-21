"""At-a-glance trading agent status.

Shows:
  - launchd job state (overnight-scan, premarket-brief): loaded? last exit?
  - last successful run timestamp (from launchd log file mtime)
  - DB inventory: theses, trades, notes (by source), post_mortems
  - recent notes from the last 7 days (counts by tag)
  - embedding coverage (notes_vec rows vs notes rows)
  - orders-in-flight count (if OpenD up)

Run: .venv/bin/python scripts/status.py

Exit 0 always. This is a read-only report, never mutates state.
"""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "trader.db"
LOGS = ROOT / "data" / "logs"
JOBS = [
    ("com.example.trading-agent.overnight-scan", "overnight-scan"),
    ("com.example.trading-agent.premarket-brief", "premarket-brief"),
    ("com.example.trading-agent.backfill-embeddings", "backfill-embeddings"),
]

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def banner(txt: str) -> None:
    print(f"\n{BOLD}── {txt} ──{RESET}")


def fmt_age(ts: float) -> str:
    if ts == 0:
        return f"{DIM}never{RESET}"
    age = datetime.now(timezone.utc).timestamp() - ts
    if age < 60:
        return f"{int(age)}s ago"
    if age < 3600:
        return f"{int(age/60)}m ago"
    if age < 86400:
        return f"{int(age/3600)}h ago"
    return f"{int(age/86400)}d ago"


def launchd_row(label: str) -> tuple[str, str]:
    """Return (pid, last_exit) or ('-','MISSING')."""
    try:
        out = subprocess.check_output(["launchctl", "list"], text=True, timeout=5)
    except Exception as e:
        return ("-", f"err({e})")
    for line in out.splitlines():
        cols = line.split()
        if len(cols) >= 3 and cols[2] == label:
            return (cols[0], cols[1])
    return ("-", "MISSING")


def last_log_mtime(short: str, kind: str) -> float:
    p = LOGS / f"{short}.{kind}.log"
    try:
        return p.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def section_launchd() -> None:
    banner("Scheduled jobs (launchd)")
    fmt = f"  {{:<30}} status={{:<8}} last_exit={{}}  stdout={{}}  stderr={{}}"
    for label, short in JOBS:
        pid, last = launchd_row(label)
        if last == "MISSING":
            status = f"{RED}NOT LOADED{RESET}"
        elif last == "0":
            status = f"{GREEN}OK{RESET}"
        elif pid != "-":
            status = f"{YELLOW}RUNNING{RESET}"
        else:
            status = f"{RED}FAIL{RESET}"
        print(fmt.format(
            short, status,
            f"{last:<3}",
            fmt_age(last_log_mtime(short, "out")),
            fmt_age(last_log_mtime(short, "err")),
        ))


def section_db() -> None:
    banner(f"SQLite inventory — {DB_PATH}")
    if not DB_PATH.exists():
        print(f"  {RED}DB MISSING{RESET}")
        return
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row

    theses = c.execute("SELECT COUNT(*) n, SUM(status='open') o FROM theses").fetchone()
    print(f"  theses:        total={theses['n']}  open={theses['o'] or 0}")

    tr = c.execute(
        "SELECT COUNT(*) n, "
        "SUM(outcome='OPEN') o, SUM(outcome='WIN') w, SUM(outcome='LOSS') l "
        "FROM trades"
    ).fetchone()
    print(f"  trades:        total={tr['n']}  open={tr['o'] or 0}  "
          f"W={tr['w'] or 0}/L={tr['l'] or 0}")

    pm = c.execute("SELECT COUNT(*) n, MAX(created_at) last FROM post_mortems").fetchone()
    print(f"  post_mortems:  total={pm['n']}  "
          f"last={pm['last'] or f'{DIM}never{RESET}'}")

    # Notes by source
    print("  notes by source:")
    for r in c.execute(
        "SELECT source, COUNT(*) n FROM notes "
        "GROUP BY source ORDER BY n DESC"
    ):
        print(f"    {r['source'] or '(null)':<16} {r['n']}")

    # Embedding coverage — vec0 extension must be loaded for notes_vec.
    try:
        import sqlite_vec  # type: ignore
        c.enable_load_extension(True)
        sqlite_vec.load(c)
        c.enable_load_extension(False)
        nvec = c.execute("SELECT COUNT(*) FROM notes_vec").fetchone()[0]
        nnotes = c.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        covered = (nvec / nnotes * 100) if nnotes else 0
        col = GREEN if covered > 80 else YELLOW if covered > 50 else RED
        print(f"  embeddings:    {col}{nvec}/{nnotes} ({covered:.0f}%){RESET}")
    except Exception as e:
        print(f"  embeddings:    {YELLOW}vec0 load failed: {e}{RESET}")

    c.close()


def section_recent() -> None:
    banner("Notes in last 7 days")
    if not DB_PATH.exists():
        return
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    rows = c.execute(
        "SELECT substr(tags, 1, instr(tags||',', ',')-1) AS first_tag, "
        "COUNT(*) n "
        "FROM notes "
        "WHERE datetime(created_at) >= datetime('now', '-7 days') "
        "GROUP BY first_tag ORDER BY n DESC"
    ).fetchall()
    if not rows:
        print(f"  {DIM}(no recent notes){RESET}")
    else:
        for r in rows:
            print(f"  {r['first_tag'] or '(untagged)':<30} {r['n']}")
    c.close()


def section_opend() -> None:
    banner("MoomooOpenD")
    try:
        with socket.create_connection(("127.0.0.1", 11111), timeout=1):
            print(f"  {GREEN}up on 127.0.0.1:11111{RESET}")
    except OSError:
        print(f"  {RED}down — paper trading will error until OpenD is running{RESET}")
        return
    # Quick order count via direct moomoo API
    try:
        from moomoo import OpenSecTradeContext, TrdEnv, TrdMarket, RET_OK  # type: ignore
        trd = OpenSecTradeContext(filter_trdmarket=TrdMarket.US)
        ret, df = trd.order_list_query(trd_env=TrdEnv.SIMULATE)
        trd.close()
        if ret == RET_OK:
            live = df[df["order_status"].isin(["SUBMITTED", "WAITING_SUBMIT", "SUBMITTING"])]
            print(f"  open paper orders: {len(live)}")
            if len(live) > 0:
                print(f"  {DIM}{live[['code','trd_side','qty','price']].to_string(index=False)}{RESET}")
    except Exception as e:
        print(f"  {YELLOW}order query failed: {e}{RESET}")


def section_audit() -> None:
    banner("Recent hook audit (last 5)")
    p = ROOT / "data" / "hook_audit.log"
    if not p.exists():
        print(f"  {DIM}(no audit log yet){RESET}")
        return
    lines = p.read_text().splitlines()[-5:]
    for ln in lines:
        try:
            o = json.loads(ln)
            print(f"  {o.get('ts','?')[:19]}  "
                  f"{o.get('decision','?'):<6}  "
                  f"{o.get('tool','?').replace('mcp__moomoo-mcp__','')}  "
                  f"{o.get('reason','')[:80]}")
        except Exception:
            print(f"  {ln[:120]}")


def main() -> int:
    print(f"{BOLD}Trading agent status{RESET}  {DIM}({datetime.now().isoformat(timespec='seconds')}){RESET}")
    section_launchd()
    section_db()
    section_recent()
    section_audit()
    section_opend()
    return 0


if __name__ == "__main__":
    sys.exit(main())
