# Phase 1 Handoff — 2026-04-22

The Phase-1 plan is **functionally complete and proven**. Everything that can be verified
without you clicking anything has been verified headless; the one remaining
piece is for Claude Code's own process to be restarted so it re-reads
`.claude/settings.json`.

## What's proven right now (headless, no restart needed)

All of this was validated in this session by running the actual code paths
end-to-end:

| Layer | Evidence |
|---|---|
| **3 MCP servers boot under the same stdio launcher Claude Code uses** | `scripts/smoke.py` — tool lists match expected set (moomoo 11, edgar 5, journal 11) |
| **moomoo-mcp against live paper account** | Real quote `US.AAPL $266.35`, option chain snapshot 6 strikes exp 2026-04-22 — via MCP JSON-RPC |
| **edgar-mcp against live SEC** | 5 recent AAPL filings with UA header + semaphore + file cache |
| **journal-mcp RAG loop end-to-end** | `record_thesis` → vec search (MiniLM 384-d) → top-1 hit `lesson-earnings_long_call-IV_crush` for query "IV crush earnings long calls" |
| **5 seeded historical lessons** | ids 2–6 in `data/trader.db` (`source=post_mortem`), each with embedding in `notes_vec` |
| **Defense layer L1 — TrdEnv.SIMULATE hardcoded** | Day 14 audit checks A1/A2, plus module-level `assert` on every order path |
| **Defense layer L2 — reject_real_env hook** | Subprocess-exercised: `trd_env=REAL` → exit 2, `trading_password` → exit 2, `TrdEnv.REAL` substring → exit 2, clean payload → exit 0 (Day 14 B1–B6) |
| **Defense layer L3 — pretool_order_guard** | Subprocess: no thesis → exit 2, fresh thesis → exit 0 (bug fix on stale-thesis SQL comparison verified — `timedelta` cutoff parsing in Python, not SQL lex compare) |
| **posttool_fill_capture** | Synthetic FILLED payload → trades row inserted, non-blocking on garbage JSON (Day 14 F1–F3) |
| **Day-14 full audit** | 22/22 checks PASS across 7 categories |
| **Option underlying parse** | OCC Moomoo format `US.AAPL260515C265000` → underlying AAPL → matched against thesis.ticker (Day 14 E) |
| **Position-sizing hard rules (R1–R6)** | Unit-tested: 2% single-trade risk, ≤5 concurrent, 10% single-name combined, sector correlation gate, option DTE/delta windows, earnings lockout (Day 14 D) |
| **EC2 jobs** | `overnight_edgar_scan` + `premarket_watchlist` tested end-to-end with `DB_PATH=/tmp/ec2_job_test.db WATCHLIST_TICKERS="AAPL,SPY"` — picked up AAPL's real 8-K from 2026-04-20, wrote `premarket-*` notes |
| **db_sync idempotency** | `/tmp/test_db_sync.py` — first merge inserts 3 notes + 1 thesis; second merge inserts 0, skips all (composite-key dedup works) |

## No restart needed — tool loop proven headlessly too

I spawned fresh `claude -p` subprocesses with the project's MCP config
and verified the full CC dispatch path end-to-end without you touching
anything. See `scripts/verify_cc_loop.py`. Two checks, both PASS:

1. **Happy path** — a fresh CC session spawned moomoo-mcp, discovered
   `mcp__moomoo-mcp__get_quote` via ToolSearch (with retry during
   "pending" state), called it, got real AAPL last price `$267.33`.
2. **Hook block** — a fresh CC session tried
   `mcp__moomoo-mcp__place_paper_order` with `symbol='US.ZZZTEST'` +
   `thesis_id=99999`; `pretool_order_guard.py` exit 2 surfaced in CC's
   `permission_denials[]` and the model correctly reported BLOCKED.

This is the same tool loop the desktop app runs — same binary, same
MCP protocol, same hook dispatcher. The restart-versus-new-conversation
question is now just a UX question, not a correctness question.

## Two one-command checks

```bash
cd $PROJECT_DIR
.venv/bin/python scripts/smoke.py          # 17 PASS / 0 FAIL — ~25s
.venv/bin/python scripts/verify_cc_loop.py #  2 PASS / 0 FAIL — ~3min
```

`smoke.py` validates MCP + hooks directly via the MCP SDK stdio client +
subprocess; `verify_cc_loop.py` validates the real `claude -p` tool loop.
Both have been run and pass at the time of this handoff.

## For interactive use in the CC desktop app

Start a **new conversation** in the project directory. That's all
"restart" means for settings.json pickup — no need to quit the desktop
app itself. Then:

```
/research AAPL
```

You should see: live quote → option chain → recent filings → Form 4 →
past-lesson RAG hits (at least the 5 seeded ones).

## What's NOT done (deliberately deferred, all in plan as Phase-2+)

- **EC2 deployment itself** — `deploy/README_EC2.md` has the recipe (apt
  install, `.env`, crontab, Mac-side `db_sync`), but nothing is running
  on EC2 yet. Run the recipe when you want overnight_edgar to start
  emailing-free.
- **Real option orders via Moomoo paper API — PROVEN 2026-04-21** via
  `scripts/option_paper_dry_run.py`. Submitted `SPY 2026-05-05 710C`
  BUY 1 @ $1.48, order_id `3040008` accepted by Moomoo paper
  (`order_status=SUBMITTING`). PostToolUse hook inserted
  `trades(1)` + `market_snapshots(1)`, cancel cleaned up. Virtual-fill
  fallback is therefore the contingency path, not the primary one.
  Caveat logged in `notes`: Moomoo option snapshots do NOT expose
  bid/ask — only `option_premium` (mark). Sizing + /enter must use
  mark for limit price.
- **`/weekly-post-mortem` with real paper data** — Day 13 ran the full
  flow on a 7-trade synthetic week. Real first run needs 5+ real paper
  trades in `data/trader.db`.
- **Backtesting factory, Polygon feed, subagents** — Phase-2 per plan.

## File map for this Phase-1 delivery

```
$PROJECT_DIR/
├── PHASE1_HANDOFF.md                   ← this file
├── scripts/smoke.py                    ← post-restart one-command check
├── deploy/README_EC2.md                ← EC2 deployment recipe
├── src/trading_agent/
│   ├── jobs/
│   │   ├── overnight_edgar_scan.py     ← EC2 02:00 UTC
│   │   ├── premarket_watchlist.py      ← EC2 12:00 UTC
│   │   └── db_sync.py                  ← Mac-side, pulls EC2 DB + merges
│   ├── hooks/
│   │   ├── pretool_order_guard.py      ← thesis freshness + R1–R6 (bug-fixed this session)
│   │   ├── posttool_fill_capture.py
│   │   └── reject_real_env.py
│   └── mcp_servers/{moomoo,edgar,journal}/server.py
└── data/trader.db                      ← 5 seeded lessons (ids 2–6) +
                                         their embeddings in notes_vec
```

## Open questions I resolved autonomously (flagging for you)

1. **Stale-thesis gate was lexically broken.** Day 14 audit check C3
   caught it — SQL compared ISO-8601 tz-aware strings
   (`'2026-04-22T01:10:00+00:00'`) vs SQLite's naive
   `datetime('now', '-10 minutes')` → `'2026-04-22 01:20:00'`. Because
   `'T' > ' '` in ASCII, any open thesis lex-sorted as fresh. Fix:
   parse `created_at` in Python with `datetime.fromisoformat()` and
   compare against `timedelta`-derived cutoff. You may want to add a
   second belt-and-braces check that sorts by `created_at DESC LIMIT 1`
   and rejects anything past cutoff regardless of Python parse failure.
2. **Defense-in-depth layer clarity.** `reject_real_env` is a
   Claude-Code-hook-level guard; direct MCP calls (scripts, tests) don't
   trigger it. The `TrdEnv.SIMULATE` constant in `moomoo/server.py` is
   the hard floor — no path in the code lets a REAL env escape. If you
   ever add a script that calls moomoo-mcp directly, it still can't
   place a real order because the server never builds a non-SIMULATE
   context. Future hardening: add a startup banner log line that
   asserts `PAPER_ENV == TrdEnv.SIMULATE`.
3. **MCP binaries were missing until `uv sync`.** `pip install -e .`
   doesn't always materialize `[project.scripts]` entry points depending
   on the pip version. `uv sync` is idempotent and does it reliably;
   `deploy/README_EC2.md` uses `pip install -e .` but if you hit
   "command not found: moomoo-mcp" on EC2, swap to `uv sync`.

## Day-by-day status against the plan

| Day | Task | Status |
|---|---|---|
| 1 | uv init, config, db, schema; FutuOpenD paper | DONE |
| 2 | futu-mcp read-only + CC smoke | DONE |
| 3 | futu-mcp order tools + SIMULATE + reject_real_env | DONE |
| 4 | edgar-mcp + AAPL/TSLA validation | DONE |
| 5 | journal-mcp theses/fills/notes/search | DONE |
| 6 | sizing + pretool + posttool; end-to-end dry-run | DONE (bug fixed this session) |
| 7 | skills + /research + /size + /enter | DONE |
| 8 | SPY stock paper single | DONE |
| 9 | single-leg option paper; virtual-fill fallback | DONE |
| 10 | EC2 jobs + db_sync + README | DONE (not yet deployed) |
| 11 | /scan + /eod-review + earnings-play | DONE |
| 12 | post-mortem skill + 5 seed lessons | DONE |
| 13 | /weekly-post-mortem end-to-end | DONE (synthetic data) |
| 14 | audit suite — bypass attempts, real-env, rate limit | 22/22 PASS |

## Next natural step (when you're ready)

Restart Claude Code → run `scripts/smoke.py` → try `/research AAPL`. If
all three look good, we're green to do our first real paper trade from
inside CC using `/enter`. After that: let it run for a week, then
`/weekly-post-mortem` on real data and freeze Phase 1.
