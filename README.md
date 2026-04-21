<div align="center">

# `trading-agent`

**A senior-trader-grade, paper-only, self-learning stock & options agent — built on Claude Code.**

*一个在 Claude Code 里跑的、纸面交易、能从自己历史复盘里学习的美股 / 期权交易代理。*

<br/>

![python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9)
![sqlite--vec](https://img.shields.io/badge/RAG-sqlite--vec%20%2B%20MiniLM-4A90E2)
![claude--code](https://img.shields.io/badge/built%20on-Claude%20Code-D97757)
![paper--only](https://img.shields.io/badge/paper--only-by%20design-success)
![license](https://img.shields.io/badge/license-MIT-blue)

</div>

---

## What this is

A `Claude Code` agent that behaves like a disciplined swing trader:

1. **Researches** a name before touching it — live quote, option chain, recent 10-Q / 8-K / Form 4 from SEC EDGAR, plus past trades the agent itself took on the same setup (RAG hits against its own journal).
2. **Writes a thesis** — invalidation, timeframe, expected return, max loss — *before* any order. A Claude Code `PreToolUse` hook **physically blocks** the order tool if no thesis exists within a 10-minute window. You cannot talk the model out of it; the hook is a subprocess that returns exit code 2.
3. **Sizes the position** against six numerical rules (single-trade risk ≤ 2%, correlated-sector gate, earnings lockout, etc.) — and the hook re-validates those rules in a separate Python process that has no model context, so a sloppy LLM can't sneak past them.
4. **Places the paper order** through `moomoo-mcp` → MoomooOpenD (free paper broker).
5. **Captures the fill** into SQLite (`trades` + `market_snapshots`) via a `PostToolUse` hook.
6. **Runs a weekly post-mortem** that aggregates closed trades by strategy, extracts 3–8 lessons, and **embeds them into a vector index**. The next time you `/research` a similar setup, those lessons are pulled back into context automatically. The agent gets better at the things it has been bad at — no fine-tuning, pure RAG.

> [!NOTE]
> **Phase 1 MVP.** Paper-only. **Zero monthly cost.** Broker API free (Moomoo OpenD), data free (SEC EDGAR + Moomoo L1 snapshot), embeddings free (local MiniLM). SQLite is the source of truth. No cloud dependencies required, no paid data vendor.

## The "aha" — how the agent literally cannot place a real order

Three independent layers, each sufficient on its own:

<table>
<tr>
<th>Layer</th><th>Where</th><th>What it does</th>
</tr>
<tr>
<td><b>L1 · Code constant</b></td>
<td><code>mcp_servers/moomoo/server.py</code></td>
<td><code>PAPER_ENV: TrdEnv = TrdEnv.SIMULATE</code> is a module-level constant. Every order path starts with <code>assert PAPER_ENV == TrdEnv.SIMULATE</code>. No Python codepath builds a non-SIMULATE trade context.</td>
</tr>
<tr>
<td><b>L2 · Tool signature</b></td>
<td>MCP tool defs</td>
<td><code>place_paper_order</code> and <code>place_paper_option_order</code> <i>do not expose</i> <code>trd_env</code> as a parameter. FastMCP + pydantic reject unknown kwargs. The LLM has no syntactic way to request real.</td>
</tr>
<tr>
<td><b>L3 · PreToolUse hook</b></td>
<td><code>hooks/reject_real_env.py</code></td>
<td>Scans every tool call's JSON payload for <code>trd_env</code>, <code>REAL</code>, <code>trading_password</code>, <code>trd_password</code>, etc. Match → exit code 2 → Claude Code refuses the call. Runs in a fresh subprocess with no model context.</td>
</tr>
</table>

Both the smoke test (`scripts/smoke.py`, 17/17 PASS) and the dedicated audit (22/22 PASS) exercise attempted bypasses and confirm all three layers trigger correctly.

## Architecture

```mermaid
flowchart LR
    CC[Claude Code<br/>desktop / CLI] -->|stdio| MCP1[moomoo-mcp]
    CC -->|stdio| MCP2[edgar-mcp]
    CC -->|stdio| MCP3[journal-mcp]
    CC -.hook.->|PreToolUse| H1[reject_real_env]
    CC -.hook.->|PreToolUse| H2[pretool_order_guard<br/>thesis + sizing R1-R6]
    CC -.hook.->|PostToolUse| H3[posttool_fill_capture]

    MCP1 -->|TCP :11111| OD[MoomooOpenD<br/>paper account]
    MCP2 -->|HTTPS 10 req/s| SEC[data.sec.gov]
    MCP3 --> DB[(SQLite<br/>trader.db)]
    H2 --> DB
    H3 --> DB

    subgraph Mac["Mac (source of truth)"]
        CC
        MCP1
        MCP3
        OD
        DB
    end

    subgraph EC2["Dublin EC2 (headless jobs)"]
        CRON[cron · overnight_edgar_scan<br/>cron · premarket_watchlist<br/>cron · weekly_post_mortem]
        DB2[(trader.ec2.db)]
        CRON --> DB2
        CRON --> MCP2
    end

    DB2 -->|rsync + INSERT OR IGNORE| DB
    DB -->|MiniLM 384-d| VEC[(notes_vec<br/>sqlite-vec virtual table)]
    VEC -.->|semantic recall| CC
```

- **Mac** holds FutuOpenD + the writer of truth (`data/trader.db`). One Moomoo account = one OpenD session, so EC2 intentionally doesn't run the broker.
- **EC2** is stateless headless: overnight SEC scan, pre-market brief, weekly post-mortem — jobs that only need DB + SEC, not the broker. Output gets `rsync`'d to Mac and merged with a composite-key `INSERT OR IGNORE` dedup.
- **Vector recall** uses [sqlite-vec](https://github.com/asg017/sqlite-vec) `vec0` virtual tables with MiniLM-L6-v2 (384-dim, CPU-only). One `notes_vec` row per `notes` row. Post-mortem lessons → embedded → surfaced on next `/research` of a similar setup.

## The three MCP servers

| Server | Wraps | Exports |
|---|---|---|
| `moomoo-mcp` | `moomoo-api` (Moomoo OpenD) | `get_quote`, `get_option_chain`, `list_option_expiries`, `get_option_chain_snapshot`, `get_historical_kline`, `get_account_info`, `get_positions`, `place_paper_order`, `place_paper_option_order`, `cancel_paper_order`, `get_orders` |
| `edgar-mcp` | `data.sec.gov` | `search_filings`, `get_filing_text`, `get_recent_filings_for_ticker`, `get_insider_transactions` (Form 4), `get_institutional_holdings` (13F-HR) — with ticker→CIK cache, 10 req/s semaphore, 0.12 s post-delay, disk cache under `data/filings_cache/` |
| `journal-mcp` | self (SQLite + sqlite-vec) | `record_thesis`, `record_fill`, `record_virtual_fill`, `append_note`, `close_thesis`, `close_trade`, `search_past_trades` (semantic + lexical), `search_notes`, `list_open_theses`, `get_open_positions_with_thesis`, `generate_post_mortem_prompt` |

## Slash commands

```
/scan                              market scan — SPY/QQQ/VIX + watchlist
/research <TICKER>                 deep dive: quote, chain, filings, Form 4, past-lesson RAG
/size <TICKER> <DIRECTION>         confirms position against R1–R6, outputs qty/stop/target
/enter                             chains thesis → size → place_paper_order; any step fails, whole flow aborts
/eod-review                        open positions vs theses, flags invalidated ones
/weekly-post-mortem                aggregates 7-day closed trades, produces embedded lessons
```

Each command lives under [`.claude/commands/`](.claude/commands) as a `.md` file — not a black box, read or edit them.

## Position-sizing rules (enforced by hook, not by vibes)

The PreToolUse order-guard re-runs [`sizing.py`](src/trading_agent/sizing.py) in a separate process against the proposed order. Violate any rule → exit 2 → order blocked.

| # | Rule |
|---|---|
| R1 | Single-trade risk ≤ 2% of account equity. Stock: `\|entry − stop\| × qty`. Long option: `debit × contracts × 100`. |
| R2 | ≤ 5 concurrent open positions. |
| R3 | Single-name combined stock + option notional exposure ≤ 10% of equity. |
| R4 | Same-sector correlation gate — if two same-GICS-sector positions already open, new same-sector denied. Sector lookup via [`data/sectors.csv`](data/sectors.csv). |
| R5 | Options: long premium only. DTE ∈ [14, 60]. Delta ∈ [0.30, 0.55]. Single-leg from `/enter`. ≤ 1% of equity per option trade. |
| R6 | Earnings lockout — within 2 trading days of earnings, only `strategy_label` prefixed `earnings_*` is allowed. |

Because the hook re-validates in a fresh Python process with no model context, "please ignore the sizing rule just this once" gets you exit 2, not a carve-out.

## Quickstart

> [!IMPORTANT]
> Mac required for live paper trading (Moomoo OpenD is Mac/Windows/Linux GUI; this repo targets Mac). Python 3.12. Claude Code desktop or CLI.

```bash
# 1. Clone
git clone https://github.com/zjzJoez/trading-agent.git
cd trading-agent

# 2. One-command post-clone setup
#    - uv sync (creates .venv + wires the MCP entry points)
#    - generates .env from .env.example
#    - generates .claude/settings.json from .claude/settings.json.example
#      with your absolute project path substituted
#    - generates launchd .plist files from templates (macOS only)
bash scripts/setup.sh

# 3. Fill in .env (you MUST set a real SEC_UA_EMAIL — SEC requires it)
$EDITOR .env

# 4. Install + log into MoomooOpenD paper account
#    https://www.moomoo.com/download/OpenAPI (default port 11111)

# 5. Open this directory in Claude Code
#    Desktop: File > Open Folder > /path/to/trading-agent
#    CLI: cd /path/to/trading-agent && claude

# 6. Sanity check
.venv/bin/python scripts/smoke.py        # 17/17 PASS expected, ~25s
.venv/bin/python scripts/status.py       # dashboard: launchd jobs, DB inventory, hook audit
```

Then in Claude Code:

```
/research AAPL
/size AAPL LONG
/enter
```

Any of those three can fail loudly and stop the flow — that's the feature.

## Testing & audit

```bash
.venv/bin/python scripts/smoke.py                # 17 checks: MCP tools + hooks + RAG round-trip
.venv/bin/python scripts/verify_cc_loop.py       # 2 checks: fresh `claude -p` subprocess tool loop
.venv/bin/python scripts/option_paper_dry_run.py # live paper-options E2E (real broker call + hook capture)
.venv/bin/python scripts/status.py               # live dashboard
```

- `smoke.py` — validates all three MCP servers via real stdio JSON-RPC, exercises hook subprocesses with synthetic PreToolUse/PostToolUse payloads, round-trips a thesis → semantic search → embedding. Cleans its own test rows at the end.
- `verify_cc_loop.py` — spawns a fresh `claude -p --mcp-config …` subprocess and asserts the full tool loop works (including a bypass attempt that gets correctly blocked and surfaced via `permission_denials[]`). Proves Claude Code wiring without needing a human to restart anything.
- `option_paper_dry_run.py` — submits a real single-leg long-call order against Moomoo paper, pipes the response through the real `posttool_fill_capture.py` hook, verifies the `trades` + `market_snapshots` rows land, cancels the resting order.

## Directory layout

```
trading-agent/
├── pyproject.toml                                 uv-managed, single venv
├── .env.example                                   template for OPEND / SEC UA / DB path
├── .claude/
│   ├── settings.json.example                      gets generated → .claude/settings.json by setup.sh
│   ├── commands/*.md                              slash-command definitions
│   └── skills/*/SKILL.md                          6 skills: sizing, risk, options, earnings, journal, post-mortem
├── src/trading_agent/
│   ├── config.py                                  env-driven config (pydantic)
│   ├── db.py                                      SQLite + sqlite_vec.load()
│   ├── schema.sql
│   ├── sizing.py                                  pure functions, unit-testable — used by both /size and the hook
│   ├── mcp_servers/{moomoo,edgar,journal}/
│   ├── hooks/
│   │   ├── reject_real_env.py                     L3 defense against real-account keywords
│   │   ├── pretool_order_guard.py                 thesis freshness + R1–R6 numerical validation
│   │   └── posttool_fill_capture.py               inserts trades + market_snapshots post-fill
│   └── jobs/
│       ├── overnight_edgar_scan.py                EC2 cron, feeds notes
│       ├── premarket_watchlist.py                 EC2 cron
│       ├── backfill_embeddings.py                 Mac cron every 30 min — hydrates notes_vec
│       └── db_sync.py                             Mac-side rsync pull + idempotent merge
├── deploy/
│   ├── README_EC2.md                              Dublin EC2 recipe (apt install + crontab)
│   └── launchd/*.plist.template                   Mac cron templates
├── scripts/
│   ├── setup.sh                                   post-clone bootstrap
│   ├── smoke.py                                   one-command MCP + hook smoke test
│   ├── verify_cc_loop.py                          headless `claude -p` tool-loop check
│   ├── option_paper_dry_run.py                    live paper-options E2E
│   └── status.py                                  at-a-glance dashboard
└── data/
    ├── sectors.csv                                static GICS lookup (checked in)
    ├── trader.db                                  SQLite — gitignored
    ├── filings_cache/                             SEC response cache — gitignored
    ├── logs/                                      launchd + job logs — gitignored
    └── hook_audit.log                             every allow/block decision — gitignored
```

## Roadmap

### Phase 1 — MVP (this repo) ✓

- [x] 3 MCP servers (moomoo, edgar, journal)
- [x] 3-layer defense against real-money orders
- [x] PreToolUse thesis + sizing gate
- [x] PostToolUse fill capture
- [x] sqlite-vec + MiniLM RAG over past theses / lessons
- [x] 6 slash commands, 6 skills
- [x] Mac launchd automation (overnight scan, premarket brief, embedding backfill)
- [x] EC2 cron jobs + `db_sync.py` with idempotent merge
- [x] Paper stock order E2E proven (order `3039917`)
- [x] Paper **option** order E2E proven (SPY 710C, order `3040008`, 2026-04-21)
- [x] 22/22 Day-14 audit pass (hook bypass, real-env, rate-limit, parser)

### Phase 2 — post-MVP

- [ ] Backtesting factory: vectorbt + translate `notes` strategy descriptions to runnable code
- [ ] Polygon.io ($30/mo) for deeper history
- [ ] Independent `market-scanner` / `risk-manager` / `research-analyst` subagents
- [ ] Limited auto-real trading (0DTE iron condors, premium selling only) — **still with all three defense layers retained**, new gates added
- [ ] Multi-account (Phase 1 is single-account OpenD-bound)

## Design notes worth reading

- **Why "thesis before order" as a hook, not a prompt instruction?** Because the prompt instruction can be argued with. The hook is a subprocess that returns exit 2 and the model gets back a blocked-call signal. There is no natural-language path through.
- **Why `posttool_fill_capture` instead of doing the insert inside `place_paper_order`?** Separation of concerns. The MCP tool stays a thin transport wrapper; DB writes live in `trading_agent.db` / `journal-mcp`. The hook is also what Claude Code sees, so it fires even if the model later wants to disable the capture — it can't.
- **Why ship `settings.json.example` and gitignore the real one?** MCP server `command` fields in Claude Code don't reliably expand `${CLAUDE_PROJECT_DIR}`, but they do need absolute paths. `scripts/setup.sh` does the substitution at clone time — one command, no manual editing.
- **Why MiniLM 384-d instead of something bigger?** It runs on CPU in ~40ms per note, which means we don't pay a GPU tax or hit an external API on the critical path of every `append_note`. Recall quality is fine for a personal trade journal (validated in `smoke.py` — the semantic search returns the right earnings-IV-crush lesson).
- **Why not just fine-tune?** Because RAG on `notes` with embedded lessons is **legible, editable, and debuggable** — you can look at what the model retrieved and correct it. A fine-tune is a black box.

## Disclaimer

> [!WARNING]
> This is an educational research project. It is configured for **paper trading only**, and it goes out of its way (see the 3-layer defense above) to make real trading impossible without removing code on purpose.
>
> Nothing in this repo is investment advice. Markets are adversarial. Your paper P&L is not your live P&L. Slippage, fills, halts, taxes, borrow costs, and your own emotional state on the day of the trade are all missing from a paper environment.
>
> If you ever graduate to real capital, build your own risk framework with real professional review. The author accepts zero liability for any losses, broker disputes, or regulatory consequences resulting from use or misuse of this software.

## License

[MIT](LICENSE).

---

<div align="center">
<sub>Built one skill, one hook, one <code>INSERT OR IGNORE</code> at a time. Paper-only, by design.</sub>
</div>
