# EC2 Dublin Deployment — Phase 1 Headless Jobs

This directory contains the deployment recipe for the Dublin t4g.small (or
any arm64 Ubuntu 24.04 box) that runs the headless EDGAR + premarket jobs
and syncs its DB back to the Mac. **The EC2 box does NOT run OpenD** — only
the Mac places paper orders. EC2 is read-only to the market (SEC only).

## One-time setup

```bash
# On the EC2 box:
sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv rsync git
git clone <your-repo-url> ~/trading-agent
cd ~/trading-agent
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .                     # installs moomoo-mcp, edgar-mcp, journal-mcp entry points
# On a headless box, moomoo-api pulls in no GUI deps — it's fine.
# If you want to strip moomoo-api from EC2 entirely, add it to optional-dependencies
# and install with:   pip install -e '.[ec2]'
cp .env.example .env
# edit .env:
#   SEC_UA_EMAIL=you@example.com            (required by SEC — real address)
#   DB_PATH=/home/ubuntu/trading-agent/data/trader.ec2.db
#   WATCHLIST_TICKERS=AAPL,MSFT,NVDA,GOOGL,META,TSLA,AMZN,SPY,QQQ,IWM
```

## Cron

Edit `crontab -e` on EC2:

```cron
# Overnight EDGAR scan — 02:00 UTC daily
0 2 * * 1-5 cd /home/ubuntu/trading-agent && /home/ubuntu/trading-agent/.venv/bin/python -m trading_agent.jobs.overnight_edgar_scan >> /home/ubuntu/trading-agent/data/logs/overnight.log 2>&1

# Premarket watchlist — 12:00 UTC (08:00 ET) weekdays
0 12 * * 1-5 cd /home/ubuntu/trading-agent && /home/ubuntu/trading-agent/.venv/bin/python -m trading_agent.jobs.premarket_watchlist >> /home/ubuntu/trading-agent/data/logs/premarket.log 2>&1
```

## DB sync — Mac side

On the Mac, add to `crontab -e`:

```cron
# Pull EC2 DB twice a day, merge into local trader.db
# Replace $PROJECT_DIR with the absolute path to your local clone.
30 2 * * 1-5 cd $PROJECT_DIR && $PROJECT_DIR/.venv/bin/python -m trading_agent.jobs.db_sync --remote ubuntu@dublin.example.com --remote-db ~/trading-agent/data/trader.ec2.db >> data/logs/db_sync.log 2>&1
30 12 * * 1-5 cd $PROJECT_DIR && $PROJECT_DIR/.venv/bin/python -m trading_agent.jobs.db_sync --remote ubuntu@dublin.example.com --remote-db ~/trading-agent/data/trader.ec2.db >> data/logs/db_sync.log 2>&1
```

Replace `dublin.example.com` with the actual host or IP. The Mac crontab
runs on the local user — make sure your ssh key is in the EC2 authorized_keys.

## Verifying it works

```bash
# On EC2 — run the overnight job manually once
cd ~/trading-agent
.venv/bin/python -m trading_agent.jobs.overnight_edgar_scan

# Expect JSON output; new rows in the local ec2 DB
sqlite3 data/trader.ec2.db \
  "SELECT id, topic, tags FROM notes WHERE tags LIKE 'overnight_scan%' ORDER BY id DESC LIMIT 5;"

# On Mac — pull the EC2 DB and merge
.venv/bin/python -m trading_agent.jobs.db_sync \
  --remote ubuntu@dublin.example.com \
  --remote-db ~/trading-agent/data/trader.ec2.db
```

Success output has `"inserted": N, "skipped": 0` on first run for the day,
`"inserted": 0, "skipped": N` on re-runs. `INSERT OR IGNORE` dedup keeps it
idempotent — you can run db_sync every 15 min safely.

## What we explicitly don't sync

- `trades`, `market_snapshots` — only the Mac places orders; these tables
  would be empty on EC2 anyway.
- `notes_vec` — embeddings are regenerated Mac-side because EC2 (t4g.small)
  doesn't have sentence-transformers installed. If you want EC2-side
  embedding too, `pip install sentence-transformers` there (~500 MB).

## Hardening later (Phase 2)

- Move cron to systemd timers for cleaner logging.
- Gzip filings_cache/ weekly to keep disk usage bounded.
- Add a CloudWatch alarm on "job hasn't completed in 60 min."
- Run db_sync via a SSH bastion instead of direct port 22 exposure.
