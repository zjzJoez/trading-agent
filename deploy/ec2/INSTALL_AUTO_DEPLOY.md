# Auto-deploy from origin/main — one-time setup

After this, every `git push origin main` triggers a deploy on EC2 within
2 min. No manual `ssh + scp + daemon-reload` rituals.

## What it does on each tick

The `trading-agent-auto-deploy.timer` fires every 2 min. The service:

1. Checks `pgrep trading_agent.orchestrator` — if a brain run is active,
   exits and waits for the next tick (race-safety for `uv sync`).
2. `git fetch origin main`. If HEAD == origin/main, exits silently.
3. Otherwise classifies the diff:
   - `pyproject.toml` or `uv.lock` changed → run `uv sync`
   - `deploy/ec2/systemd/` changed → `sudo systemctl daemon-reload`
     (unit files are symlinks into the repo, so content is already current)
   - `migrations/*.sql` changed → apply any not yet in `schema_migrations`
   - `deploy/ec2/sudoers/` changed → ntfy the operator (auto-deploy can't
     grant itself new sudo rights for safety; install manually)
4. `git pull --ff-only`.
5. ntfy push on the `ops` topic with the new commit subject.

## One-time install on EC2

### 1. Make systemd unit files symlinks into the repo

```bash
cd /home/ubuntu/trading-agent

# Replace any existing /etc/systemd/system/trading-agent-*.{service,timer}
# with symlinks. Backs up the originals first.
sudo mkdir -p /etc/systemd/system/trading-agent.bak
for f in deploy/ec2/systemd/*.service deploy/ec2/systemd/*.timer; do
    name=$(basename "$f")
    if [ -e "/etc/systemd/system/$name" ] && [ ! -L "/etc/systemd/system/$name" ]; then
        sudo mv "/etc/systemd/system/$name" "/etc/systemd/system/trading-agent.bak/"
    fi
    sudo ln -sf "/home/ubuntu/trading-agent/$f" "/etc/systemd/system/$name"
done

sudo systemctl daemon-reload
```

### 2. Install the auto-deploy sudoers grant

```bash
sudo install -m 440 -o root -g root \
    /home/ubuntu/trading-agent/deploy/ec2/sudoers/trading-agent-deploy \
    /etc/sudoers.d/trading-agent-deploy
```

### 3. Enable the auto-deploy timer

```bash
sudo systemctl enable --now trading-agent-auto-deploy.timer
```

### 4. Verify

```bash
# Should show next fire time
systemctl list-timers | grep auto-deploy

# Should run cleanly with no changes (HEAD already matches origin/main)
sudo systemctl start trading-agent-auto-deploy.service
sudo journalctl -u trading-agent-auto-deploy.service -n 10
```

## Operating notes

- **Log location**: `/var/log/trading-agent-auto-deploy.log` (append-only)
- **Manual deploy**: `sudo systemctl start trading-agent-auto-deploy.service`
- **Disable temporarily**: `sudo systemctl stop trading-agent-auto-deploy.timer`
- **Forced re-pull**: just `cd /home/ubuntu/trading-agent && git pull` —
  but better to push a real commit so the audit trail shows what changed.

## Failure modes & recovery

| Failure | Effect | Recovery |
|---|---|---|
| `git pull` conflict | Deploy halts mid-step | Operator fixes conflict manually; next tick continues |
| `uv sync` network error | Deploy halts at uv step | Next tick retries; ntfy alert fires |
| migration SQL fails | One migration unapplied | Fix the migration, push again; auto-deploy retries idempotently |
| daemon-reload fails | Old units still active | Inspect with `journalctl -u trading-agent-auto-deploy` |
| race with brain run | Tick skipped | Next tick (2 min later) tries again |

The script has `trap ... ERR` that fires an ntfy on any failure, so the
operator finds out within seconds.
