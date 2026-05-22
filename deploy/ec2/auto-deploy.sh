#!/bin/bash
#
# Auto-deploy: pull origin/main and apply side-effects when needed.
#
# Invoked every 2 min by trading-agent-auto-deploy.timer. Idempotent —
# exits silently when HEAD already matches origin/main.
#
# DESIGN NOTES:
#
# 1. /etc/systemd/system/trading-agent-*.{service,timer} are SYMLINKS
#    pointing into deploy/ec2/systemd/ in this repo. That means `git pull`
#    alone updates the unit file content — we only need
#    `sudo systemctl daemon-reload` to make systemd notice. No copies, no
#    `sudo install`, no fragile arg-matching sudoers entries.
#
# 2. Sudoers: only `/bin/systemctl daemon-reload` (see
#    deploy/ec2/sudoers/trading-agent-deploy). Tight, single command.
#
# 3. RACE SAFETY: skips this tick if any brain orchestrator process is
#    currently mid-run. Next tick (2 min) retries. This prevents
#    `uv sync` from mutating .venv while a candidate_entry LLM chain is
#    still importing.
#
# 4. RECOVERY: each step is idempotent (uv sync, daemon-reload, migration
#    apply-if-not-applied). A failed deploy leaves a partial state; the
#    next tick continues from there. Operator gets an ntfy on the `ops`
#    topic for both success and failure.

set -euo pipefail

REPO_DIR="/home/ubuntu/trading-agent"
LOG_TAG="[auto-deploy $(date -u +%FT%TZ)]"
UV_BIN="/home/ubuntu/.local/bin/uv"

# ntfy helper — defined first so failure paths can use it. Best-effort,
# never raises. Reads NTFY_TOPIC_PREFIX from the systemd EnvironmentFile.
_notify_ops() {
    local body="$1"
    local prio="${2:-3}"
    if [ -z "${NTFY_TOPIC_PREFIX:-}" ]; then
        return 0
    fi
    curl -s --max-time 5 \
        -H "Title: trading-agent auto-deploy" \
        -H "Priority: $prio" \
        -H "Tags: rocket" \
        -d "$body" \
        "${NTFY_BASE_URL:-https://ntfy.sh}/${NTFY_TOPIC_PREFIX}-ops" > /dev/null 2>&1 || true
}

# Failure de-bouncer. Without this, a stuck-bad commit causes auto-deploy
# to retry every 2 min for 12 hours, sending 360+ ntfy "FAILED" pings —
# what happened on 5/22. Now: first failure pings priority=5 immediately;
# subsequent failures within the same SHA only ping once per hour at most.
# After 3 consecutive failures on the same SHA, the retry cadence stops
# being important — operator already knows it's broken.
FAILURE_STATE_DIR="/var/lib/trading-agent-auto-deploy"
_ntfy_failure_with_backoff() {
    local body="$1"
    mkdir -p "$FAILURE_STATE_DIR" 2>/dev/null || true
    local marker="$FAILURE_STATE_DIR/last_failure_sha_${REMOTE:0:7}"
    local count_marker="$FAILURE_STATE_DIR/failure_count_${REMOTE:0:7}"
    local now=$(date +%s)
    local last_ts=0
    local count=0
    if [ -f "$marker" ]; then
        last_ts=$(stat -c '%Y' "$marker" 2>/dev/null || echo 0)
    fi
    if [ -f "$count_marker" ]; then
        count=$(cat "$count_marker" 2>/dev/null || echo 0)
    fi
    count=$((count + 1))
    echo "$count" > "$count_marker"
    touch "$marker"
    local elapsed=$((now - last_ts))
    # First failure → priority 5. Subsequent failures on the same SHA within
    # 1 hour → silent. After 1 hour → priority 4 reminder.
    if [ "$last_ts" -eq 0 ]; then
        _notify_ops "$body (first occurrence)" 5
    elif [ "$elapsed" -gt 3600 ]; then
        _notify_ops "$body (still failing, attempt #$count)" 4
    fi
    # Otherwise: silent. Log it for audit.
    echo "$LOG_TAG failure #$count on ${REMOTE:0:7}; elapsed=${elapsed}s" >&2
}

trap '_notify_ops "auto-deploy unexpected exit on EC2 (exit $?)" 4' ERR

cd "$REPO_DIR"

# ---- Race guard: skip if a brain run is in progress ----------------------
if pgrep -f 'trading_agent.orchestrator' > /dev/null; then
    echo "$LOG_TAG brain run in progress, skipping this tick"
    exit 0
fi

# ---- Fetch + check whether anything is new -------------------------------
git fetch --quiet origin main

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    # Silent exit on no-op — this is the common case and would spam logs.
    exit 0
fi

echo "$LOG_TAG deploying ${LOCAL:0:7} -> ${REMOTE:0:7}"

# ---- Pre-pull test gate --------------------------------------------------
#
# Run pytest against the INCOMING revision before we actually move HEAD.
# Strategy: use a throwaway worktree pinned to origin/main so the live tree
# is never touched. If tests fail, we abort the deploy and ntfy — the live
# system keeps running on the current (presumably good) HEAD.
#
# Why a worktree instead of `git stash + checkout + run + revert`:
#   • Atomic: failure leaves zero side-effects on the live tree
#   • Fast: shares the .git dir, no new clone
#   • Composable with `set -e` — any failure path is auto-handled
#
# Why not just trust the dev's local pytest run: humans skip tests when
# rushed. CI gates are valuable precisely because they don't.

TEST_WORKTREE=$(mktemp -d -p /tmp trading-agent-deploy-test-XXXXXX)
trap 'rm -rf "$TEST_WORKTREE"' EXIT  # cleanup on any exit path

echo "$LOG_TAG pytest gate: worktree=$TEST_WORKTREE"
git worktree add --quiet --detach "$TEST_WORKTREE" "$REMOTE"

# Use the LIVE .venv (already has all deps installed from previous syncs).
# This is fine because:
#   • If the new commit added a dep that's not yet in .venv, the import will
#     fail in pytest and we abort — exactly what we want.
#   • Reusing .venv is ~30s faster than a fresh `uv sync` in the worktree.
#   • A real `uv sync` happens later in the deploy if pytest passes.
PYTHON_BIN="$REPO_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "$LOG_TAG ERROR live .venv missing $PYTHON_BIN" >&2
    _notify_ops "deploy ABORTED: .venv missing on EC2 (${REMOTE:0:7})" 5
    exit 1
fi

# `-m 'not integration'` skips tests that touch live infrastructure
# (real Postgres writes, real ntfy pushes, real LLM calls). Critical:
# without this, every retry of a failing deploy spam-fires EOD digests
# and heartbeats to production (5/22 incident — 40+ EOD LLM calls billed
# while the deploy retried itself to death).
# 4-min timeout (240s) inside systemd's 5-min TimeoutStartSec — give it
# slack to clean up before systemd SIGTERMs the whole tree.
if ! ( cd "$TEST_WORKTREE" && timeout 240 "$PYTHON_BIN" -m pytest tests/ -q -x --no-header -m 'not integration' 2>&1 | tail -30 ); then
    echo "$LOG_TAG ABORT: pytest failed on ${REMOTE:0:7}"
    _ntfy_failure_with_backoff "deploy ABORTED: pytest failed on ${REMOTE:0:7} — live HEAD ${LOCAL:0:7} kept"
    git worktree remove --force "$TEST_WORKTREE" 2>/dev/null || true
    exit 1
fi

git worktree remove --force "$TEST_WORKTREE" 2>/dev/null || true
trap - EXIT
echo "$LOG_TAG pytest gate PASSED"

# ---- Classify changed paths ----------------------------------------------
CHANGED_FILES=$(git diff --name-only "$LOCAL" "$REMOTE")
HAS_DEPS=$(echo "$CHANGED_FILES" | grep -E '^(pyproject\.toml|uv\.lock)$' || true)
HAS_SYSTEMD=$(echo "$CHANGED_FILES" | grep -E '^deploy/ec2/systemd/' || true)
HAS_MIGRATIONS=$(echo "$CHANGED_FILES" | grep -E '^migrations/.*\.sql$' || true)
HAS_SUDOERS=$(echo "$CHANGED_FILES" | grep -E '^deploy/ec2/sudoers/' || true)
# Long-running services that load Python modules at startup — need an explicit
# restart when their source files change (oneshot services like brain@.service
# don't, because they re-import on every fire).
HAS_DASHBOARD=$(echo "$CHANGED_FILES" | grep -E '^src/trading_agent/web/' || true)

NUM_CHANGED=$(echo "$CHANGED_FILES" | wc -l)
echo "$LOG_TAG changed=$NUM_CHANGED deps=${HAS_DEPS:+y} systemd=${HAS_SYSTEMD:+y} migrations=${HAS_MIGRATIONS:+y} sudoers=${HAS_SUDOERS:+y}"

# ---- Pull ---------------------------------------------------------------
git pull --ff-only origin main 2>&1 | tail -3

# ---- Post-pull conditional steps -----------------------------------------

if [ -n "$HAS_DEPS" ]; then
    # --extra dev keeps pytest + ruff + pytest-asyncio in the venv on EC2,
    # so the pre-deploy pytest gate (above) works. Without this, the first
    # deploy after a fresh setup aborts with "No module named pytest".
    echo "$LOG_TAG uv sync --extra dev"
    "$UV_BIN" sync --extra dev --quiet 2>&1 | tail -10
fi

if [ -n "$HAS_SYSTEMD" ]; then
    echo "$LOG_TAG systemctl daemon-reload (units are symlinks; content is already current)"
    sudo /bin/systemctl daemon-reload
fi

if [ -n "$HAS_DASHBOARD" ]; then
    # The dashboard is a long-running uvicorn — its source files are loaded
    # at process start, so file changes don't take effect without a restart.
    # `restart` is the right verb (not reload) because uvicorn doesn't watch
    # files in production mode.
    echo "$LOG_TAG restarting dashboard service (web source files changed)"
    sudo /bin/systemctl restart trading-agent-dashboard.service || {
        echo "$LOG_TAG WARN dashboard restart failed; investigate" >&2
        _notify_ops "dashboard restart FAILED after deploy ${REMOTE:0:7}" 4
    }
fi

if [ -n "$HAS_SUDOERS" ]; then
    # Sudoers changes need manual install — auto-deploy can't grant itself
    # new sudo rights for safety. Alert the operator.
    echo "$LOG_TAG SUDOERS CHANGED — manual install required: deploy/ec2/sudoers/"
    _notify_ops "auto-deploy: sudoers file changed in ${REMOTE:0:7} — install manually: sudo install -m 440 -o root -g root deploy/ec2/sudoers/* /etc/sudoers.d/" 4
fi

if [ -n "$HAS_MIGRATIONS" ]; then
    echo "$LOG_TAG applying pending migrations"
    # POSTGRES_DSN comes from the EnvironmentFile (.env.postgres).
    for sql in $(ls migrations/*.sql | sort); do
        ver=$(basename "$sql" .sql | grep -oE '^[0-9]+' || echo "")
        if [ -z "$ver" ]; then
            continue
        fi
        # Match either '010' or '010_watchlist_members' style — early
        # migrations used the long form, later ones use just the number.
        applied=$(psql "$POSTGRES_DSN" -tAc \
            "SELECT 1 FROM schema_migrations WHERE version = '$ver' OR version LIKE '${ver}_%' LIMIT 1" 2>&1)
        if [ -z "$applied" ]; then
            echo "$LOG_TAG  applying $(basename "$sql")"
            psql "$POSTGRES_DSN" -q -f "$sql" 2>&1 | tail -5
        fi
    done
fi

# ---- ntfy success notification ------------------------------------------
SUBJECT_LINE=$(git log -1 --pretty=format:'%s' "$REMOTE")
_notify_ops "deployed ${REMOTE:0:7}: ${SUBJECT_LINE:0:160}" 3

echo "$LOG_TAG done"
