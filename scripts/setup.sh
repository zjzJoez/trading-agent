#!/usr/bin/env bash
# One-command post-clone setup.
#
# What this does:
#   1. Installs the Python toolchain with `uv sync` (creates .venv + wires up
#      the moomoo-mcp / edgar-mcp / journal-mcp console_scripts).
#   2. Generates `.env` from `.env.example` if missing, and prompts you to
#      edit SEC_UA_EMAIL.
#   3. Generates `.claude/settings.json` from its `.example` template by
#      substituting the absolute path to this repo in place of
#      `__PROJECT_DIR__`. Hook commands already use ${CLAUDE_PROJECT_DIR};
#      only MCP server commands need the explicit substitution because
#      Claude Code doesn't guarantee variable expansion in `mcpServers.command`.
#   4. Copies the launchd plist templates in deploy/launchd/ to real plists
#      under the same directory (you still have to `launchctl bootstrap`
#      them yourself if you want the overnight-scan cron on Mac).
#
# Safe to re-run — all operations are idempotent.

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"
echo "→ setting up in $PROJECT_DIR"

# ---- 1. Python env ----
if ! command -v uv >/dev/null 2>&1; then
  echo "✗ uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
echo "→ uv sync"
uv sync

# ---- 2. .env ----
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "→ created .env from .env.example"
  echo "  ⚠  edit .env and set SEC_UA_EMAIL to a real address you control"
else
  echo "→ .env already present, leaving alone"
fi

# ---- 3. .claude/settings.json ----
if [[ ! -f .claude/settings.json ]]; then
  sed "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
    .claude/settings.json.example > .claude/settings.json
  echo "→ generated .claude/settings.json (MCP paths pinned to $PROJECT_DIR)"
else
  echo "→ .claude/settings.json already present, leaving alone"
fi

# ---- 3b. .mcp.json ----
if [[ ! -f .mcp.json ]]; then
  sed "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
    .mcp.json.example > .mcp.json
  echo "→ generated .mcp.json (MCP server paths pinned to $PROJECT_DIR)"
else
  echo "→ .mcp.json already present, leaving alone"
fi

# ---- 4. launchd plists (macOS only, optional) ----
if [[ "$(uname -s)" == "Darwin" ]]; then
  for tpl in deploy/launchd/*.plist.template; do
    [[ -f "$tpl" ]] || continue
    out="${tpl%.template}"
    if [[ ! -f "$out" ]]; then
      # .env is sourced so SEC_UA_EMAIL can be substituted
      set -a; source .env; set +a
      sed \
        -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
        -e "s|__SEC_UA_EMAIL__|${SEC_UA_EMAIL:-you@example.com}|g" \
        "$tpl" > "$out"
      echo "→ generated $out"
    fi
  done
fi

echo
echo "✓ setup done. Next:"
echo "    1. Review .env (especially SEC_UA_EMAIL)"
echo "    2. Start MoomooOpenD (Mac) and log into the paper account"
echo "    3. Open this directory in Claude Code (File > Open Folder)"
echo "    4. Run ./scripts/smoke.py to verify (17/17 PASS expected)"
