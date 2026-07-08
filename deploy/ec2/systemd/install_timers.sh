#!/usr/bin/env bash
# Install + enable trading-agent systemd timers on EC2.
# Run as a normal user with sudo access. Idempotent.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
DEST=/etc/systemd/system
SUDO="sudo"

units=(
  "trading-agent-brain@.service"
  "trading-agent-candidate-entry@.service"
  "trading-agent-watchlist-refresh.service"
  "trading-agent-watchlist-refresh.timer"
  "trading-agent-premarket.timer"
  "trading-agent-intraday.timer"
  "trading-agent-eod.timer"
  "trading-agent-healthcheck.timer"
  "trading-agent-learning.timer"
  "trading-agent-chain-cache.service"
  "trading-agent-chain-cache.timer"
  "trading-agent-reconcile.service"
  "trading-agent-reconcile.timer"
)

for u in "${units[@]}"; do
  $SUDO cp -v "$DIR/$u" "$DEST/$u"
done

$SUDO systemctl daemon-reload

# Enable + start the timers (NOT the @-templates; they run on demand)
for t in trading-agent-watchlist-refresh trading-agent-premarket trading-agent-intraday trading-agent-eod trading-agent-healthcheck trading-agent-learning trading-agent-chain-cache trading-agent-reconcile; do
  $SUDO systemctl enable --now "$t.timer"
done

echo
echo "Active timers:"
$SUDO systemctl list-timers 'trading-agent-*' --all
