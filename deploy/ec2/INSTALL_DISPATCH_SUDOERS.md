# EC2 install — dispatch sudoers permission

The premarket scan dispatches candidate_entry by calling
`systemctl start trading-agent-candidate-entry@<TICKER>.service` from the
`ubuntu` user. Without an explicit sudoers entry that errors with
"Interactive authentication required" and the dispatch silently fails.

## Why we need this

Before the 5/14 fix, dispatch used `subprocess.Popen(..., start_new_session=True)`.
On 5/13 the dispatched SPY child (PID 96421) was SIGTERM'd by systemd
because `start_new_session` only changes the process group, NOT the cgroup;
when the parent `trading-agent-brain@premarket_scan.service` ExecStart
returned, systemd's default `KillMode=control-group` reaped the entire
cgroup including the still-importing Python interpreter for the child.

The fix is to start an independent unit (`trading-agent-candidate-entry@.service`)
per ticker — each gets its own cgroup, journal trace, and TimeoutStartSec.
Doing that from `ubuntu` requires this sudoers entry.

## Install (one-time, on EC2)

```bash
sudo tee /etc/sudoers.d/trading-agent-dispatch >/dev/null <<'EOF'
# Allow the trading-agent service user to start per-ticker dispatched units
# without a password. Restricted to the specific units used by dispatch.
ubuntu ALL=(root) NOPASSWD: /bin/systemctl start trading-agent-candidate-entry@*.service, /bin/systemctl reset-failed trading-agent-candidate-entry@*.service
EOF
sudo chmod 0440 /etc/sudoers.d/trading-agent-dispatch
sudo visudo -c -f /etc/sudoers.d/trading-agent-dispatch   # syntax check
```

Smoke test (should print nothing to stderr, exit 0):

```bash
sudo -n /bin/systemctl reset-failed trading-agent-candidate-entry@SPY.service
sudo -n /bin/systemctl start --no-block trading-agent-candidate-entry@SPY.service
echo "exit=$?"
```

## Inspecting a dispatched child

Because the child is now its own unit, debugging is straightforward:

```bash
# Live journal trace for the most recent SPY dispatch
sudo journalctl -u trading-agent-candidate-entry@SPY.service -f

# All units that ran today
systemctl list-units --all 'trading-agent-candidate-entry@*'

# Reset a stuck failed instance manually
sudo systemctl reset-failed trading-agent-candidate-entry@SPY.service
```

## Related files

- Service unit: `deploy/ec2/systemd/trading-agent-candidate-entry@.service`
- Dispatch code: `src/trading_agent/graph/nodes/premarket_nodes.py`
  → `_dispatch_candidate_entry_if_eligible`
- Watchdog (catches silent dies in ≤5 min): `src/trading_agent/graph/nodes/health_nodes.py`
  → `_check_dispatch_silent_die`
