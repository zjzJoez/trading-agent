# Mobile dashboard via Cloudflare Tunnel — one-time setup

After this you can open `https://<your-name>.trycloudflare.com/mobile` on
your phone and see the dashboard. No EC2 ports open to the internet, free
HTTPS via Cloudflare edge, auto-reconnects on IP/network changes.

Why Cloudflare Tunnel instead of nginx + public IP + DNS + ACME:
- **0 firewall changes**: EC2 SG stays locked down
- **Auto TLS**: no certbot, no renewal cron, edge serves the cert
- **Auth-ready**: Cloudflare Access (free 50 users) gates by email/Google/etc.
- **Stable URL**: the tunnel name survives EC2 IP rotation

## One-time bootstrap

### 1. Install cloudflared on EC2

```bash
# Latest stable, single static binary
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -o /tmp/cloudflared
sudo install -m 755 -o root -g root /tmp/cloudflared /usr/local/bin/cloudflared
cloudflared --version
```

### 2. Authenticate (interactive, one-time)

```bash
cloudflared tunnel login
# → opens browser to cloudflare.com, you log in, cert.pem is written to
#   ~/.cloudflared/cert.pem on EC2
```

If EC2 is headless: run `cloudflared tunnel login` and copy the URL it
prints. Open that URL in your laptop browser, log into Cloudflare, then
the EC2 process completes.

### 3. Create the tunnel + assign hostname

You have two routing choices: a free `*.trycloudflare.com` URL (quickest)
or a custom subdomain on your own Cloudflare zone (cleaner long-term).

**Quick — try.cloudflare.com (no zone needed):**
```bash
cloudflared tunnel --url http://localhost:8002 &
# → prints a one-shot https://...trycloudflare.com URL
# → terminates when the process dies; not suitable for persistent service
```

**Persistent — named tunnel on your own zone (recommended):**
```bash
cloudflared tunnel create trading-agent
# → writes ~/.cloudflared/<UUID>.json (the tunnel's credentials)
# → output prints the UUID

# Route a hostname to it (requires you to own a CF zone, e.g. example.com):
cloudflared tunnel route dns trading-agent ta.example.com
# → creates the CNAME automatically
```

### 4. Write the tunnel config

`~/.cloudflared/config.yml`:

```yaml
tunnel: trading-agent
credentials-file: /home/ubuntu/.cloudflared/<UUID>.json

ingress:
  - hostname: ta.example.com     # or whatever you routed in step 3
    service: http://localhost:8002
    originRequest:
      noTLSVerify: true
  - service: http_status:404
```

### 5. Install + enable the systemd service

The auto-deploy script already installed the unit file as a symlink to
`deploy/ec2/systemd/trading-agent-tunnel.service`. You just need to
enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now trading-agent-tunnel.service
sudo systemctl status trading-agent-tunnel.service --no-pager
```

### 6. (Optional) Add Cloudflare Access auth

The tunnel URL is publicly reachable by default. To require login:

1. Cloudflare dashboard → Zero Trust → Access → Applications
2. Add app → self-hosted, hostname = `ta.example.com`
3. Policy: `email IS <your email>` (or Google SSO, GitHub, etc.)

After this, hitting `ta.example.com/mobile` redirects to a CF login page.

## Operating notes

- **Log**: `/var/log/trading-agent-tunnel.log` (rotated by systemd journal)
- **Restart**: `sudo systemctl restart trading-agent-tunnel.service`
- **Disable temporarily**: `sudo systemctl stop trading-agent-tunnel.service`
  (the dashboard FastAPI keeps running on localhost; just the public route closes)

## What you'll see on the phone

`https://ta.example.com/mobile` →
- Top: market regime label, confidence, HMM version, size multiplier
- Today's ops summary: trigger fires, dispatches, alerts, LLM cost
- Burn-in progress bar
- Open positions w/ PnL
- Recent severity > 0 events
- Pipeline subgraph last-fire times

Auto-refreshes every 30s, and when you bring the app to the foreground.
