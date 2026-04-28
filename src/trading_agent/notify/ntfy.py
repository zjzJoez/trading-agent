"""ntfy.sh notification sender + Discord mirror + filesystem fallback.

Public-server free tier; topic prefix is randomized in `.env` and
shared between EC2 sender and the user's iOS app subscription.

Topic structure (one prefix, five suffixes):
    {prefix}-trades   — orders, decisions
    {prefix}-risk     — regime changes, vetoes, forced exits
    {prefix}-ops      — health, errors, OpenD restarts
    {prefix}-digest   — daily / weekly summaries
    {prefix}-learning — canary lifecycle events
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx

log = logging.getLogger(__name__)

Topic = Literal["trades", "risk", "ops", "digest", "learning"]

# Filesystem fallback if ntfy is unreachable (e.g., outbound HTTPS blocked).
# Important for audit replay — we never silently drop a notification.
FALLBACK_LOG = Path.home() / "trading-agent" / "data" / "notifications.log"


def _config() -> dict[str, str]:
    """Read ntfy config from env. NTFY_TOPIC_PREFIX is required."""
    base = os.environ.get("NTFY_BASE_URL", "https://ntfy.sh").rstrip("/")
    prefix = os.environ.get("NTFY_TOPIC_PREFIX")
    if not prefix:
        # Try .env files in common project locations
        env_candidates: list[Path] = [
            Path.cwd() / ".env",
            Path.home() / "trading-agent" / ".env",
        ]
        # Also try the project root if this module lives inside src/
        try:
            env_candidates.append(Path(__file__).resolve().parents[3] / ".env")
        except IndexError:
            pass
        for env_path in env_candidates:
            if env_path.is_file():
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("NTFY_TOPIC_PREFIX="):
                        prefix = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("NTFY_BASE_URL="):
                        base = (
                            line.split("=", 1)[1].strip().strip('"').strip("'").rstrip("/")
                        )
                if prefix:
                    break
    if not prefix:
        raise RuntimeError(
            "NTFY_TOPIC_PREFIX not set in env or .env. "
            "Run scripts/setup_ntfy.sh or set NTFY_TOPIC_PREFIX manually."
        )
    return {"base": base, "prefix": prefix}


def send(
    topic: Topic,
    title: str,
    body: str,
    *,
    priority: int = 3,
    tags: list[str] | None = None,
    click: str | None = None,
    timeout_s: float = 10.0,
    discord_mirror: bool = True,
) -> dict:
    """Send a notification.

    priority: 1=min (silent), 2=low, 3=default, 4=high (sound), 5=urgent (alarm)
    tags:     emoji shortcuts (e.g. ['warning', 'rotating_light']) — see ntfy docs
    click:    URL the user is sent to when tapping the push notification

    Returns a dict {ok, status, ntfy_id?}. Never raises — failures are logged
    + written to FALLBACK_LOG. Discord mirror is best-effort.
    """
    cfg = _config()
    url = f"{cfg['base']}/{cfg['prefix']}-{topic}"

    # HTTP/1.1 header values must be Latin-1. ntfy's documented escape:
    # encode non-ASCII chars (like emoji) as RFC 2047 / quoted printable,
    # OR use the `X-Title` raw bytes via RFC 8187 (`Title*=utf-8''...`).
    # Simplest path: encode Title as RFC 8187 utf-8 when needed.
    def _http_safe(value: str) -> str:
        try:
            value.encode("latin-1")
            return value
        except UnicodeEncodeError:
            from urllib.parse import quote
            return f"=?utf-8?B?{__import__('base64').b64encode(value.encode('utf-8')).decode('ascii')}?="

    headers = {
        "Title": _http_safe(title),
        "Priority": str(priority),
    }
    if tags:
        headers["Tags"] = ",".join(tags)
    if click:
        headers["Click"] = click

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "topic": topic,
        "title": title,
        "body": body,
        "priority": priority,
        "tags": tags or [],
        "click": click,
    }

    result: dict = {"ok": False, "topic": topic}
    try:
        r = httpx.post(url, headers=headers, content=body.encode(), timeout=timeout_s)
        r.raise_for_status()
        try:
            data = r.json()
            result.update({"ok": True, "status": r.status_code, "ntfy_id": data.get("id")})
        except Exception:
            result.update({"ok": True, "status": r.status_code})
    except Exception as e:
        result.update({"ok": False, "error": str(e)})
        log.error("ntfy send failed (topic=%s): %s", topic, e)

    # Filesystem fallback (always written, regardless of ntfy success — audit trail)
    try:
        FALLBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        with FALLBACK_LOG.open("a") as f:
            f.write(json.dumps({**record, "result": result}) + "\n")
    except Exception as e:
        log.error("notifications.log write failed: %s", e)

    # Discord mirror (best-effort, fire-and-forget)
    if discord_mirror:
        webhook = os.environ.get("DISCORD_WEBHOOK_URL")
        if webhook:
            try:
                emoji = {1: "ℹ️", 2: "ℹ️", 3: "📊", 4: "⚠️", 5: "🚨"}.get(priority, "📊")
                content = f"{emoji} **[{topic}] {title}**\n{body}"[:1900]
                httpx.post(webhook, json={"content": content}, timeout=5.0)
            except Exception as e:
                log.warning("Discord mirror failed: %s", e)

    return result


# ---------------------------------------------------------------------------
# CLI: ta-ntfy-test — used by scripts and ops
# ---------------------------------------------------------------------------


def cli_test() -> None:
    """Console entry point: send a test message to each topic.

    Usage:
        ta-ntfy-test                         # send one to ops
        ta-ntfy-test --all                   # send to all 5
        ta-ntfy-test --topic trades --body "test"
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args = sys.argv[1:]
    if "--all" in args:
        topics: list[Topic] = ["trades", "risk", "ops", "digest", "learning"]
    elif "--topic" in args:
        i = args.index("--topic")
        topics = [args[i + 1]]
    else:
        topics = ["ops"]

    body = "ntfy is alive on EC2."
    if "--body" in args:
        i = args.index("--body")
        body = args[i + 1]

    cfg = _config()
    print(f"Sending to {cfg['base']}/{cfg['prefix']}-* ({len(topics)} topics)")

    for t in topics:
        priority = {"trades": 3, "risk": 4, "ops": 3, "digest": 2, "learning": 3}.get(t, 3)
        result = send(
            t,  # type: ignore
            title=f"🧪 ntfy test → {t}",
            body=body,
            priority=priority,
            tags=["test"],
        )
        ok = "✅" if result["ok"] else "❌"
        print(f"  {ok} {t}: {result}")


if __name__ == "__main__":
    cli_test()
