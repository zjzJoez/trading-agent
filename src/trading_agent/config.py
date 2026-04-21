"""Central config. Reads .env from repo root; all paths resolve off PROJECT_ROOT.

Single source of truth — MCP servers, hooks, jobs, and skills all import from here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Config:
    project_root: Path
    data_dir: Path
    db_path: Path
    filings_cache_dir: Path
    opend_host: str
    opend_port: int
    sec_user_agent: str
    embedding_model: str
    log_dir: Path

    @classmethod
    def load(cls) -> "Config":
        data_dir = PROJECT_ROOT / "data"
        sec_email = os.environ.get("SEC_UA_EMAIL", "").strip()
        if not sec_email:
            # Fall back rather than crash so migrations/tests can run. Real usage must set it.
            sec_email = "anonymous@example.invalid"
        db_path_env = os.environ.get("DB_PATH", "").strip()
        return cls(
            project_root=PROJECT_ROOT,
            data_dir=data_dir,
            db_path=Path(db_path_env) if db_path_env else data_dir / "trader.db",
            filings_cache_dir=data_dir / "filings_cache",
            opend_host=os.environ.get("OPEND_HOST", "127.0.0.1"),
            opend_port=int(os.environ.get("OPEND_PORT", "11111")),
            sec_user_agent=f"Trading Agent MVP {sec_email}",
            embedding_model=os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            log_dir=data_dir / "logs",
        )


CONFIG = Config.load()


def ensure_dirs() -> None:
    """Create data/log/cache dirs if missing. Safe to call repeatedly."""
    for p in (CONFIG.data_dir, CONFIG.filings_cache_dir, CONFIG.log_dir):
        p.mkdir(parents=True, exist_ok=True)
