"""Backfill MiniLM embeddings for notes missing from `notes_vec`.

Runs on Mac only (EC2 is intentionally lean: per Phase-1 plan, EC2 jobs
insert notes without embedding, and Mac-side re-hydrates them so the RAG
index stays current).

Strategy:
  SELECT n.id, n.topic, n.text FROM notes n
  LEFT JOIN notes_vec v ON v.note_id = n.id
  WHERE v.note_id IS NULL;
  → for each row, encode with MiniLM → INSERT INTO notes_vec.

Idempotent: re-running inserts 0 new vectors if all notes are already
embedded. Safe to run on a high-frequency cron.
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone

from trading_agent.config import CONFIG, ensure_dirs
from trading_agent.db import connection, migrate


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    ensure_dirs()
    migrate(CONFIG.db_path)

    # Defer the heavy import so a no-op run doesn't pay MiniLM startup.
    # We only spin up the model if there's at least one missing row.
    with connection() as conn:
        missing = conn.execute(
            """
            SELECT n.id, n.topic, n.text
            FROM notes n
            LEFT JOIN notes_vec v ON v.note_id = n.id
            WHERE v.note_id IS NULL
            ORDER BY n.id
            """
        ).fetchall()

        # Orphans: vec rows whose parent note no longer exists. vec0 doesn't
        # support FK cascade, so manual cleanups of `notes` leave stragglers
        # that inflate the index. Sweep them here.
        orphan_ids = [r[0] for r in conn.execute(
            "SELECT v.note_id FROM notes_vec v "
            "LEFT JOIN notes n ON n.id = v.note_id "
            "WHERE n.id IS NULL"
        ).fetchall()]
        if orphan_ids:
            conn.executemany(
                "DELETE FROM notes_vec WHERE note_id = ?",
                [(oid,) for oid in orphan_ids],
            )
            conn.commit()

    summary = {
        "ts": _now(),
        "job": "backfill_embeddings",
        "missing_count": len(missing),
        "orphans_removed": len(orphan_ids),
        "embedded": 0,
        "errors": [],
    }

    if not missing:
        print(json.dumps(summary, indent=2))
        return 0

    # Import MiniLM only when there's work.
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import struct
        model = SentenceTransformer(CONFIG.embedding_model)
    except Exception as e:
        summary["errors"].append({"phase": "model_load", "error": repr(e)})
        print(json.dumps(summary, indent=2))
        return 2

    def _embed(text: str) -> bytes:
        vec = model.encode([text], normalize_embeddings=True)[0]
        if len(vec) != 384:
            raise RuntimeError(f"dim {len(vec)} != 384")
        return struct.pack(f"{len(vec)}f", *[float(x) for x in vec])

    with connection() as conn:
        for row in missing:
            try:
                # Embed topic + text together, same as append_note() in the MCP.
                payload = f"{row['topic']}\n{row['text']}" if row['topic'] else row['text']
                emb = _embed(payload)
                conn.execute(
                    "INSERT INTO notes_vec (note_id, embedding) VALUES (?, ?)",
                    (row["id"], emb),
                )
                summary["embedded"] += 1
            except Exception as e:
                summary["errors"].append({
                    "note_id": row["id"],
                    "error": repr(e),
                    "tb": traceback.format_exc(limit=2),
                })
        conn.commit()

    print(json.dumps(summary, indent=2))
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
