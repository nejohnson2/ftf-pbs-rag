"""Query and response logging.

Writes newline-delimited JSON to logs/queries.jsonl and to the database
query_logs table for analysis.

Log record schema:
  {
    "ts": "ISO timestamp",
    "session_id": "uuid",
    "query": "user question",
    "answer": "LLM response",
    "entities": {...},
    "docs_cited": [{"doc_id": ..., "title": ...}, ...]
  }
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from loguru import logger

from config import get_config, get_database_url


def log_query(
    session_id: str,
    query: str,
    answer: str,
    entities: dict,
    docs_cited: list[dict],
) -> None:
    """Write a query log entry to file and database."""
    cfg = get_config()

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "query": query,
        "answer": answer,
        "entities": entities,
        "docs_cited": docs_cited,
    }

    # ── File log ──────────────────────────────────────────────────────────────
    if cfg.logging.log_queries:
        log_path = Path(cfg.logging.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"Could not write to log file: {e}")

    # ── Database log ──────────────────────────────────────────────────────────
    try:
        with psycopg2.connect(get_database_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_logs
                        (session_id, query, response, entities, docs_cited)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        session_id,
                        query,
                        answer,
                        json.dumps(entities),
                        json.dumps(docs_cited),
                    ),
                )
    except Exception as e:
        logger.warning(f"Could not write query log to DB: {e}")
