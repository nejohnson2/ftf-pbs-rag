"""Database migration — run as Heroku release phase command.

Creates all required tables if they don't exist:
  - langchain_pg_collection / langchain_pg_embedding  (managed by langchain-postgres)
  - chat_history
  - query_logs

Safe to run repeatedly (uses CREATE TABLE IF NOT EXISTS).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import psycopg2
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from config import get_database_url


SQL = """
-- pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Session chat history
CREATE TABLE IF NOT EXISTS chat_history (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT        NOT NULL,
    role        TEXT        NOT NULL,
    content     TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chat_history_session
    ON chat_history (session_id, created_at);

-- Query audit log
CREATE TABLE IF NOT EXISTS query_logs (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT        NOT NULL,
    query       TEXT        NOT NULL,
    response    TEXT        NOT NULL,
    entities    JSONB       NOT NULL DEFAULT '{}',
    docs_cited  JSONB       NOT NULL DEFAULT '[]',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_query_logs_session
    ON query_logs (session_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_created
    ON query_logs (created_at);
"""


def main() -> None:
    logger.info("Running database migration...")
    try:
        with psycopg2.connect(get_database_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(SQL)
        logger.info("Migration complete.")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
