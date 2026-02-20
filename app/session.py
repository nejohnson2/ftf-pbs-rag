"""Cookie-based anonymous session management.

Each visitor gets a UUID stored in a signed cookie. Conversation history
is kept in a database table (chat_history) so it survives dyno restarts.
"""

import uuid
from typing import Optional

import psycopg2
from itsdangerous import BadSignature, URLSafeSerializer
from loguru import logger

from config import get_config, get_database_url, get_secret_key


def _get_serializer() -> URLSafeSerializer:
    return URLSafeSerializer(get_secret_key(), salt="session")


def create_session_id() -> str:
    return str(uuid.uuid4())


def sign_session_id(session_id: str) -> str:
    return _get_serializer().dumps(session_id)


def verify_session_cookie(cookie_value: str) -> Optional[str]:
    """Verify and return the session_id from a signed cookie, or None."""
    try:
        return _get_serializer().loads(cookie_value)
    except BadSignature:
        return None


# ── Database helpers ──────────────────────────────────────────────────────────

def _get_conn():
    return psycopg2.connect(get_database_url())


def get_history(session_id: str, max_turns: int) -> list[dict]:
    """Load the last max_turns*2 messages for a session."""
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, content FROM chat_history
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (session_id, max_turns * 2),
                )
                rows = cur.fetchall()
        # Rows come in reverse order; flip them
        return [{"role": r, "content": c} for r, c in reversed(rows)]
    except Exception as e:
        logger.warning(f"Could not load history for {session_id}: {e}")
        return []


def append_history(session_id: str, role: str, content: str) -> None:
    """Append a single message to the session history."""
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_history (session_id, role, content)
                    VALUES (%s, %s, %s)
                    """,
                    (session_id, role, content),
                )
    except Exception as e:
        logger.warning(f"Could not save history for {session_id}: {e}")
