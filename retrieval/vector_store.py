"""PostgreSQL + pgvector store wrapper.

Uses langchain-postgres PGVector. Provides:
  - initialize()     — create tables and indexes
  - add_documents()  — bulk ingest LangChain Documents
  - clear()          — delete all vectors (use with --clear during re-ingest)
  - similarity_search() — filtered cosine similarity search
  - get_all_chunks() — retrieve every chunk (used to build BM25 index on startup)
"""

from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from loguru import logger

from config import Config

COLLECTION_NAME = "ftf_pbs_chunks"


class VectorStore:
    def __init__(self, db_url: str, embeddings: Embeddings, cfg: Config) -> None:
        self._db_url = db_url
        self._embeddings = embeddings
        self._cfg = cfg
        self._store: Optional[PGVector] = None

    def initialize(self) -> None:
        """Connect to pgvector and create tables/indexes if they don't exist."""
        self._store = PGVector(
            embeddings=self._embeddings,
            collection_name=COLLECTION_NAME,
            connection=self._db_url,
            use_jsonb=True,
        )
        logger.info("VectorStore initialized.")

    def _require_store(self) -> PGVector:
        if self._store is None:
            self.initialize()
        return self._store  # type: ignore[return-value]

    def add_documents(self, documents: list[Document], batch_size: int = 50) -> None:
        """Embed and store documents in batches."""
        store = self._require_store()
        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            store.add_documents(batch)
            logger.info(f"  Stored batch {i // batch_size + 1} ({min(i + batch_size, total)}/{total})")

    def clear(self) -> None:
        """Delete the collection and all its vectors."""
        store = self._require_store()
        store.delete_collection()
        # Re-create empty collection
        self._store = None
        self.initialize()

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[Document]:
        """Return top-k chunks by cosine similarity, with optional metadata filter."""
        store = self._require_store()
        return store.similarity_search(query, k=k, filter=filter)

    def get_all_chunks(self) -> list[Document]:
        """Fetch every chunk from the store — used to build the BM25 index."""
        store = self._require_store()
        # PGVector exposes the underlying session; we query the embeddings table directly
        try:
            from sqlalchemy import text as sql_text

            with store._make_sync_session() as session:
                rows = session.execute(
                    sql_text(
                        "SELECT document, cmetadata FROM langchain_pg_embedding "
                        "WHERE collection_id = ("
                        "  SELECT uuid FROM langchain_pg_collection WHERE name = :name"
                        ")",
                    ),
                    {"name": COLLECTION_NAME},
                ).fetchall()

            docs = []
            for row in rows:
                docs.append(Document(page_content=row[0], metadata=row[1] or {}))
            logger.info(f"Loaded {len(docs)} chunks for BM25 index.")
            return docs
        except Exception as e:
            logger.warning(f"Could not load all chunks for BM25: {e}")
            return []
