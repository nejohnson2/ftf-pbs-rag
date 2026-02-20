"""In-memory BM25 keyword index.

Built at startup from the full corpus loaded out of pgvector.
Allows keyword/exact-term search alongside the semantic vector search
(especially useful for acronyms like ZOI, PBS, RFZ, and country names).
"""

import re
from typing import Optional

from langchain_core.documents import Document
from loguru import logger
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self) -> None:
        self._docs: list[Document] = []
        self._bm25: Optional[BM25Okapi] = None

    def build(self, documents: list[Document]) -> None:
        """Build the BM25 index from a list of LangChain Documents."""
        if not documents:
            logger.warning("BM25: no documents provided — index will be empty.")
            return
        self._docs = documents
        tokenized = [_tokenize(d.page_content) for d in documents]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built over {len(documents)} chunks.")

    def search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """Return top-k documents by BM25 score.

        Args:
            query:  The search query.
            k:      Number of results to return.
            filter: Optional metadata filter dict (same format as pgvector filter).
                    Applied post-scoring — only docs matching ALL filter keys are returned.
        """
        if self._bm25 is None:
            logger.warning("BM25 index is empty. Returning no results.")
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Pair scores with docs and sort descending
        ranked = sorted(
            zip(scores, self._docs),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, doc in ranked:
            if score <= 0:
                break
            if filter and not _matches_filter(doc.metadata, filter):
                continue
            results.append(doc)
            if len(results) >= k:
                break

        return results

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None


def _matches_filter(metadata: dict, filter: dict) -> bool:
    """Check if a document's metadata satisfies a filter dict.

    Supports simple equality and list membership:
      {"country": "Kenya"}
      {"country": {"$in": ["Kenya", "Uganda"]}}
    """
    for key, value in filter.items():
        doc_val = metadata.get(key)
        if isinstance(value, dict):
            op, operand = next(iter(value.items()))
            if op == "$in" and doc_val not in operand:
                return False
            if op == "$eq" and doc_val != operand:
                return False
        else:
            if doc_val != value:
                return False
    return True
