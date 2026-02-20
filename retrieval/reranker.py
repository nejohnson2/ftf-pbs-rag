"""Cross-encoder reranker (optional).

Requires sentence-transformers + torch. Disabled by default for Heroku.
Enable with: enable_reranker: true in config.yaml

The cross-encoder scores each (query, passage) pair jointly, giving
significantly better ranking than the bi-encoder similarity used in
the initial vector search.
"""

from typing import TYPE_CHECKING

from langchain_core.documents import Document
from loguru import logger

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: "CrossEncoder | None" = None

    def load(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
            logger.info(f"Reranker loaded: {self._model_name}")
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is required for reranking. "
                "Install it with: pip install sentence-transformers"
            ) from e

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        if self._model is None:
            logger.warning("Reranker not loaded — returning documents unsorted.")
            return documents[:top_k]

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)

        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
