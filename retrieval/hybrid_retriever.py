"""Hybrid retriever: semantic search + BM25 fused with Reciprocal Rank Fusion.

Pipeline:
  1. query_analyzer extracts entities → builds metadata filter
  2. pgvector similarity_search (filtered)
  3. BM25 search (filtered)
  4. Reciprocal Rank Fusion merges both ranked lists
  5. Optional cross-encoder reranking (reranker.py)
  6. Return top-k chunks with citation metadata attached
"""

from langchain_core.documents import Document
from loguru import logger

from config import Config
from retrieval import query_analyzer
from retrieval.bm25_index import BM25Index
from retrieval.vector_store import VectorStore


def _reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (rank + k)) across all lists.
    Documents are identified by their page_content hash.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = hash(doc.page_content)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + k)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


def retrieve(
    query: str,
    vector_store: VectorStore,
    bm25_index: BM25Index,
    cfg: Config,
    reranker=None,
) -> list[Document]:
    """Full hybrid retrieval pipeline.

    Args:
        query:        User's natural language question.
        vector_store: Initialized VectorStore.
        bm25_index:   Initialized BM25Index.
        cfg:          Application config.
        reranker:     Optional Reranker instance (None if disabled).

    Returns:
        Top-k LangChain Documents with metadata (doc_id, title, url, etc.)
    """
    r_cfg = cfg.retrieval

    # ── Step 1: Extract query entities ──────────────────────────────────────
    entities = query_analyzer.analyze(query)
    metadata_filter = entities.to_metadata_filter()

    if entities.has_filters():
        logger.info(
            f"Query entities — countries: {entities.countries}, "
            f"phases: {entities.phases}, types: {entities.survey_types}, "
            f"years: {entities.years}"
        )
    else:
        logger.info("No specific entities detected — searching full corpus.")

    # ── Step 2: Semantic search ──────────────────────────────────────────────
    semantic_results = vector_store.similarity_search(
        query,
        k=r_cfg.top_k_semantic,
        filter=metadata_filter,
    )
    logger.debug(f"Semantic search returned {len(semantic_results)} results.")

    # ── Step 3: BM25 keyword search ──────────────────────────────────────────
    bm25_results: list[Document] = []
    if bm25_index.is_ready:
        bm25_results = bm25_index.search(
            query,
            k=r_cfg.top_k_bm25,
            filter=metadata_filter,
        )
        logger.debug(f"BM25 search returned {len(bm25_results)} results.")

    # ── Step 4: RRF fusion ───────────────────────────────────────────────────
    if bm25_results:
        fused = _reciprocal_rank_fusion([semantic_results, bm25_results])
    else:
        fused = semantic_results

    # ── Step 5: Cross-encoder reranking (optional) ───────────────────────────
    if reranker is not None and cfg.retrieval.enable_reranker:
        final = reranker.rerank(query, fused, top_k=r_cfg.top_k_reranked)
    else:
        final = fused[: r_cfg.top_k_reranked]

    logger.info(f"Returning {len(final)} chunks to the generator.")
    return final
