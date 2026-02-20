"""RAG chain: ties together retrieval and generation.

Entry point for the application routes:
    result = await rag_chain.invoke(query, session_history)

Returns a RAGResult with the LLM answer text and structured citations.
"""

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from loguru import logger

from config import Config
from generation.citations import Citation, build_citations, format_context_block
from generation.ollama_client import get_llm
from generation.prompts import RAG_PROMPT, RAG_PROMPT_WITH_HISTORY
from retrieval.bm25_index import BM25Index
from retrieval.hybrid_retriever import retrieve
from retrieval.vector_store import VectorStore


@dataclass
class RAGResult:
    answer: str
    citations: list[Citation]
    retrieved_docs: list[Document]
    query_entities: dict        # for logging


class RAGChain:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        cfg: Config,
        reranker=None,
    ) -> None:
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._cfg = cfg
        self._reranker = reranker
        self._llm = get_llm(cfg)

    async def invoke(
        self,
        query: str,
        history: list[dict] | None = None,
    ) -> RAGResult:
        """Execute the full RAG pipeline for a user query.

        Args:
            query:   The user's question.
            history: List of {"role": "user"|"assistant", "content": "..."} dicts
                     representing the current session's conversation history.

        Returns:
            RAGResult with answer text and citation objects.
        """
        # ── Retrieve ──────────────────────────────────────────────────────────
        from retrieval.query_analyzer import analyze

        entities = analyze(query)

        docs = retrieve(
            query=query,
            vector_store=self._vector_store,
            bm25_index=self._bm25_index,
            cfg=self._cfg,
            reranker=self._reranker,
        )

        if not docs:
            logger.warning("No documents retrieved for query.")
            return RAGResult(
                answer=(
                    "I couldn't find relevant information in the survey reports "
                    "to answer your question. Please try rephrasing or asking about "
                    "a specific country, phase, or survey round."
                ),
                citations=[],
                retrieved_docs=[],
                query_entities={
                    "countries": entities.countries,
                    "phases": entities.phases,
                    "survey_types": entities.survey_types,
                    "years": entities.years,
                },
            )

        # ── Format context ────────────────────────────────────────────────────
        context_block = format_context_block(docs)

        # ── Select prompt (with or without history) ───────────────────────────
        if history:
            history_text = _format_history(history, self._cfg.session.max_history_turns)
            prompt = RAG_PROMPT_WITH_HISTORY
            prompt_input = {
                "question": query,
                "context": context_block,
                "history": history_text,
            }
        else:
            prompt = RAG_PROMPT
            prompt_input = {
                "question": query,
                "context": context_block,
            }

        # ── Generate ──────────────────────────────────────────────────────────
        chain = prompt | self._llm
        response = await chain.ainvoke(prompt_input)
        answer = response.content if hasattr(response, "content") else str(response)

        logger.info(f"Generated answer ({len(answer)} chars) from {len(docs)} chunks.")

        return RAGResult(
            answer=answer,
            citations=build_citations(docs),
            retrieved_docs=docs,
            query_entities={
                "countries": entities.countries,
                "phases": entities.phases,
                "survey_types": entities.survey_types,
                "years": entities.years,
            },
        )


def _format_history(history: list[dict], max_turns: int) -> str:
    """Format the last N conversation turns as a readable block."""
    recent = history[-(max_turns * 2):]    # each turn = 2 messages (user + assistant)
    lines = []
    for msg in recent:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
