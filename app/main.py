"""FastAPI application entry point.

Lifespan events:
  - Initialize DB connection and create tables
  - Load embedding model
  - Initialize VectorStore and BM25 index
  - Optionally load cross-encoder reranker
  - Assemble RAGChain and attach to app.state
"""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger

load_dotenv()

from config import get_config, get_database_url
from generation.chain import RAGChain
from retrieval.bm25_index import BM25Index
from retrieval.embeddings import get_embeddings
from retrieval.vector_store import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all heavy resources once at startup."""
    cfg = get_config()

    # ── Logging setup ─────────────────────────────────────────────────────────
    logging.basicConfig(level=cfg.logging.level)
    logger.info("Starting FTF PBS RAG system...")

    # ── Database ──────────────────────────────────────────────────────────────
    db_url = get_database_url()

    # ── Embeddings ────────────────────────────────────────────────────────────
    logger.info(f"Loading embedding model (provider: {cfg.embeddings.provider})...")
    embeddings = get_embeddings(cfg)

    # ── Vector store ──────────────────────────────────────────────────────────
    vector_store = VectorStore(db_url=db_url, embeddings=embeddings, cfg=cfg)
    vector_store.initialize()

    # ── BM25 index (built in memory from the full corpus) ─────────────────────
    logger.info("Building BM25 index from corpus...")
    bm25_index = BM25Index()
    try:
        all_chunks = vector_store.get_all_chunks()
        bm25_index.build(all_chunks)
    except Exception as e:
        logger.warning(f"BM25 index could not be built: {e}. Continuing without keyword search.")

    # ── Optional cross-encoder reranker ───────────────────────────────────────
    reranker = None
    if cfg.retrieval.enable_reranker:
        logger.info(f"Loading reranker: {cfg.retrieval.reranker_model}")
        try:
            from retrieval.reranker import Reranker

            reranker = Reranker(cfg.retrieval.reranker_model)
            reranker.load()
        except Exception as e:
            logger.warning(f"Reranker failed to load: {e}. Continuing without reranking.")

    # ── RAG chain ─────────────────────────────────────────────────────────────
    app.state.rag_chain = RAGChain(
        vector_store=vector_store,
        bm25_index=bm25_index,
        cfg=cfg,
        reranker=reranker,
    )

    logger.info("System ready.")
    yield

    logger.info("Shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    cfg = get_config()
    app = FastAPI(
        title=cfg.app.title,
        description=cfg.app.description,
        lifespan=lifespan,
    )

    # Static files (CSS, JS)
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

    # Routers
    from app.routes.chat import router as chat_router
    from app.routes.documents import router as docs_router

    app.include_router(chat_router)
    app.include_router(docs_router)

    return app


app = create_app()
