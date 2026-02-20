"""Ingest preprocessed markdown files into the pgvector database.

Reads data/processed/metadata.json, loads each markdown file where
include=True, chunks the text, embeds with the configured model, and
stores everything in PostgreSQL + pgvector.

Run this after scripts/preprocess.py and after manually editing markdowns:
    python -m ingestion.build_index
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger

# Project root on sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv()

from config import get_config, get_database_url
from retrieval.embeddings import get_embeddings
from retrieval.vector_store import VectorStore


def load_metadata(metadata_path: Path) -> list[dict]:
    with open(metadata_path) as f:
        data = json.load(f)
    return data.get("documents", [])


def build_langchain_doc(chunk: str, doc_meta: dict, chunk_index: int) -> Document:
    """Wrap a text chunk as a LangChain Document with rich metadata."""
    return Document(
        page_content=chunk,
        metadata={
            "doc_id": doc_meta["doc_id"],
            "chunk_index": chunk_index,
            "country": doc_meta.get("country"),
            "phase": doc_meta.get("phase"),
            "survey_type": doc_meta.get("survey_type"),
            "doc_type": doc_meta.get("doc_type"),
            "year": doc_meta.get("year"),
            "title": doc_meta.get("title", ""),
            "online_url": doc_meta.get("online_url"),
        },
    )


def ingest(clear_existing: bool = False) -> None:
    cfg = get_config()
    db_url = get_database_url()
    embeddings = get_embeddings(cfg)

    metadata_path = ROOT / cfg.preprocessing.metadata_output
    if not metadata_path.exists():
        logger.error(f"metadata.json not found at {metadata_path}. Run scripts/preprocess.py first.")
        sys.exit(1)

    documents_meta = load_metadata(metadata_path)
    included = [d for d in documents_meta if d.get("include", True)]
    logger.info(f"Found {len(included)} documents to ingest (of {len(documents_meta)} total)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.retrieval.chunk_size,
        chunk_overlap=cfg.retrieval.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Document] = []

    for doc_meta in included:
        md_path = ROOT / doc_meta["markdown_path"]
        if not md_path.exists():
            logger.warning(f"Markdown not found for {doc_meta['doc_id']}: {md_path}")
            continue

        text = md_path.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning(f"Empty markdown for {doc_meta['doc_id']}")
            continue

        raw_chunks = splitter.split_text(text)
        for i, chunk in enumerate(raw_chunks):
            all_chunks.append(build_langchain_doc(chunk, doc_meta, i))

        logger.info(f"  {doc_meta['doc_id']}: {len(raw_chunks)} chunks")

    if not all_chunks:
        logger.error("No chunks to ingest.")
        sys.exit(1)

    logger.info(f"Total chunks to embed and store: {len(all_chunks)}")

    store = VectorStore(db_url=db_url, embeddings=embeddings, cfg=cfg)
    store.initialize()

    if clear_existing:
        logger.warning("Clearing existing vectors from the store...")
        store.clear()

    store.add_documents(all_chunks)
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest preprocessed markdown into pgvector")
    parser.add_argument("--clear", action="store_true", help="Clear existing vectors before ingesting")
    args = parser.parse_args()

    ingest(clear_existing=args.clear)
