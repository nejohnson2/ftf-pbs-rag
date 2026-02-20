"""Developer tool: test the retrieval pipeline interactively.

Usage:
    python scripts/test_retrieval.py "What was the stunting rate in Kenya?"
    python scripts/test_retrieval.py  # interactive prompt
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from config import get_config, get_database_url
from generation.chain import RAGChain
from retrieval.bm25_index import BM25Index
from retrieval.embeddings import get_embeddings
from retrieval.query_analyzer import analyze
from retrieval.vector_store import VectorStore


async def run_query(query: str, verbose: bool = True) -> None:
    cfg = get_config()
    db_url = get_database_url()

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    # Entity analysis
    entities = analyze(query)
    print(f"\nDetected entities:")
    print(f"  Countries:    {entities.countries or '(none)'}")
    print(f"  Phases:       {entities.phases or '(none)'}")
    print(f"  Survey types: {entities.survey_types or '(none)'}")
    print(f"  Years:        {entities.years or '(none)'}")
    print(f"  Filter:       {entities.to_metadata_filter() or '(none — full corpus)'}")

    # Initialize components
    embeddings = get_embeddings(cfg)
    vector_store = VectorStore(db_url=db_url, embeddings=embeddings, cfg=cfg)
    vector_store.initialize()

    bm25_index = BM25Index()
    chunks = vector_store.get_all_chunks()
    bm25_index.build(chunks)

    chain = RAGChain(
        vector_store=vector_store,
        bm25_index=bm25_index,
        cfg=cfg,
    )

    result = await chain.invoke(query)

    if verbose:
        print(f"\nRetrieved {len(result.retrieved_docs)} chunks:")
        for i, doc in enumerate(result.retrieved_docs, 1):
            m = doc.metadata
            print(f"\n  [{i}] {m.get('country')} | Phase {m.get('phase')} | "
                  f"{m.get('survey_type')} | {m.get('doc_type')}")
            print(f"      {doc.page_content[:150].strip()}…")

    print(f"\n{'─'*60}")
    print("Answer:")
    print(result.answer)

    if result.citations:
        print(f"\nCitations ({len(result.citations)}):")
        for c in result.citations:
            url = f" → {c.online_url}" if c.online_url else ""
            print(f"  [{c.number}] {c.title}{url}")


async def interactive() -> None:
    print("FTF PBS RAG — Retrieval Test")
    print("Type 'quit' to exit.\n")
    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in ("quit", "exit", "q"):
            break
        if query:
            await run_query(query)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        asyncio.run(run_query(" ".join(sys.argv[1:])))
    else:
        asyncio.run(interactive())
