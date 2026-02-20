"""Embedding model loader.

Supports two providers (configured in config.yaml):
  - "ollama"               — calls the Ollama /api/embeddings endpoint (default)
  - "sentence_transformers" — runs a local SentenceTransformer model (requires torch)

Returns a LangChain Embeddings object compatible with all LangChain retrievers.
"""

from functools import lru_cache
from typing import TYPE_CHECKING

from config import Config

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


@lru_cache(maxsize=1)
def get_embeddings(cfg: Config) -> "Embeddings":
    """Return a cached embedding model instance."""
    provider = cfg.embeddings.provider

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            base_url=cfg.ollama.base_url,
            model=cfg.ollama.embed_model,
        )

    if provider == "sentence_transformers":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name=cfg.embeddings.st_model,
                model_kwargs={"device": cfg.embeddings.st_device},
                encode_kwargs={"normalize_embeddings": True},
                cache_folder=cfg.embeddings.st_cache_dir,
            )
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install -r requirements-ingest.txt"
            ) from e

    raise ValueError(f"Unknown embedding provider: {provider!r}")
