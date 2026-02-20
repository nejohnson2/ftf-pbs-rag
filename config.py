"""Central configuration loader.

Reads config.yaml, applies environment variable overrides, and exposes a
typed Config object via get_config(). All modules import from here.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, field_validator

CONFIG_PATH = Path(__file__).parent / "config.yaml"


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    embed_model: str = "nomic-embed-text"
    temperature: float = 0.1
    max_tokens: int = 2048
    request_timeout: int = 120


class EmbeddingsConfig(BaseModel):
    provider: Literal["ollama", "sentence_transformers"] = "ollama"
    dimensions: int = 768
    st_model: str = "BAAI/bge-base-en-v1.5"
    st_device: str = "cpu"
    st_cache_dir: str = ".model_cache"


class RetrievalConfig(BaseModel):
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k_semantic: int = 12
    top_k_bm25: int = 12
    top_k_reranked: int = 5
    hybrid_alpha: float = 0.5
    enable_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class QueryAnalysisConfig(BaseModel):
    extract_countries: bool = True
    extract_phases: bool = True
    extract_years: bool = True
    extract_survey_types: bool = True


class PreprocessingConfig(BaseModel):
    docs_root: str = "Archived Population-Based Survey Reports"
    output_dir: str = "data/processed/markdown"
    metadata_output: str = "data/processed/metadata.json"
    remove_toc: bool = True
    remove_references_section: bool = True
    remove_appendices: bool = True
    remove_survey_instruments: bool = True
    min_paragraph_length: int = 80


class SessionConfig(BaseModel):
    cookie_name: str = "ftf_session"
    cookie_max_age: int = 86400
    max_history_turns: int = 8


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_queries: bool = True
    log_file: str = "logs/queries.jsonl"


class DatabaseConfig(BaseModel):
    pool_size: int = 5
    max_overflow: int = 10


class AppConfig(BaseModel):
    title: str = "Feed the Future PBS Reports"
    description: str = "Explore Population-Based Survey reports from 20 countries"


class Config(BaseModel):
    ollama: OllamaConfig = OllamaConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    query_analysis: QueryAnalysisConfig = QueryAnalysisConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    session: SessionConfig = SessionConfig()
    logging: LoggingConfig = LoggingConfig()
    database: DatabaseConfig = DatabaseConfig()
    app: AppConfig = AppConfig()


def _apply_env_overrides(raw: dict) -> dict:
    """Apply environment variable overrides to the raw config dict."""
    env_map = {
        "OLLAMA_BASE_URL": ("ollama", "base_url"),
        "OLLAMA_MODEL": ("ollama", "model"),
        "OLLAMA_EMBED_MODEL": ("ollama", "embed_model"),
        "EMBEDDING_PROVIDER": ("embeddings", "provider"),
        "EMBEDDING_DIMENSIONS": ("embeddings", "dimensions"),
    }
    for env_key, (section, field) in env_map.items():
        if env_key in os.environ:
            raw.setdefault(section, {})[field] = os.environ[env_key]
    return raw


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load and return the cached Config singleton."""
    raw: dict = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            raw = yaml.safe_load(f) or {}
    raw = _apply_env_overrides(raw)
    return Config(**raw)


def get_database_url() -> str:
    """Return the database URL, always from the environment for security."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "Add it to your .env file for local development."
        )
    # Heroku returns postgres:// but SQLAlchemy requires postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    # Heroku Postgres requires SSL; add sslmode=require for non-local connections
    if "localhost" not in url and "127.0.0.1" not in url:
        if "sslmode" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}sslmode=require"
    return url


def get_secret_key() -> str:
    """Return the session secret key from the environment."""
    key = os.environ.get("SECRET_KEY", "dev-insecure-secret-change-me")
    return key
