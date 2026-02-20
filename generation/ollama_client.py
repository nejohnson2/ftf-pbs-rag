"""Ollama LLM client wrapper.

Wraps langchain-ollama's ChatOllama with our config. The client is a
LangChain ChatModel, so it integrates directly into LCEL chains.
"""

from functools import lru_cache

from langchain_ollama import ChatOllama

from config import Config


@lru_cache(maxsize=1)
def get_llm(cfg: Config) -> ChatOllama:
    """Return a cached ChatOllama instance."""
    return ChatOllama(
        base_url=cfg.ollama.base_url,
        model=cfg.ollama.model,
        temperature=cfg.ollama.temperature,
        num_predict=cfg.ollama.max_tokens,
        timeout=cfg.ollama.request_timeout,
    )
