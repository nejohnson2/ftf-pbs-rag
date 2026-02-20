"""Ollama LLM client wrapper.

Wraps langchain-ollama's ChatOllama with our config. The client is a
LangChain ChatModel, so it integrates directly into LCEL chains.
"""

from langchain_ollama import ChatOllama

from config import Config

_llm_instance = None


def get_llm(cfg: Config) -> ChatOllama:
    """Return a cached ChatOllama instance."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance
    _llm_instance = ChatOllama(
        base_url=cfg.ollama.base_url,
        model=cfg.ollama.model,
        temperature=cfg.ollama.temperature,
        num_predict=cfg.ollama.max_tokens,
        timeout=cfg.ollama.request_timeout,
    )
    return _llm_instance
