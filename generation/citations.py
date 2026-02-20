"""Citation formatter.

Takes the list of retrieved LangChain Documents and formats them into:
  1. A numbered context block injected into the LLM prompt.
  2. A structured citation list returned to the UI alongside the answer.
"""

from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document


@dataclass
class Citation:
    number: int
    doc_id: str
    title: str
    country: Optional[str]
    phase: Optional[int]
    survey_type: Optional[str]
    year: Optional[int]
    doc_type: str
    online_url: Optional[str]
    excerpt: str            # first ~200 chars of the chunk


def format_context_block(documents: list[Document]) -> str:
    """Format retrieved documents into a numbered context block for the prompt."""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata
        label_parts = [f"[{i}]"]
        if meta.get("country"):
            label_parts.append(meta["country"])
        if meta.get("phase"):
            label_parts.append(f"Phase {meta['phase']}")
        if meta.get("survey_type"):
            label_parts.append(meta["survey_type"].replace("_", " ").title())
        if meta.get("year"):
            label_parts.append(str(meta["year"]))
        if meta.get("title"):
            label_parts.append(f'"{meta["title"]}"')

        header = " | ".join(label_parts)
        blocks.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(blocks)


def build_citations(documents: list[Document]) -> list[Citation]:
    """Build a structured list of Citation objects for the UI."""
    citations = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata
        citations.append(
            Citation(
                number=i,
                doc_id=meta.get("doc_id", ""),
                title=meta.get("title") or _infer_title(meta),
                country=meta.get("country"),
                phase=meta.get("phase"),
                survey_type=meta.get("survey_type"),
                year=meta.get("year"),
                doc_type=meta.get("doc_type", "full_report"),
                online_url=meta.get("online_url"),
                excerpt=doc.page_content[:200].strip() + "…",
            )
        )
    return citations


def _infer_title(meta: dict) -> str:
    """Generate a readable title from metadata when none is stored."""
    parts = []
    if meta.get("country"):
        parts.append(meta["country"])
    if meta.get("phase"):
        parts.append(f"Phase {meta['phase']}")
    if meta.get("survey_type"):
        parts.append(meta["survey_type"].replace("_", " ").title())
    if meta.get("year"):
        parts.append(str(meta["year"]))
    return " ".join(parts) if parts else "FTF PBS Report"
