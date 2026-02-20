"""Prompt templates for the RAG chain."""

from langchain_core.prompts import ChatPromptTemplate

# ── System message ────────────────────────────────────────────────────────────

SYSTEM_MESSAGE = """\
You are an expert analyst specializing in Feed the Future (FTF) \
Population-Based Survey (PBS) reports. These are impact evaluation surveys \
commissioned by USAID to measure food security, nutrition, and poverty \
indicators across 20 countries.

Answer rules:
- Draw ONLY from the provided document excerpts. If the answer is not in the \
excerpts, say so explicitly.
- Use inline citation markers like [1], [2] that correspond to the numbered \
sources in the excerpt block.
- Be precise about country, phase (Phase 1 or Phase 2), and survey round \
(baseline, interim, midline, endline) when citing data.
- When comparing across countries or time periods, organize your answer clearly.
- Write only your final answer — do not show your reasoning steps or restate \
the question.\
"""

# ── Human message ─────────────────────────────────────────────────────────────

HUMAN_MESSAGE = """\
Relevant document excerpts:

{context}

Question: {question}\
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MESSAGE),
    ("human", HUMAN_MESSAGE),
])

# ── Conversation history variant ──────────────────────────────────────────────

HUMAN_MESSAGE_WITH_HISTORY = """\
Previous conversation:
{history}

---
Relevant document excerpts:

{context}

Current question: {question}\
"""

RAG_PROMPT_WITH_HISTORY = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MESSAGE),
    ("human", HUMAN_MESSAGE_WITH_HISTORY),
])
