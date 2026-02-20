"""Prompt templates for the RAG chain.

Uses a chain-of-thought structure that instructs the model to:
  1. Identify what the question asks (countries, indicators, time periods)
  2. Reason over the provided document excerpts
  3. Synthesize a clear answer with inline citation markers
"""

from langchain_core.prompts import ChatPromptTemplate

# ── System message ────────────────────────────────────────────────────────────

SYSTEM_MESSAGE = """\
You are an expert analyst specializing in Feed the Future (FTF) \
Population-Based Survey (PBS) reports. These are impact evaluation surveys \
commissioned by USAID to measure food security, nutrition, and poverty \
indicators across 20 countries.

When answering questions:
- Draw ONLY from the provided document excerpts.
- If the answer is not in the excerpts, say so explicitly.
- Use inline citation markers like [1], [2] that correspond to the \
numbered sources listed at the end of each excerpt block.
- Be precise about country, phase (Phase 1 or Phase 2), and survey round \
(baseline, interim, midline, endline) when citing data.
- When comparing across countries or time periods, organize your answer clearly.
"""

# ── Human message (chain-of-thought structure) ────────────────────────────────

HUMAN_MESSAGE = """\
Question: {question}

---
Relevant document excerpts (each labelled with a citation number):

{context}
---

Please answer the question following these steps:

**Step 1 – Understand the question:**
What specific countries, time periods, indicators, or survey rounds is this \
question asking about?

**Step 2 – Review the evidence:**
Which of the above excerpts are most directly relevant? Note any important \
figures, findings, or limitations.

**Step 3 – Answer:**
Provide a clear, accurate, well-organized answer. Use inline citations [1], \
[2], etc. where you draw on specific excerpts. If the data is unavailable \
or the excerpts are insufficient, say so.\
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
Current question: {question}

Relevant document excerpts:

{context}
---

**Step 1 – Understand the question (considering prior conversation):**
What is being asked, and does the history provide useful context?

**Step 2 – Review the evidence:**
Which excerpts are most relevant?

**Step 3 – Answer:**
Provide a clear answer with inline citations [1], [2], etc.\
"""

RAG_PROMPT_WITH_HISTORY = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MESSAGE),
    ("human", HUMAN_MESSAGE_WITH_HISTORY),
])
