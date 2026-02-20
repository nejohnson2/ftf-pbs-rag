# FTF PBS RAG System

A Retrieval-Augmented Generation (RAG) system for exploring **Feed the Future (FTF) Population-Based Survey (PBS)** reports across 20 countries. Built with FastAPI + HTMX, LangChain, pgvector, and Ollama.

---

## Architecture

The system is cleanly separated into three independent layers:

```
┌─────────────────────────────────────────────────────────────┐
│  INGESTION  (run locally, one time)                          │
│                                                             │
│  PDF Archive → Docling OCR → Markdown → pgvector           │
│                               ↑                             │
│                     manually edit these files               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  RETRIEVAL  (at query time)                                  │
│                                                             │
│  Query → Entity extraction → Metadata filter               │
│        → Semantic search (pgvector) ─┐                     │
│        → BM25 keyword search ────────┴→ RRF fusion         │
│                                       → (optional rerank)  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  GENERATION  (at query time)                                 │
│                                                             │
│  Retrieved chunks → Chain-of-thought prompt → Ollama LLM   │
│                  → Answer + structured citations            │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
ftf-pbs-rag/
├── config.yaml                     # All tunable parameters
├── config.py                       # Typed config loader (Pydantic)
├── requirements.txt                # Heroku runtime dependencies
├── requirements-ingest.txt         # Preprocessing dependencies (local only)
├── Procfile                        # Heroku process definitions
├── runtime.txt                     # Python version for Heroku
├── .env.example                    # Environment variable template
│
├── data/
│   ├── Archived Population-Based Survey Reports/  ← original PDFs (untouched)
│   ├── processed/
│   │   ├── metadata.json           # Document registry (YOU EDIT THIS)
│   │   └── markdown/               # Editable markdown files (YOU EDIT THESE)
│   └── links.json                  # Optional: doc_id → URL map
│
├── ingestion/
│   ├── scanner.py                  # Extracts metadata from folder structure
│   ├── ocr.py                      # Docling PDF → markdown extraction
│   ├── preprocessor.py             # Strips TOC, references, appendices
│   └── build_index.py              # Chunks markdown → embeds → pgvector
│
├── retrieval/
│   ├── embeddings.py               # Embedding model loader (Ollama or ST)
│   ├── vector_store.py             # pgvector operations
│   ├── bm25_index.py               # In-memory BM25 keyword index
│   ├── query_analyzer.py           # Entity extraction from user queries
│   ├── hybrid_retriever.py         # Semantic + BM25 + RRF fusion
│   └── reranker.py                 # Cross-encoder reranking (optional)
│
├── generation/
│   ├── ollama_client.py            # Ollama LLM wrapper
│   ├── prompts.py                  # Chain-of-thought prompt templates
│   ├── chain.py                    # Full RAG pipeline
│   └── citations.py                # Citation formatting
│
├── app/
│   ├── main.py                     # FastAPI app + startup/shutdown lifecycle
│   ├── session.py                  # Cookie-based anonymous sessions
│   ├── logging_middleware.py       # Query/response audit logging
│   ├── routes/
│   │   ├── chat.py                 # POST /chat, GET /
│   │   └── documents.py            # GET /documents
│   ├── templates/                  # Jinja2 + HTMX templates
│   └── static/                     # CSS + JS
│
├── scripts/
│   ├── preprocess.py               # Step 1: OCR → markdown + metadata stub
│   ├── build_index.py              # Step 2: markdown → pgvector
│   ├── db_migrate.py               # Heroku release: create DB tables
│   └── test_retrieval.py           # Developer tool: test queries
│
└── logs/
    └── queries.jsonl               # Newline-delimited query audit log
```

---

## Prerequisites

- Python 3.12+
- PostgreSQL with pgvector extension (local or Heroku)
- Ollama running at a remote URL with:
  - A chat model (e.g. `llama3.2`)
  - An embedding model (e.g. `nomic-embed-text`)

---

## Workflow

### Phase 1 — Preprocessing (run locally, one time)

**1. Install ingestion dependencies**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-ingest.txt
```

**2. Copy and configure environment**

```bash
cp .env.example .env
# Edit .env — set DATABASE_URL, OLLAMA_BASE_URL, SECRET_KEY
```

**3. Run database migration**

```bash
python scripts/db_migrate.py
```

**4. Run OCR and generate markdown files**

```bash
python scripts/preprocess.py
```

This will:
- Scan the `Archived Population-Based Survey Reports/` folder
- Extract metadata from the path structure (country, phase, survey type)
- Write a `data/processed/metadata.json` stub
- Run Docling OCR on each PDF and write cleaned markdown to `data/processed/markdown/`

**5. Edit `data/processed/metadata.json`**

For each document, fill in:
```json
{
  "title": "Bangladesh FtF ZOI Phase 1 Baseline Survey",
  "online_url": "https://www.dropbox.com/your-link",
  "include": true
}
```

Set `"include": false` for any document you want to exclude from the index.

**6. (Optional) Edit the markdown files**

The markdown files in `data/processed/markdown/` are the source of truth for what gets indexed. You can:
- Delete sections you don't want indexed
- Correct OCR errors
- Add clarifying notes

**7. Build the vector index**

```bash
python -m ingestion.build_index
```

Use `--clear` to wipe and rebuild from scratch:

```bash
python -m ingestion.build_index --clear
```

**8. Test retrieval locally**

```bash
python scripts/test_retrieval.py "What was the stunting rate in Kenya at endline?"
```

---

### Phase 2 — Running the App

**Local development**

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000

**Heroku deployment**

```bash
heroku create your-app-name
heroku addons:create heroku-postgresql:essential-0

heroku config:set OLLAMA_BASE_URL=http://your-ollama-server:11434
heroku config:set OLLAMA_MODEL=llama3.2
heroku config:set OLLAMA_EMBED_MODEL=nomic-embed-text
heroku config:set SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")

git push heroku main
```

The `release` phase in `Procfile` automatically runs `scripts/db_migrate.py` on each deploy to create/update database tables.

**Push the vector index to Heroku**

After building the index locally, the vectors are already in the database (since both local and Heroku use the same `DATABASE_URL`). No additional step is needed.

If you used a local database for development and want to push to Heroku's database:

```bash
heroku pg:push <local-db-name> DATABASE_URL
```

---

## Configuration

All tunable parameters are in [config.yaml](config.yaml). Key settings:

| Section | Key | Description |
|---|---|---|
| `ollama` | `base_url` | Ollama server URL (override with `OLLAMA_BASE_URL`) |
| `ollama` | `model` | Chat model name (override with `OLLAMA_MODEL`) |
| `ollama` | `embed_model` | Embedding model for Ollama provider |
| `embeddings` | `provider` | `"ollama"` (default) or `"sentence_transformers"` |
| `retrieval` | `chunk_size` | Characters per text chunk (default: 800) |
| `retrieval` | `hybrid_alpha` | `0.0` = pure BM25, `1.0` = pure semantic (default: 0.5) |
| `retrieval` | `top_k_reranked` | Final chunks sent to LLM (default: 5) |
| `retrieval` | `enable_reranker` | Cross-encoder reranking — requires torch, off by default |
| `session` | `max_history_turns` | Conversation turns kept per session (default: 8) |
| `preprocessing` | `remove_toc` | Strip table of contents from OCR output |
| `preprocessing` | `remove_references_section` | Strip references/bibliography |
| `preprocessing` | `remove_appendices` | Strip appendix sections |

---

## Document Metadata Schema

`data/processed/metadata.json` has one entry per document:

```json
{
  "doc_id": "kenya_p1_baseline_full_report",
  "file_path": "Archived .../Kenya/Phase I/Baseline/Northern Kenya Baseline report 2014.pdf",
  "markdown_path": "data/processed/markdown/kenya_p1_baseline_full_report.md",
  "country": "Kenya",
  "phase": 1,
  "survey_type": "baseline",
  "doc_type": "full_report",
  "year": 2014,
  "title": "Northern Kenya FtF ZOI Baseline Survey 2014",
  "online_url": "https://www.dropbox.com/...",
  "include": true,
  "notes": ""
}
```

`doc_type` values: `full_report`, `key_findings`, `reference`, `planning`

`survey_type` values: `baseline`, `interim`, `midline`, `endline`, `baseline_midline`

---

## Retrieval Pipeline Detail

1. **Query analysis** — Rule-based extraction of country names, phase numbers, survey types, and year ranges from the user's question
2. **Metadata pre-filter** — pgvector `WHERE` clause narrows the search space (e.g. if user asks about Kenya, only Kenya chunks are searched)
3. **Semantic search** — Cosine similarity over Ollama embeddings stored in pgvector
4. **BM25 keyword search** — In-memory index rebuilt at startup from all corpus chunks; handles acronyms (ZOI, PBS, RFZ) and exact term matching
5. **Reciprocal Rank Fusion** — Merges semantic + BM25 ranked lists without needing calibrated scores
6. **Cross-encoder reranking** (optional) — Re-scores the merged candidates with a cross-encoder model; enable with `enable_reranker: true` in config
7. **Chain-of-thought generation** — Structured prompt asks the LLM to reason step-by-step before synthesizing an answer

---

## Logging

Every query is logged to:
- `logs/queries.jsonl` — newline-delimited JSON, one record per query
- `query_logs` database table — for querying via SQL

Log record structure:
```json
{
  "ts": "2024-01-15T14:32:00Z",
  "session_id": "uuid",
  "query": "user's question",
  "answer": "LLM response",
  "entities": {"countries": ["Kenya"], "phases": [1], "survey_types": ["baseline"]},
  "docs_cited": [{"doc_id": "...", "title": "...", "country": "Kenya"}]
}
```

---

## Heroku Constraints

| Issue | Solution |
|---|---|
| Ephemeral filesystem | All data in Heroku Postgres (pgvector). BM25 rebuilt in memory at startup. |
| torch too large for slug | Embeddings via Ollama API (no torch needed at runtime). Reranker disabled by default. |
| Model downloads | No ML models downloaded at runtime — embeddings are Ollama API calls. |
| Dyno RAM | Standard-2x (1 GB) recommended. Increase if BM25 index is large. |

---

## Re-ingesting After Edits

To update the index after editing markdown files or metadata:

```bash
# Re-ingest a single document (delete its chunks first is handled by --clear):
python -m ingestion.build_index --clear

# Or: only re-run OCR for changed documents:
python scripts/preprocess.py --force
python -m ingestion.build_index --clear
```
