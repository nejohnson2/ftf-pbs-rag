"""Documents route — exposes the document registry as JSON and HTML.

GET /documents        — JSON list of all included documents with metadata
GET /documents/list   — HTML fragment (HTMX) for rendering in the UI
"""

import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from config import get_config

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


def _load_documents() -> list[dict]:
    cfg = get_config()
    meta_path = Path(cfg.preprocessing.metadata_output)
    if not meta_path.exists():
        return []
    with open(meta_path) as f:
        data = json.load(f)
    return [d for d in data.get("documents", []) if d.get("include", True)]


@router.get("/documents", response_class=JSONResponse)
async def list_documents_json():
    docs = _load_documents()
    return JSONResponse(content={"documents": docs, "total": len(docs)})


@router.get("/documents/list", response_class=HTMLResponse)
async def list_documents_html(request: Request):
    docs = _load_documents()
    # Group by country for display
    by_country: dict[str, list] = {}
    for doc in docs:
        country = doc.get("country", "Unknown")
        by_country.setdefault(country, []).append(doc)

    return templates.TemplateResponse(
        "partials/document_list.html",
        {"request": request, "by_country": by_country},
    )
