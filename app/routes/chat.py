"""Chat routes.

POST /chat  — submit a question, receive an HTML fragment (HTMX swap)
GET  /      — serve the main chat page
"""

import json
from typing import Annotated, Optional

from fastapi import APIRouter, Cookie, Form, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from loguru import logger

from app import logging_middleware, session as session_mgr
from config import get_config

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


def _get_or_create_session(
    response: Response,
    cookie_value: Optional[str],
) -> str:
    """Return existing verified session_id or create a fresh one."""
    cfg = get_config()
    if cookie_value:
        sid = session_mgr.verify_session_cookie(cookie_value)
        if sid:
            return sid

    # New session
    sid = session_mgr.create_session_id()
    signed = session_mgr.sign_session_id(sid)
    response.set_cookie(
        key=cfg.session.cookie_name,
        value=signed,
        max_age=cfg.session.cookie_max_age,
        httponly=True,
        samesite="lax",
    )
    return sid


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    cfg = get_config()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": cfg.app.title, "description": cfg.app.description},
    )


@router.post("/chat", response_class=HTMLResponse)
async def chat(
    request: Request,
    response: Response,
    query: Annotated[str, Form()],
    ftf_session: Annotated[Optional[str], Cookie()] = None,
):
    cfg = get_config()
    session_id = _get_or_create_session(response, ftf_session)

    # Get RAG chain from app state (initialized in main.py lifespan)
    rag_chain = request.app.state.rag_chain

    # Load history
    history = session_mgr.get_history(session_id, cfg.session.max_history_turns)

    try:
        result = await rag_chain.invoke(query, history)
        answer = result.answer
        citations = result.citations
        error = None
    except Exception as e:
        logger.error(f"RAG chain error: {e}")
        answer = "An error occurred while processing your question. Please try again."
        citations = []
        error = str(e)
        result = None

    # Persist conversation turn
    session_mgr.append_history(session_id, "user", query)
    session_mgr.append_history(session_id, "assistant", answer)

    # Log query
    if result:
        docs_cited = [
            {"doc_id": c.doc_id, "title": c.title, "country": c.country}
            for c in citations
        ]
        logging_middleware.log_query(
            session_id=session_id,
            query=query,
            answer=answer,
            entities=result.query_entities,
            docs_cited=docs_cited,
        )

    # Return HTML fragment for HTMX to append
    return templates.TemplateResponse(
        "partials/message.html",
        {
            "request": request,
            "query": query,
            "answer": answer,
            "citations": citations,
            "error": error,
        },
    )
