"""FastAPI app (M4): POST /api/chat, GET /api/conversations/{id}, the static web chat UI, health.

    uvicorn screening_agent.api:app --reload

Conversations live in an in-memory, module-level dict for this single-process dev/demo server —
`Conversation` carries turn-scoped state (attempts, history, language) that isn't fully persisted
to SQLite on its own, so it can't be reconstructed from the DB alone on a cold request. A finished
conversation's structured data and transcript *are* durable, in `data/screening.db` and its JSON
export; only an in-flight one is lost on restart. Scaling this past one process — sticky sessions,
or moving conversation state into Redis/the DB — is a `docs/deployment.md` design question (M10).

`lifespan` (below) warms the local FAQ embedding model at startup instead of on the first
candidate question — see its docstring, and `rag/retrieve.py::warmup`, for why.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from google.genai import errors as genai_errors
from pydantic import BaseModel

from screening_agent.engine import Conversation
from screening_agent.llm.client import LLMClient
from screening_agent.llm.retry import TransportError
from screening_agent.rag.retrieve import warmup as _warmup_faq_index
from screening_agent.store import Store

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"

_store = Store()
_client: LLMClient | None = None
_conversations: dict[str, Conversation] = {}
_faq_ready = False  # set by `lifespan` below; reported at GET /api/health


def _get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warms the local FAQ embedding model and opens the Chroma collection handle before the
    server starts accepting requests, so the *first* candidate to ask an FAQ question mid-turn
    isn't the one who pays the ~6s model-load cost (`rag/retrieve.py::warmup`,
    `llm/providers/local.py`). uvicorn does not serve requests until this coroutine reaches
    `yield`, so that cost lands here, once, at boot — not on a live conversation.

    `_get_client()` is deliberately outside the try/except below: constructing an `LLMClient`
    runs R7's free-tier check (`config.assert_model_allowed`), and that must keep failing startup
    loudly — it's a privacy rule, not a nice-to-have. Only the FAQ warm-up itself is allowed to
    fail quietly: FAQ retrieval is a degradable feature (the screening flow works without it),
    so a broken or missing index shouldn't take the whole server down with it. `/api/health`
    reports the outcome so a degraded FAQ path is loud rather than silently broken.
    """
    global _faq_ready
    client = _get_client()
    try:
        _warmup_faq_index(client)
    except Exception:
        logger.warning(
            "FAQ retrieval warm-up failed; the screening flow will continue without it and fall "
            "back to loading the embedding model lazily on the first FAQ question",
            exc_info=True,
        )
        _faq_ready = False
    else:
        _faq_ready = True
        logger.info("FAQ retrieval warmed up and ready")
    yield


app = FastAPI(title="Grupo Sazón Screening Agent", lifespan=lifespan)


@app.exception_handler(TransportError)
async def _handle_transport_error(request: Request, exc: TransportError) -> JSONResponse:
    # Retry + fallback (R4/R5) already exhausted every option before this reached us.
    logger.error("transport error, retries/fallback exhausted, on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "The screening agent is temporarily unavailable. Please try again shortly."
        },
    )


@app.exception_handler(genai_errors.APIError)
async def _handle_vendor_api_error(request: Request, exc: genai_errors.APIError) -> JSONResponse:
    # A non-transport vendor error (e.g. a permission/config problem) — not retried per R5, but
    # still not something the candidate should see a raw stack trace for.
    logger.error("vendor API error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "The screening agent hit a provider error. Please try again shortly."},
    )


@app.exception_handler(Exception)
async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong on our end."})


class ChatRequest(BaseModel):
    conversation_id: str | None = None  # omit to start a new conversation
    message: str | None = None  # omit on the very first request


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    finished: bool
    outcome: str | None = None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "faq_index": "ready" if _faq_ready else "unavailable"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if request.conversation_id is None:
        conversation = Conversation(store=_store, client=_get_client())
        _conversations[conversation.id] = conversation
        reply = conversation.start()
        return ChatResponse(conversation_id=conversation.id, reply=reply, finished=False)

    conversation = _conversations.get(request.conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="unknown conversation_id")
    if conversation.finished:
        raise HTTPException(
            status_code=409, detail="conversation already reached a terminal outcome"
        )
    if not request.message:
        raise HTTPException(status_code=422, detail="message is required after the first turn")

    reply = conversation.step(request.message)
    return ChatResponse(
        conversation_id=conversation.id,
        reply=reply,
        finished=conversation.finished,
        outcome=conversation.outcome.value if conversation.outcome else None,
    )


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict:
    try:
        record = _store.get(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown conversation_id") from None
    return {
        "id": record.id,
        "stage": record.stage,
        "outcome": record.outcome,
        "disqualify_reason": record.disqualify_reason,
        "language": record.language,
        "profile": record.profile,
        "transcript": record.transcript,
    }


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
