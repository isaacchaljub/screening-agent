"""FastAPI app: POST /api/chat, GET /api/conversations/{id}, the static web chat UI, health.

    uvicorn screening_agent.api:app --reload

Conversations live in an in-memory, module-level dict for this single-process dev/demo server —
`Conversation` carries turn-scoped state (attempts, history, language) that isn't fully persisted
to SQLite on its own, so it can't be reconstructed from the DB alone on a cold request. A finished
conversation's structured data and transcript *are* durable, in `data/screening.db` and its JSON
export; only an in-flight one is lost on restart. Scaling this past one process — sticky sessions,
or moving conversation state into Redis/the DB — is a `docs/deployment.md` design question.

`lifespan` (below) warms the local FAQ embedding model at startup instead of on the first
candidate question — see its docstring, and `rag/retrieve.py::warmup`, for why.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from elevenlabs.core.api_error import ApiError as ElevenLabsApiError
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from google.genai import errors as genai_errors
from pydantic import BaseModel

from screening_agent import config
from screening_agent.engine import Conversation
from screening_agent.llm.client import LLMClient
from screening_agent.llm.retry import TransportError
from screening_agent.rag.retrieve import warmup as _warmup_faq_index
from screening_agent.store import Store
from screening_agent.voice import elevenlabs as voice_provider

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
    server starts accepting requests — see README "Embeddings run locally" for why. uvicorn does
    not serve requests until this coroutine reaches `yield`.

    `_get_client()` is deliberately outside the try/except below: constructing an `LLMClient` runs
    R7's free-tier check, which must keep failing startup loudly — a privacy rule, not a
    nice-to-have. Only the FAQ warm-up itself is allowed to fail quietly, since FAQ retrieval is a
    degradable feature; `/api/health` reports the outcome so a degraded path is loud rather than
    silently broken.
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


@app.exception_handler(ElevenLabsApiError)
async def _handle_voice_api_error(request: Request, exc: ElevenLabsApiError) -> JSONResponse:
    # Same shape as _handle_vendor_api_error above, for the one vendor call outside llm/client.py
    # (voice/elevenlabs.py) — a non-transport ElevenLabs error (e.g. an unsupported audio format)
    # already passed voice/elevenlabs.py's own transport-error/retry handling, so it reaches here
    # as something the candidate shouldn't see a raw stack trace for either.
    logger.error("voice transcription API error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Voice input is temporarily unavailable. Please try again or type instead."
        },
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


class VoiceChatResponse(ChatResponse):
    transcript: str  # what Scribe heard — shown back so a mis-hearing is visible, not silent


def _get_active_conversation(conversation_id: str) -> Conversation:
    """Shared by /api/chat and /api/chat/voice: the same conversation must exist and still be
    in flight — text and voice are two input channels onto one `Conversation`, not two flows."""
    conversation = _conversations.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="unknown conversation_id")
    if conversation.finished:
        raise HTTPException(
            status_code=409, detail="conversation already reached a terminal outcome"
        )
    return conversation


@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "faq_index": "ready" if _faq_ready else "unavailable",
        # The browser UI hides the mic button when this is "unavailable" rather than showing a
        # button that would 503 on every use — voice input is optional, not a hard dependency.
        "voice_input": "ready" if config.ELEVENLABS_API_KEY else "unavailable",
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if request.conversation_id is None:
        conversation = Conversation(store=_store, client=_get_client())
        _conversations[conversation.id] = conversation
        reply = conversation.start()
        return ChatResponse(conversation_id=conversation.id, reply=reply, finished=False)

    conversation = _get_active_conversation(request.conversation_id)
    if not request.message:
        raise HTTPException(status_code=422, detail="message is required after the first turn")

    reply = conversation.step(request.message)
    return ChatResponse(
        conversation_id=conversation.id,
        reply=reply,
        finished=conversation.finished,
        outcome=conversation.outcome.value if conversation.outcome else None,
    )


@app.post("/api/chat/voice", response_model=VoiceChatResponse)
async def chat_voice(
    conversation_id: str = Form(...),  # noqa: B008 — FastAPI's own idiom for a form field default
    audio: UploadFile = File(...),  # noqa: B008 — same, for a multipart file field
) -> VoiceChatResponse:
    """Voice input only — there is no voice equivalent of the very first `/api/chat` call, since
    the GREETING message needs no candidate input to transcribe. See `voice/elevenlabs.py` for why
    voice *output* isn't offered here: the reply is text, same as the `/api/chat` path.

    An empty/silent recording transcribes to `""`, which is passed through to `conversation.step()`
    unchanged rather than rejected — `engine.py` already treats a reply that captures nothing as a
    normal failed attempt ("didn't get an answer to that"), the correct outcome for silence too.
    """
    conversation = _get_active_conversation(conversation_id)
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="audio is required")

    transcript = voice_provider.transcribe(audio_bytes, filename=audio.filename or "recording.webm")
    reply = conversation.step(transcript)
    return VoiceChatResponse(
        conversation_id=conversation.id,
        transcript=transcript,
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
