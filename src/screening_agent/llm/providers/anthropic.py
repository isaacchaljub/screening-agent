"""Anthropic (Claude) provider — M9, no longer blocked once `ANTHROPIC_API_KEY` was added.
Verified live via the `claude-api` skill and the installed `anthropic==1.0.0` SDK on 2026-08-24
(not from training-data recall, which the skill's own drift table flags as stale for this API):

- `anthropic.Anthropic(api_key=..., max_retries=0)` — R4: the SDK's own retry disabled, since
  `llm/retry.py` is the only retry layer.
- The system prompt is a top-level `system` kwarg to `messages.create`/`.parse()` — not a message
  in the `messages` list, unlike Chat Completions' shape (Groq, `providers/chat_completions.py`).
  Real OpenAI's Responses API (`providers/openai.py`) actually matches this top-level-kwarg
  pattern too (its own `instructions` param), so the "unlike OpenAI" framing is really "unlike
  Chat Completions" — the calling convention that differs, not the vendor.
- Output-token cap is `max_tokens` (top-level, required by the SDK's own signature).
- Structured output: `client.messages.parse(..., output_format=<pydantic model>)` →
  `response.parsed_output`, the validated instance directly — confirmed this handles
  `ExtractedFields`' `Optional[SomeEnum]` fields fine (unlike Groq's strict JSON-schema mode,
  which live-verified rejects that exact shape — see `providers/chat_completions.py`'s docstring).
  No manual schema-building or JSON parsing needed here, unlike the Groq path.
- No sampling params (`temperature`/`top_p`/`top_k`) — live-verified removed entirely from the
  current SDK's typed signature on current-generation models; `llm/params.py`'s builder never
  emits them.
- Errors: `RateLimitError` (429) and any `APIStatusError` with `status_code >= 500` (covers
  `InternalServerError`/`OverloadedError`/`ServiceUnavailableError`) map to `TransportError` (R5);
  `APIConnectionError` (network, including its `APITimeoutError` subclass) does too. A
  `BadRequestError`/other 4xx propagates unchanged, so `extract.py`'s own schema-retry loop (R5)
  can handle it instead of triggering a vendor fallback.
- No embeddings endpoint — same as Groq; embeddings stay on Google in dev (§5).
"""

from __future__ import annotations

from typing import Any

import anthropic as anthropic_sdk
from pydantic import BaseModel

from screening_agent import config
from screening_agent.llm.base import EmbedResult, Message, SchemaError, StructuredResult, TextResult
from screening_agent.llm.registry import ModelSpec
from screening_agent.llm.retry import TransportError

_client: anthropic_sdk.Anthropic | None = None

_TRANSPORT_ERRORS = (anthropic_sdk.RateLimitError, anthropic_sdk.APIConnectionError)


def _get_client() -> anthropic_sdk.Anthropic:
    global _client
    if _client is None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        _client = anthropic_sdk.Anthropic(api_key=config.ANTHROPIC_API_KEY, max_retries=0)
    return _client


def _messages_payload(messages: list[Message]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _is_transport_error(exc: Exception) -> bool:
    if isinstance(exc, _TRANSPORT_ERRORS):
        return True
    return isinstance(exc, anthropic_sdk.APIStatusError) and exc.status_code >= 500


def complete_text(
    spec: ModelSpec, *, messages: list[Message], params: dict[str, Any]
) -> TextResult:
    client = _get_client()
    call_params = dict(params)
    call_params.pop("response_format", None)  # not meaningful for a plain-text call
    try:
        response = client.messages.create(
            model=spec.model_id, messages=_messages_payload(messages), **call_params
        )
    except Exception as exc:
        if _is_transport_error(exc):
            raise TransportError(f"anthropic transport error: {exc}", original=exc) from exc
        raise
    text = next((block.text for block in response.content if block.type == "text"), "")
    usage = response.usage
    return TextResult(
        text=text,
        model=spec.model_id,
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
    )


def complete_structured(
    spec: ModelSpec, *, messages: list[Message], schema: type[BaseModel], params: dict[str, Any]
) -> StructuredResult:
    client = _get_client()
    call_params = dict(params)
    call_params.pop("response_format", None)  # passed explicitly below as output_format
    try:
        response = client.messages.parse(
            model=spec.model_id,
            messages=_messages_payload(messages),
            output_format=schema,
            **call_params,
        )
    except Exception as exc:
        if _is_transport_error(exc):
            raise TransportError(f"anthropic transport error: {exc}", original=exc) from exc
        raise
    parsed = response.parsed_output
    if parsed is None:
        raise SchemaError(f"anthropic did not return a parsed {schema.__name__}")
    usage = response.usage
    return StructuredResult(
        data=parsed,
        model=spec.model_id,
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
    )


def embed(spec: ModelSpec, *, texts: list[str], params: dict[str, Any]) -> EmbedResult:
    raise NotImplementedError(
        "anthropic has no embeddings endpoint — see providers/google.py, the only embed path "
        "this repo uses"
    )
