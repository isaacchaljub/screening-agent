"""Anthropic (Claude) provider.

The system prompt is a top-level `system` kwarg to `messages.create`/`.parse()` — not a message in
the `messages` list, unlike Chat Completions' shape (Groq, `providers/chat_completions.py`). Real
OpenAI's Responses API (`providers/openai.py`) matches this top-level-kwarg pattern too (its own
`instructions` param), so the "unlike OpenAI" framing is really "unlike Chat Completions" — the
calling convention that differs, not the vendor. `max_retries=0` on the client (R4 — the SDK's own
retry disabled, since `llm/retry.py` is the only retry layer). No sampling params
(`temperature`/`top_p`/`top_k`) on current-generation models; `llm/params.py`'s builder never
emits them.
"""

from __future__ import annotations

from typing import Any

import anthropic as anthropic_sdk
import pydantic
from pydantic import BaseModel

from screening_agent import config
from screening_agent.llm.base import (
    EmbedResult,
    Message,
    SchemaError,
    StructuredResult,
    TextResult,
    TruncatedResponseError,
)
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
    # A thinking model that spends its whole budget reasoning returns content with no text block
    # at all. Reading `""` out of that and sending it to the candidate as the agent's reply is a
    # silent failure — fail loudly instead (see base.TruncatedResponseError).
    if response.stop_reason == "max_tokens" and not text.strip():
        raise TruncatedResponseError(
            f"anthropic hit max_tokens ({call_params.get('max_tokens')}) before producing any "
            f"text — likely spent the budget on reasoning tokens"
        )
    usage = response.usage
    return TextResult(
        text=text,
        model=spec.full_name,
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
        # `.parse()` validates the JSON itself, so a response truncated mid-object surfaces from
        # inside the SDK as a pydantic error, not as anything Anthropic-shaped. Map it to
        # SchemaError so R5's retry-the-same-model path can catch it instead of the turn dying.
        if isinstance(exc, pydantic.ValidationError):
            raise SchemaError(f"anthropic returned invalid {schema.__name__} JSON: {exc}") from exc
        raise
    parsed = response.parsed_output
    if parsed is None:
        if response.stop_reason == "max_tokens":
            raise TruncatedResponseError(
                f"anthropic hit max_tokens ({call_params.get('max_tokens')}) before finishing "
                f"{schema.__name__} — likely spent the budget on reasoning tokens"
            )
        raise SchemaError(f"anthropic did not return a parsed {schema.__name__}")
    usage = response.usage
    return StructuredResult(
        data=parsed,
        model=spec.full_name,
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
    )


def embed(spec: ModelSpec, *, texts: list[str], params: dict[str, Any]) -> EmbedResult:
    raise NotImplementedError(
        "anthropic has no embeddings endpoint — see providers/google.py, the only embed path "
        "this repo uses"
    )
