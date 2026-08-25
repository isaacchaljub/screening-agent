"""Real OpenAI provider — `client.responses.create`/`.parse`, the Responses API GPT-5.6
`luna`/`terra` require. A materially different calling convention from Chat Completions (Groq,
`providers/chat_completions.py`):

- `client.responses.create(...)` / `client.responses.parse(...)`, not `.chat.completions.*`.
- System prompt is a top-level `instructions` kwarg — like Anthropic's `system`
  (`providers/anthropic.py`), *unlike* Chat Completions' `{"role": "system", ...}` message.
- `messages` become `input`: a list of `{"role", "content"}` dicts, same shape as Chat Completions
  minus the system entry (which moves to `instructions` instead).
- Output-token cap is `max_output_tokens`, not Chat Completions' `max_completion_tokens`.
- Structured output: `.parse(text_format=<pydantic model>)` → `response.output_parsed`, the
  validated instance directly — no manual schema-building or JSON parsing needed here, unlike
  Groq's best-effort mode (`providers/chat_completions.py`).
- Reasoning is a nested `reasoning={"effort": "low"}` dict, not Chat Completions' flat
  `reasoning_effort` string.
- Errors: same `openai` SDK client as `chat_completions.py`, so the same exception classes apply —
  `openai.RateLimitError`/`InternalServerError`/`APIConnectionError` map to `TransportError` (R5).
- No embeddings endpoint via Responses API — embeddings stay on Google in dev (§5).

Per R4, the client is built with `max_retries=0` — this module's own `TransportError` +
`llm/retry.py` is the only retry layer.

**Not in `registry.ROLES`.** The account behind `OPENAI_API_KEY` has no billing credits, so no
call through this module has completed end to end — see README "Model choice" for the detail.
"""

from __future__ import annotations

from typing import Any

import openai
import pydantic
from openai import OpenAI
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

_client: OpenAI | None = None

_TRANSPORT_ERRORS = (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _client = OpenAI(api_key=config.OPENAI_API_KEY, max_retries=0)
    return _client


def _input_payload(messages: list[Message]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _hit_token_cap(response: Any) -> bool:
    """The Responses API reports truncation as `status == "incomplete"` with an
    `incomplete_details.reason` — not as a `finish_reason` the way Chat Completions does."""
    if getattr(response, "status", None) != "incomplete":
        return False
    reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
    return reason == "max_output_tokens"


def complete_text(
    spec: ModelSpec, *, messages: list[Message], params: dict[str, Any]
) -> TextResult:
    client = _get_client()
    call_params = dict(params)
    call_params.pop("response_format", None)  # not meaningful for a plain-text call
    try:
        response = client.responses.create(
            model=spec.model_id, input=_input_payload(messages), **call_params
        )
    except _TRANSPORT_ERRORS as exc:
        raise TransportError(f"{spec.vendor} transport error: {exc}", original=exc) from exc
    if _hit_token_cap(response) and not (response.output_text or "").strip():
        raise TruncatedResponseError(
            f"{spec.vendor} hit max_output_tokens ({call_params.get('max_output_tokens')}) "
            "before producing any text — reasoning tokens count against the same cap"
        )
    usage = response.usage
    return TextResult(
        text=response.output_text,
        model=spec.full_name,
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
    )


def complete_structured(
    spec: ModelSpec, *, messages: list[Message], schema: type[BaseModel], params: dict[str, Any]
) -> StructuredResult:
    client = _get_client()
    call_params = dict(params)
    call_params.pop("response_format", None)  # passed explicitly below as text_format
    try:
        response = client.responses.parse(
            model=spec.model_id,
            input=_input_payload(messages),
            text_format=schema,
            **call_params,
        )
    except _TRANSPORT_ERRORS as exc:
        raise TransportError(f"{spec.vendor} transport error: {exc}", original=exc) from exc
    except pydantic.ValidationError as exc:
        # Same failure mode as Anthropic's `.parse()`: the SDK validates the JSON itself, so a
        # response truncated mid-object arrives as a pydantic error rather than anything
        # vendor-shaped. R5 wants that retried on the same model, not failed outright.
        raise SchemaError(f"{spec.vendor} returned invalid {schema.__name__} JSON: {exc}") from exc
    parsed = response.output_parsed
    if parsed is None:
        if _hit_token_cap(response):
            raise TruncatedResponseError(
                f"{spec.vendor} hit max_output_tokens ({call_params.get('max_output_tokens')}) "
                f"before finishing {schema.__name__}"
            )
        raise SchemaError(f"{spec.vendor} did not return a parsed {schema.__name__}")
    usage = response.usage
    return StructuredResult(
        data=parsed,
        model=spec.full_name,
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
    )


def embed(spec: ModelSpec, *, texts: list[str], params: dict[str, Any]) -> EmbedResult:
    # No embeddings endpoint via the Responses API — embeddings stay on Google in dev per §5,
    # which is the only path anything in this repo actually calls `.embed()` through.
    raise NotImplementedError(
        f"embed() is not implemented for vendor {spec.vendor!r} — see providers/google.py, "
        "the only embed path this repo uses"
    )
