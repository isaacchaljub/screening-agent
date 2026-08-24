"""Real OpenAI provider — `client.responses.create`/`.parse`, the Responses API GPT-5.6
`luna`/`terra` require. A materially different calling convention from Chat Completions (Groq,
`providers/chat_completions.py`), confirmed by introspecting the installed `openai` SDK's own
`resources/responses/responses.py` and `types/responses/` on 2026-08-24:

- `client.responses.create(...)` / `client.responses.parse(...)`, not `.chat.completions.*`.
- System prompt is a top-level `instructions` kwarg — like Anthropic's `system`
  (`providers/anthropic.py`), *unlike* Chat Completions' `{"role": "system", ...}` message.
- `messages` become `input`: a list of `{"role", "content"}` dicts, same shape as Chat Completions
  minus the system entry (which moves to `instructions` instead).
- Output-token cap is `max_output_tokens`, not Chat Completions' `max_completion_tokens`.
- Structured output: `.parse(text_format=<pydantic model>)` → `response.output_parsed`, the
  validated instance directly (`openai.types.responses.parsed_response.ParsedResponse
  .output_parsed`) — no manual schema-building or JSON parsing needed here, unlike Groq's
  best-effort mode (`providers/chat_completions.py`).
- Reasoning is a nested `reasoning={"effort": "low"}` dict (`openai.types.shared.Reasoning`), not
  Chat Completions' flat `reasoning_effort` string. The SDK's own docstring scopes `Reasoning` to
  "gpt-5 and o-series models only" — the GPT-5.6 family this module targets qualifies.
- Errors: same `openai` SDK client as `chat_completions.py`, so the same exception classes apply —
  `openai.RateLimitError`/`InternalServerError`/`APIConnectionError` map to `TransportError` (R5).
- No embeddings endpoint via Responses API — embeddings stay on Google in dev (§5).

Per R4, the client is built with `max_retries=0` — this module's own `TransportError` +
`llm/retry.py` is the only retry layer.

**Not live-verified.** Unlike `providers/anthropic.py` and the Groq half of
`providers/chat_completions.py`, no call here has actually been run against `OPENAI_API_KEY` yet
(§2's docs-verification step). Everything above is read out of the installed SDK's own method and
type signatures rather than guessed from training-data recall, but "matches the SDK's declared
shape" and "verified against a live response" are different claims — run a real call before
trusting this path outside `dev`.
"""

from __future__ import annotations

from typing import Any

import openai
from openai import OpenAI
from pydantic import BaseModel

from screening_agent import config
from screening_agent.llm.base import EmbedResult, Message, SchemaError, StructuredResult, TextResult
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
    usage = response.usage
    return TextResult(
        text=response.output_text,
        model=spec.model_id,
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
    parsed = response.output_parsed
    if parsed is None:
        raise SchemaError(f"{spec.vendor} did not return a parsed {schema.__name__}")
    usage = response.usage
    return StructuredResult(
        data=parsed,
        model=spec.model_id,
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
