"""Chat-Completions-shaped provider — any vendor whose endpoint speaks the OpenAI Chat Completions
dialect (`client.chat.completions.create`/`.parse`) rather than the newer Responses API. Currently
that's Groq only; real OpenAI moved to `providers/openai.py` once M9 confirmed GPT-5.6
`luna`/`terra` require Responses API instead (see that module's docstring, and `registry.py`'s).
This module is named for the calling convention, not a vendor, so a future OpenAI-compatible Chat
Completions vendor belongs here too:

- **Groq** — confirmed live against `console.groq.com/docs/{openai,models,structured-outputs}` on
  2026-08-24: `openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)`,
  and the rest of the OpenAI Python SDK works as documented. Structured output is a
  `response_format={"type": "json_schema", "json_schema": {...}}` chat-completions param, built
  by hand here in **best-effort mode** rather than via the SDK's `.parse(response_format=<pydantic
  model>)` convenience — that builds an OpenAI *strict*-mode schema (every field required,
  `additionalProperties:false`), which live-verified 400s on Groq for any `Optional[SomeEnum]`
  field (e.g. `ExtractedFields.language`): Groq's strict-mode validator rejects the
  anyOf-with-null shape pydantic naturally produces there, where OpenAI's own backend tolerates
  it. Best-effort mode has no such requirement and is supported on the same models. Structured
  output at all is currently only supported on the GPT-OSS 20B/120B models (see
  `registry.GROQ_EVAL_MODEL`) — not a free choice, the only ones that support it. System prompt is
  a `{"role": "system", ...}` message, not a config field. Output-token cap is
  `max_completion_tokens` (the modern name; Groq accepts it). Errors:
  `openai.RateLimitError`/`InternalServerError`/`APIConnectionError` map to `TransportError`
  (R5); a schema failure (Groq returned invalid or non-conforming JSON, more likely than on
  strict-schema vendors given best-effort mode's weaker guarantee) surfaces as `SchemaError`, same
  as `providers/google.py` — which per R5 retries the *same* model with the parse error appended,
  not a vendor fallback.

Per R4, every client is built with `max_retries=0` — this module's own `TransportError` +
`llm/retry.py` is the only retry layer; the SDK's own would multiply attempts silently.
"""

from __future__ import annotations

from typing import Any

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from screening_agent import config
from screening_agent.llm.base import EmbedResult, Message, SchemaError, StructuredResult, TextResult
from screening_agent.llm.registry import ModelSpec
from screening_agent.llm.retry import TransportError

_clients: dict[tuple[str, str | None], OpenAI] = {}

_TRANSPORT_ERRORS = (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError)


def _api_key_for(vendor: str) -> str | None:
    if vendor == "groq":
        return config.GROQ_API_KEY
    return None


def _get_client(spec: ModelSpec) -> OpenAI:
    key = (spec.vendor, spec.base_url)
    client = _clients.get(key)
    if client is None:
        api_key = _api_key_for(spec.vendor)
        if not api_key:
            raise RuntimeError(
                f"no API key configured for vendor {spec.vendor!r} "
                f"({spec.vendor.upper()}_API_KEY is not set)"
            )
        client = OpenAI(api_key=api_key, base_url=spec.base_url, max_retries=0)
        _clients[key] = client
    return client


def _messages_payload(system: str | None, messages: list[Message]) -> list[dict[str, str]]:
    payload = [{"role": "system", "content": system}] if system is not None else []
    payload.extend({"role": m.role, "content": m.content} for m in messages)
    return payload


def complete_text(
    spec: ModelSpec, *, messages: list[Message], params: dict[str, Any]
) -> TextResult:
    client = _get_client(spec)
    call_params = dict(params)
    system = call_params.pop("system", None)
    call_params.pop("response_format", None)  # not meaningful for a plain-text call
    try:
        response = client.chat.completions.create(
            model=spec.model_id, messages=_messages_payload(system, messages), **call_params
        )
    except _TRANSPORT_ERRORS as exc:
        raise TransportError(f"{spec.vendor} transport error: {exc}", original=exc) from exc
    usage = response.usage
    return TextResult(
        text=response.choices[0].message.content or "",
        model=spec.model_id,
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
    )


def complete_structured(
    spec: ModelSpec, *, messages: list[Message], schema: type[BaseModel], params: dict[str, Any]
) -> StructuredResult:
    client = _get_client(spec)
    call_params = dict(params)
    system = call_params.pop("system", None)
    call_params.pop("response_format", None)  # built manually below, not via .parse()
    # `.parse(response_format=<pydantic model>)` builds an OpenAI *strict*-mode schema (every
    # field required, additionalProperties:false) — live-verified this breaks on Groq for a
    # schema with an `Optional[SomeEnum]` field (e.g. `ExtractedFields.language`): Groq's
    # strict-mode validator rejects the anyOf-with-null shape pydantic naturally produces there
    # ("BadRequestError: anyOf branches must be disambiguated..."), where OpenAI's own backend
    # tolerates it. Groq's *best-effort* mode (no `strict`) has no such requirement and is
    # supported on the same GPT-OSS models — so build the schema by hand and parse the response
    # ourselves instead of using `.parse()`.
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
    }
    try:
        response = client.chat.completions.create(
            model=spec.model_id,
            messages=_messages_payload(system, messages),
            response_format=response_format,
            **call_params,
        )
    except _TRANSPORT_ERRORS as exc:
        raise TransportError(f"{spec.vendor} transport error: {exc}", original=exc) from exc
    content = response.choices[0].message.content
    try:
        parsed = schema.model_validate_json(content) if content else None
    except ValidationError as exc:
        raise SchemaError(f"{spec.vendor} returned invalid {schema.__name__} JSON: {exc}") from exc
    if parsed is None:
        raise SchemaError(f"{spec.vendor} did not return a parsed {schema.__name__}")
    usage = response.usage
    return StructuredResult(
        data=parsed,
        model=spec.model_id,
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
    )


def embed(spec: ModelSpec, *, texts: list[str], params: dict[str, Any]) -> EmbedResult:
    # Groq has no embedding endpoint — embeddings stay on Google in dev per §5, which is the
    # only path anything in this repo actually calls `.embed()` through.
    raise NotImplementedError(
        f"embed() is not implemented for vendor {spec.vendor!r} — see providers/google.py, "
        "the only embed path this repo uses"
    )
