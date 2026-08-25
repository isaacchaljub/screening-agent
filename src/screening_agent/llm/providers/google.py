"""Google Gemini provider (§5).

The `google-genai` SDK surface postdates this session's reliable training data (§2), so every
claim below was confirmed by introspecting the actually-installed package (`google-genai==1.60.0`,
pinned in `pyproject.toml`) rather than recalled from memory, plus a live fetch of Google's current
docs for model ids and free-tier terms:

- Calls go through `genai.Client(api_key=...)`, then `client.models.generate_content(model=,
  contents=, config=)` / `client.models.embed_content(model=, contents=, config=)`.
- The system prompt is a `system_instruction` field *inside* the config object — not a top-level
  kwarg, and not a message in `contents`.
- The output-token cap is `max_output_tokens`, also inside the config.
- Structured output: `response_mime_type="application/json"` + `response_schema=<pydantic model>`
  on the config; the parsed instance comes back on `response.parsed` (`response.text` is the raw
  text either way). Confirmed via `google.genai.types.GenerateContentResponse.parsed`'s field
  docstring: "First candidate from the parsed response if response_schema is provided."
- Reasoning/thinking is `thinking_config` (`{"thinking_level": ...}` or `{"thinking_budget": ...}`),
  also inside the config — sampling params (`temperature`, `top_p`, `top_k`) sit alongside it,
  top-level within the config, not nested further.
- Errors: `google.genai.errors.ClientError` (4xx, includes 429) / `ServerError` (5xx), both
  `APIError` subclasses with `.code` (HTTP status) and `.details` (parsed error body — where a
  429's `RetryInfo.retryDelay` lives, which `llm/retry.py` parses out).
- Embedding vectors come back as `response.embeddings[i].values` (`list[float]`).
- Model ids, confirmed live on 2026-08-24 against https://ai.google.dev/gemini-api/docs/models and
  /pricing: `gemini-3.5-flash-lite` (extract/compose dev override) and `gemini-embedding-001`
  (embeddings). Both currently sit on the free tier, whose terms say prompts may be used to
  improve Google's products — the reason `config.assert_model_allowed` (R7) gates them to dev.
"""

from __future__ import annotations

import os
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from pydantic import BaseModel

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

_ROLE_MAP = {"user": "user", "assistant": "model"}

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=api_key)
    return _client


def _contents(messages: list[Message]) -> list[dict[str, Any]]:
    return [{"role": _ROLE_MAP[m.role], "parts": [{"text": m.content}]} for m in messages]


def _generate(spec: ModelSpec, messages: list[Message], params: dict[str, Any]) -> Any:
    client = _get_client()
    try:
        return client.models.generate_content(
            model=spec.model_id, contents=_contents(messages), config=params
        )
    except genai_errors.ServerError as exc:
        raise TransportError(f"google server error: {exc}", original=exc) from exc
    except genai_errors.ClientError as exc:
        if exc.code == 429:
            raise TransportError(f"google rate limited: {exc}", original=exc) from exc
        raise  # a 400 etc. is a request/schema problem — not retried here, not a transport error
    except (TimeoutError, ConnectionError) as exc:
        raise TransportError(f"google transport error: {exc}", original=exc) from exc


def _hit_token_cap(response: Any) -> bool:
    """Google reports truncation as `finish_reason == MAX_TOKENS` on the candidate, not on the
    response — and reports it as an enum, so compare by name rather than importing the type."""
    candidate = next(iter(response.candidates or []), None)
    finish_reason = getattr(candidate, "finish_reason", None)
    return getattr(finish_reason, "name", str(finish_reason)) == "MAX_TOKENS"


def complete_text(
    spec: ModelSpec, *, messages: list[Message], params: dict[str, Any]
) -> TextResult:
    response = _generate(spec, messages, params)
    if _hit_token_cap(response) and not (response.text or "").strip():
        raise TruncatedResponseError(
            f"google hit max_output_tokens ({params.get('max_output_tokens')}) before producing "
            "any text"
        )
    usage = response.usage_metadata
    return TextResult(
        text=response.text or "",
        model=spec.full_name,
        input_tokens=getattr(usage, "prompt_token_count", None),
        output_tokens=getattr(usage, "candidates_token_count", None),
    )


def complete_structured(
    spec: ModelSpec, *, messages: list[Message], schema: type[BaseModel], params: dict[str, Any]
) -> StructuredResult:
    response = _generate(spec, messages, params)
    if response.parsed is None:
        if _hit_token_cap(response):
            raise TruncatedResponseError(
                f"google hit max_output_tokens ({params.get('max_output_tokens')}) before "
                f"finishing {schema.__name__}"
            )
        raise SchemaError(f"google did not return a parsed {schema.__name__}: {response.text!r}")
    usage = response.usage_metadata
    return StructuredResult(
        data=response.parsed,
        model=spec.full_name,
        input_tokens=getattr(usage, "prompt_token_count", None),
        output_tokens=getattr(usage, "candidates_token_count", None),
    )


def embed(spec: ModelSpec, *, texts: list[str], params: dict[str, Any]) -> EmbedResult:
    client = _get_client()
    try:
        response = client.models.embed_content(
            model=spec.model_id, contents=texts, config=params or None
        )
    except genai_errors.ServerError as exc:
        raise TransportError(f"google server error: {exc}", original=exc) from exc
    except genai_errors.ClientError as exc:
        if exc.code == 429:
            raise TransportError(f"google rate limited: {exc}", original=exc) from exc
        raise
    except (TimeoutError, ConnectionError) as exc:
        raise TransportError(f"google transport error: {exc}", original=exc) from exc
    vectors = [list(item.values or []) for item in response.embeddings or []]
    return EmbedResult(vectors=vectors, model=spec.full_name)
