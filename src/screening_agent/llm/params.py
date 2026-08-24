"""`build_params()` (§5): translates one vendor-neutral request into the exact kwargs a vendor
SDK expects, reconciling the output-token cap name, where the system prompt goes, the
reasoning/thinking parameter, the structured-output mechanism, and whether sampling parameters
are top-level or nested. `NeutralParams` is frozen and every builder returns a fresh `dict` — R6:
`ModelSpec` is frozen, `build_params()` never mutates a caller's dict, popping keys out of a
shared one works once per process and then quietly stops working.

`google` is confirmed against `google-genai==1.60.0` (installed; see `providers/google.py`).
`groq` (M8, eval sweeps) is confirmed against the installed `openai==3.3.1` SDK plus Groq's current
docs (see `providers/chat_completions.py`) — its calling code is genuinely OpenAI-compatible Chat
Completions, which is a different claim from "the real `openai` vendor's shape is verified": that
one (GPT-5.6 `luna`/`terra`) uses the Responses API instead (`providers/openai.py`), a materially
different param set built by `_build_openai_responses` below — read out of the installed SDK's own
`resources/responses/responses.py` signatures, not from an assumption that Groq's dialect and real
OpenAI's are identical.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from screening_agent.llm.registry import ModelSpec

DEFAULT_MAX_OUTPUT_TOKENS = 400  # extract's JSON schema and compose's <25-word replies both fit
# comfortably well under this — the prior 1024 default reserved far more quota per call than any
# real response needs, which matters on Groq's free-tier TPM budget (M8: 8000 tokens/min).
DEFAULT_TEMPERATURE = 0.3


@dataclass(frozen=True, slots=True)
class NeutralParams:
    system: str | None = None
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    response_schema: type[BaseModel] | None = None
    thinking: bool = False
    # Embedding calls (`LLMClient.embed`) go through a *different* vendor config shape than
    # generation calls do (confirmed live: Google's `EmbedContentConfig` rejects
    # `max_output_tokens`/`temperature` outright — `embed()` was unverified until M6). `task_type`
    # ("RETRIEVAL_DOCUMENT" at index time, "RETRIEVAL_QUERY" at query time) asymmetrically tunes
    # Gemini's embedding for retrieval quality; unset for a symmetric embedding.
    for_embedding: bool = False
    embed_task_type: str | None = None


def _build_google(neutral: NeutralParams) -> dict[str, Any]:
    if neutral.for_embedding:
        params: dict[str, Any] = {}
        if neutral.embed_task_type is not None:
            params["task_type"] = neutral.embed_task_type
        return params
    params = {
        "max_output_tokens": neutral.max_output_tokens,
        "temperature": neutral.temperature,
    }
    if neutral.system is not None:
        params["system_instruction"] = neutral.system
    if neutral.response_schema is not None:
        params["response_mime_type"] = "application/json"
        params["response_schema"] = neutral.response_schema
    if neutral.thinking:
        params["thinking_config"] = {"thinking_level": "low"}
    return params


def _build_openai_compatible(neutral: NeutralParams) -> dict[str, Any]:
    """Groq — Chat Completions shape. The system prompt and the structured-output schema can't
    live in this dict as OpenAI-shaped kwargs (system is a *message*, not a param; the schema is a
    `response_format=<pydantic model>` kwarg to `.parse()`, not a raw dict) —
    `providers/chat_completions.py` pops both back out before the real SDK call. Keeping them here
    anyway is what R6 and "provider modules only see the params dict" both require: this is the
    only place a caller's `system`/`response_schema` become visible to the provider layer at all.
    """
    if neutral.for_embedding:
        return {}  # Groq has no embedding endpoint; embeddings stay on Google in dev (§5)
    params: dict[str, Any] = {
        "max_completion_tokens": neutral.max_output_tokens,
        "temperature": neutral.temperature,
    }
    if neutral.system is not None:
        params["system"] = neutral.system
    if neutral.response_schema is not None:
        params["response_format"] = neutral.response_schema
    # Always low, not gated by `neutral.thinking` (which nothing here sets) — extract and compose
    # are both short, low-reasoning tasks, and GPT-OSS's hidden reasoning tokens count against
    # `max_completion_tokens`: a higher effort risks the response getting cut off mid-reasoning,
    # before it ever reaches the answer, on top of burning quota for no quality benefit here.
    params["reasoning_effort"] = "low"
    return params


def _build_anthropic(neutral: NeutralParams) -> dict[str, Any]:
    """Confirmed live via the `claude-api` skill + the installed `anthropic==1.0.0` SDK on
    2026-08-24 (M9): `max_tokens` is the output-token cap (top-level, required); `system` is a
    top-level kwarg, not a message; structured output is `client.messages.parse(...,
    output_format=<pydantic model>)` → `response.parsed_output` (`providers/anthropic.py` pops
    `response_format` back out and uses it there, same pattern as the OpenAI-compatible builder).
    `thinking={"type": "adaptive"}` is the current mechanism (`budget_tokens` is removed on
    current-generation models). Sampling params (`temperature`/`top_p`/`top_k`) are deliberately
    absent here — live-verified they no longer exist as parameters on `messages.create`/`.parse()`
    at all on current models (inspecting the installed SDK's own method signature confirms this,
    matching the skill's "Sampling: Removed" note); passing one raises a TypeError before any
    request is even sent.
    """
    if neutral.for_embedding:
        return {}  # Anthropic has no embeddings endpoint; embeddings stay on Google in dev (§5)
    params: dict[str, Any] = {"max_tokens": neutral.max_output_tokens}
    if neutral.system is not None:
        params["system"] = neutral.system
    if neutral.response_schema is not None:
        params["response_format"] = neutral.response_schema
    if neutral.thinking:
        params["thinking"] = {"type": "adaptive"}
    return params


def _build_openai_responses(neutral: NeutralParams) -> dict[str, Any]:
    """Real OpenAI (GPT-5.6 `luna`/`terra`) — Responses API shape, confirmed against the installed
    `openai` SDK's `resources/responses/responses.py` signatures (`providers/openai.py`'s
    docstring has the full read-out). `max_output_tokens` is the output-token cap (top-level, same
    name as this dataclass's field — no rename needed, unlike Chat Completions'
    `max_completion_tokens`). System prompt is a top-level `instructions` kwarg, not a message —
    like Anthropic, unlike Groq/Chat Completions. Structured output is `response_format=<pydantic
    model>` here too (same neutral-params convention as the other builders — `providers/openai.py`
    pops it back out and passes it on as `.parse(text_format=...)`). Reasoning is the nested
    `reasoning={"effort": "low"}` shape the SDK's `Reasoning` type declares, not Chat Completions'
    flat `reasoning_effort` string.
    """
    if neutral.for_embedding:
        return {}  # no embeddings endpoint via Responses API; embeddings stay on Google in dev (§5)
    params: dict[str, Any] = {
        "max_output_tokens": neutral.max_output_tokens,
        "temperature": neutral.temperature,
    }
    if neutral.system is not None:
        params["instructions"] = neutral.system
    if neutral.response_schema is not None:
        params["response_format"] = neutral.response_schema
    # Always low, same call as the Groq builder above and for the same reason: extract and compose
    # are both short, low-reasoning tasks, and reasoning tokens count against max_output_tokens —
    # a higher effort risks the response getting cut off mid-reasoning for no quality benefit here.
    params["reasoning"] = {"effort": "low"}
    return params


_BUILDERS: dict[str, Callable[[NeutralParams], dict[str, Any]]] = {
    "google": _build_google,
    "groq": _build_openai_compatible,
    "anthropic": _build_anthropic,
    "openai": _build_openai_responses,
}


def build_params(spec: ModelSpec, neutral: NeutralParams) -> dict[str, Any]:
    try:
        builder = _BUILDERS[spec.vendor]
    except KeyError:
        raise NotImplementedError(
            f"build_params for vendor {spec.vendor!r} is unverified — see M9"
        ) from None
    return builder(neutral)
