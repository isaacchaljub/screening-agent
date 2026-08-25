"""`build_params()` (§5): translates one vendor-neutral request into the exact kwargs a vendor
SDK expects, reconciling the output-token cap name, where the system prompt goes, the
reasoning/thinking parameter, the structured-output mechanism, and whether sampling parameters
are top-level or nested. `NeutralParams` is frozen and every builder returns a fresh `dict` — R6:
`ModelSpec` is frozen, `build_params()` never mutates a caller's dict, popping keys out of a
shared one works once per process and then quietly stops working.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from screening_agent.llm.registry import ModelSpec

# The cap has to cover the model's *hidden reasoning* tokens, not just the visible answer — an
# adaptive-thinking model can spend the whole budget mid-thought and return truncated JSON or no
# text at all. 2048 is a ceiling, not a reservation: every vendor here bills and rate-limits on
# tokens actually generated, so raising it costs nothing on a call that doesn't need it. The cap
# exists to stop a runaway generation, not to fit the expected answer.
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.3


@dataclass(frozen=True, slots=True)
class NeutralParams:
    system: str | None = None
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    response_schema: type[BaseModel] | None = None
    thinking: bool = False
    # Embedding calls (`LLMClient.embed`) go through a *different* vendor config shape than
    # generation calls do: Google's `EmbedContentConfig` rejects `max_output_tokens`/`temperature`
    # outright. `task_type` ("RETRIEVAL_DOCUMENT" at index time, "RETRIEVAL_QUERY" at query time)
    # asymmetrically tunes Gemini's embedding for retrieval quality; unset for a symmetric
    # embedding.
    for_embedding: bool = False
    embed_task_type: str | None = None


def _build_google(spec: ModelSpec, neutral: NeutralParams) -> dict[str, Any]:
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


def _build_openai_compatible(spec: ModelSpec, neutral: NeutralParams) -> dict[str, Any]:
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


def _build_anthropic(spec: ModelSpec, neutral: NeutralParams) -> dict[str, Any]:
    """`max_tokens` is the output-token cap (top-level, required); `system` is a top-level kwarg,
    not a message; structured output is `client.messages.parse(..., output_format=<pydantic
    model>)` → `response.parsed_output` (`providers/anthropic.py` pops `response_format` back out
    and uses it there, same pattern as the OpenAI-compatible builder). Sampling params
    (`temperature`/`top_p`/`top_k`) are deliberately absent here — they don't exist as parameters
    on current models; passing one raises before any request is sent.

    **Reasoning is the one place the vendor alone isn't enough to build a request** — which is why
    this is the only builder that reads `spec` and not just `neutral`: `claude-sonnet-5` accepts
    `thinking={"type": "adaptive"}` and `output_config={"effort": ...}`, but `claude-haiku-4-5`
    400s on both (it predates the generation where adaptive thinking replaced `budget_tokens`).
    `registry.ModelSpec.supports_adaptive_thinking` carries that distinction, as an allowlist, so an
    unknown model id degrades to "send neither param" rather than to a 400.

    Effort is pinned to `"low"` on models that have it, for the same reason the Groq and OpenAI
    builders pin theirs: extract is a factual pull and compose writes one sentence under 25 words,
    so deeper reasoning buys nothing and costs latency the candidate feels in a messaging UI.
    `neutral.thinking=True` asks for the default depth instead, for a caller that wants it.
    """
    if neutral.for_embedding:
        return {}  # Anthropic has no embeddings endpoint; embeddings stay on Google in dev (§5)
    params: dict[str, Any] = {"max_tokens": neutral.max_output_tokens}
    if neutral.system is not None:
        params["system"] = neutral.system
    if neutral.response_schema is not None:
        params["response_format"] = neutral.response_schema
    if spec.supports_adaptive_thinking:
        # Explicit rather than omitted: on a 4.6+ model, omitting `thinking` does NOT mean "off"
        # — Sonnet 5 runs adaptive anyway. Saying so out loud is what lets the `effort` knob below
        # be the thing that actually bounds it. (`providers/anthropic.py` merges `output_config`
        # with the schema-derived `format`.)
        params["thinking"] = {"type": "adaptive"}
        params["output_config"] = {"effort": "high" if neutral.thinking else "low"}
    return params


def _build_openai_responses(spec: ModelSpec, neutral: NeutralParams) -> dict[str, Any]:
    """Real OpenAI (GPT-5.6 `luna`/`terra`) — Responses API shape (`providers/openai.py`'s
    docstring has the full read-out). `max_output_tokens` is the output-token cap (top-level, same
    name as this dataclass's field — no rename needed, unlike Chat Completions'
    `max_completion_tokens`). System prompt is a top-level `instructions` kwarg, not a message —
    like Anthropic, unlike Groq/Chat Completions. Structured output is `response_format=<pydantic
    model>` here too (same neutral-params convention as the other builders — `providers/openai.py`
    pops it back out and passes it on as `.parse(text_format=...)`). Reasoning is the nested
    `reasoning={"effort": "low"}` shape the SDK's `Reasoning` type declares, not Chat Completions'
    flat `reasoning_effort` string.

    No `temperature` — this model 400s on it the same way current-generation Anthropic models
    reject sampling params entirely (`_build_anthropic` above): a reasoning model, sampled by its
    `reasoning.effort` instead. The SDK's typed signature still accepts the kwarg; it's the live
    endpoint that rejects it.
    """
    if neutral.for_embedding:
        return {}  # no embeddings endpoint via Responses API; embeddings stay on Google in dev (§5)
    params: dict[str, Any] = {"max_output_tokens": neutral.max_output_tokens}
    if neutral.system is not None:
        params["instructions"] = neutral.system
    if neutral.response_schema is not None:
        params["response_format"] = neutral.response_schema
    # Always low, same call as the Groq builder above and for the same reason: extract and compose
    # are both short, low-reasoning tasks, and reasoning tokens count against max_output_tokens —
    # a higher effort risks the response getting cut off mid-reasoning for no quality benefit here.
    params["reasoning"] = {"effort": "low"}
    return params


def _build_local(spec: ModelSpec, neutral: NeutralParams) -> dict[str, Any]:
    """`providers/local.py` — in-process sentence-transformers, embeddings only. There is no
    request to shape here (no HTTP, no vendor SDK), so the only thing worth carrying through is the
    neutral `task_type`, which the provider translates into E5's `query:`/`passage:` prefixes. Kept
    as a builder rather than special-cased in `client.py` so the local path goes through exactly the
    same resolve -> build -> dispatch pipeline as every hosted vendor."""
    if not neutral.for_embedding:
        raise NotImplementedError(
            "the local provider is embeddings-only — generation stays on a hosted vendor"
        )
    return {"task_type": neutral.embed_task_type} if neutral.embed_task_type else {}


_BUILDERS: dict[str, Callable[[ModelSpec, NeutralParams], dict[str, Any]]] = {
    "google": _build_google,
    "groq": _build_openai_compatible,
    "anthropic": _build_anthropic,
    "openai": _build_openai_responses,
    "local": _build_local,
}


def build_params(spec: ModelSpec, neutral: NeutralParams) -> dict[str, Any]:
    try:
        builder = _BUILDERS[spec.vendor]
    except KeyError:
        raise NotImplementedError(
            f"no params builder registered for vendor {spec.vendor!r}"
        ) from None
    return builder(spec, neutral)
