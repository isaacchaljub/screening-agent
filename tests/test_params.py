import copy

import pytest
from pydantic import BaseModel

from screening_agent.llm.params import NeutralParams, build_params
from screening_agent.llm.registry import ModelSpec

GOOGLE_SPEC = ModelSpec.parse("google:gemini-3.5-flash-lite")
GROQ_SPEC = ModelSpec.parse("groq:openai/gpt-oss-120b")
# Haiku 4.5 predates adaptive thinking; Sonnet 5 has it. The two specs exist separately here
# because that difference is *within* the vendor and changes the request shape (live-verified
# 400s both ways) — the exact thing `supports_adaptive_thinking` was added to carry.
ANTHROPIC_SPEC = ModelSpec.parse("anthropic:claude-haiku-4-5")
ANTHROPIC_THINKING_SPEC = ModelSpec.parse("anthropic:claude-sonnet-5")
OPENAI_SPEC = ModelSpec.parse("openai:gpt-5.6-luna")


class _DummySchema(BaseModel):
    foo: str


def test_google_system_prompt_and_token_cap():
    params = build_params(GOOGLE_SPEC, NeutralParams(system="be terse", max_output_tokens=200))
    assert params["system_instruction"] == "be terse"
    assert params["max_output_tokens"] == 200
    assert "response_schema" not in params


def test_google_structured_output_mechanism():
    params = build_params(GOOGLE_SPEC, NeutralParams(response_schema=_DummySchema))
    assert params["response_mime_type"] == "application/json"
    assert params["response_schema"] is _DummySchema


def test_google_thinking_only_set_when_requested():
    off = build_params(GOOGLE_SPEC, NeutralParams(thinking=False))
    on = build_params(GOOGLE_SPEC, NeutralParams(thinking=True))
    assert "thinking_config" not in off
    assert "thinking_config" in on


def test_build_params_never_mutates_neutral():
    neutral = NeutralParams(system="hello")
    before = copy.deepcopy(neutral)
    build_params(GOOGLE_SPEC, neutral)
    assert neutral == before


def test_build_params_returns_a_fresh_dict_each_call():
    neutral = NeutralParams(system="hello")
    first = build_params(GOOGLE_SPEC, neutral)
    second = build_params(GOOGLE_SPEC, neutral)
    assert first is not second
    first["max_output_tokens"] = -999
    assert second["max_output_tokens"] != -999


def test_google_embed_params_omit_generation_only_fields():
    # Regression: EmbedContentConfig rejects max_output_tokens/temperature outright (live-verified
    # ValidationError) — building embed params the same way as generation params breaks embed().
    params = build_params(GOOGLE_SPEC, NeutralParams(for_embedding=True))
    assert "max_output_tokens" not in params
    assert "temperature" not in params
    assert "system_instruction" not in params
    assert params == {}


def test_google_embed_params_include_task_type_when_given():
    params = build_params(
        GOOGLE_SPEC, NeutralParams(for_embedding=True, embed_task_type="RETRIEVAL_QUERY")
    )
    assert params == {"task_type": "RETRIEVAL_QUERY"}


def test_groq_uses_openai_compatible_shape():
    params = build_params(GROQ_SPEC, NeutralParams(system="be terse", max_output_tokens=200))
    assert params["system"] == "be terse"
    assert params["max_completion_tokens"] == 200
    assert "max_output_tokens" not in params  # OpenAI-shaped name, not Google's
    assert "response_format" not in params


def test_groq_structured_output_mechanism():
    params = build_params(GROQ_SPEC, NeutralParams(response_schema=_DummySchema))
    assert params["response_format"] is _DummySchema


def test_groq_always_uses_low_reasoning_effort():
    # Extract/compose are both short, low-reasoning tasks, and GPT-OSS's hidden reasoning tokens
    # count against max_completion_tokens — worth asserting this stays on, not opt-in.
    params = build_params(GROQ_SPEC, NeutralParams())
    assert params["reasoning_effort"] == "low"


def test_groq_embed_is_empty_no_embedding_endpoint():
    params = build_params(GROQ_SPEC, NeutralParams(for_embedding=True))
    assert params == {}


def test_anthropic_output_token_cap_and_system_prompt():
    params = build_params(ANTHROPIC_SPEC, NeutralParams(system="be terse", max_output_tokens=200))
    assert params["max_tokens"] == 200  # Anthropic's name, not Google's or OpenAI's
    assert params["system"] == "be terse"
    assert "temperature" not in params  # removed entirely on current models (live-verified)


def test_anthropic_structured_output_mechanism():
    params = build_params(ANTHROPIC_SPEC, NeutralParams(response_schema=_DummySchema))
    assert params["response_format"] is _DummySchema


def test_anthropic_pre_4_6_model_never_gets_thinking_or_effort():
    # Live-verified 2026-08-25: claude-haiku-4-5 400s on BOTH `thinking={"type": "adaptive"}`
    # ("adaptive thinking is not supported on this model") and `output_config.effort` ("This
    # model does not support the effort parameter"). Sending either is an outage, not a downgrade.
    for neutral in (NeutralParams(thinking=False), NeutralParams(thinking=True)):
        params = build_params(ANTHROPIC_SPEC, neutral)
        assert "thinking" not in params
        assert "output_config" not in params


def test_anthropic_4_6_plus_model_pins_effort_low_by_default():
    # Sonnet 5 runs adaptive thinking even when `thinking` is omitted, and thinking tokens count
    # against max_tokens — so the point of setting it explicitly is to get at the `effort` knob
    # that bounds it. Default low: extract is a factual pull, compose writes one short sentence.
    params = build_params(ANTHROPIC_THINKING_SPEC, NeutralParams())
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": "low"}


def test_anthropic_thinking_flag_raises_effort_rather_than_toggling_it_on():
    params = build_params(ANTHROPIC_THINKING_SPEC, NeutralParams(thinking=True))
    assert params["output_config"] == {"effort": "high"}


def test_unknown_anthropic_model_degrades_to_sending_neither_param():
    # The allowlist has to fail toward "omit", which every model accepts — the other direction is
    # a 400 the first time an unrecognised id is used.
    spec = ModelSpec.parse("anthropic:claude-something-unreleased")
    assert spec.supports_adaptive_thinking is False
    params = build_params(spec, NeutralParams())
    assert "thinking" not in params
    assert "output_config" not in params


def test_anthropic_embed_is_empty_no_embedding_endpoint():
    params = build_params(ANTHROPIC_SPEC, NeutralParams(for_embedding=True))
    assert params == {}


def test_openai_uses_responses_shape():
    params = build_params(OPENAI_SPEC, NeutralParams(system="be terse", max_output_tokens=200))
    assert params["instructions"] == "be terse"
    assert params["max_output_tokens"] == 200
    assert "max_completion_tokens" not in params  # Chat Completions' name, not this shape's
    assert "system" not in params  # Chat Completions' key, not the Responses API's
    # live-verified (M9): GPT-5.6 terra 400s on `temperature` — a reasoning model, sampled by
    # `reasoning.effort` instead, same story as `_build_anthropic`.
    assert "temperature" not in params


def test_openai_structured_output_mechanism():
    params = build_params(OPENAI_SPEC, NeutralParams(response_schema=_DummySchema))
    assert params["response_format"] is _DummySchema


def test_openai_always_uses_low_reasoning_effort():
    # Same reasoning as the Groq test above: extract/compose are short, low-reasoning tasks, and
    # reasoning tokens count against max_output_tokens — worth asserting this stays on, not opt-in.
    params = build_params(OPENAI_SPEC, NeutralParams())
    assert params["reasoning"] == {"effort": "low"}


def test_default_token_cap_leaves_room_for_hidden_reasoning_tokens():
    # Regression (M9 bake-off): at the previous 400 default, claude-sonnet-5's adaptive thinking
    # could spend the whole budget before emitting the JSON, failing two eval scenarios with
    # "Invalid JSON: EOF while parsing a value". The cap must cover reasoning, not just the answer.
    from screening_agent.llm.params import DEFAULT_MAX_OUTPUT_TOKENS

    assert DEFAULT_MAX_OUTPUT_TOKENS >= 1024
    for spec, key in (
        (GOOGLE_SPEC, "max_output_tokens"),
        (GROQ_SPEC, "max_completion_tokens"),
        (ANTHROPIC_SPEC, "max_tokens"),
        (OPENAI_SPEC, "max_output_tokens"),
    ):
        assert build_params(spec, NeutralParams())[key] == DEFAULT_MAX_OUTPUT_TOKENS


def test_openai_embed_is_empty_no_embedding_endpoint():
    params = build_params(OPENAI_SPEC, NeutralParams(for_embedding=True))
    assert params == {}


def test_unverified_vendor_raises_not_implemented():
    spec = ModelSpec(vendor="unknown", model_id="some-model")
    with pytest.raises(NotImplementedError):
        build_params(spec, NeutralParams())


def test_model_spec_full_name_roundtrip():
    spec = ModelSpec.parse("anthropic:claude-sonnet-5")
    assert spec.vendor == "anthropic"
    assert spec.model_id == "claude-sonnet-5"
    assert spec.full_name == "anthropic:claude-sonnet-5"


def test_model_spec_rejects_missing_vendor():
    with pytest.raises(ValueError):
        ModelSpec.parse("gemini-3.5-flash-lite")
