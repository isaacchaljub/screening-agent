import copy

import pytest
from pydantic import BaseModel

from screening_agent.llm.params import NeutralParams, build_params
from screening_agent.llm.registry import ModelSpec

GOOGLE_SPEC = ModelSpec.parse("google:gemini-3.5-flash-lite")


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


def test_unverified_vendor_raises_not_implemented():
    spec = ModelSpec.parse("openai:gpt-5.6-luna")
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
