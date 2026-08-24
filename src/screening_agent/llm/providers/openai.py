"""OpenAI provider — built in M9. The GPT-5.6 family's current API shape postdates this session's
reliable knowledge (§2); read the vendor's current docs with WebFetch/WebSearch before implementing
this, the way `providers/google.py`'s docstring was built from a live, verified read.

Also serves Groq eval sweeps (`ModelSpec.base_url` pointed at Groq's OpenAI-compatible endpoint) —
verify Groq's current model ids from their docs when M8 wires that up.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from screening_agent.llm.base import EmbedResult, Message, StructuredResult, TextResult
from screening_agent.llm.registry import ModelSpec

_NOT_BUILT_YET = "providers/openai.py is not implemented — see M9 in _internal/PLAN_FOR_SONNET.md"


def complete_text(
    spec: ModelSpec, *, messages: list[Message], params: dict[str, Any]
) -> TextResult:
    raise NotImplementedError(_NOT_BUILT_YET)


def complete_structured(
    spec: ModelSpec, *, messages: list[Message], schema: type[BaseModel], params: dict[str, Any]
) -> StructuredResult:
    raise NotImplementedError(_NOT_BUILT_YET)


def embed(spec: ModelSpec, *, texts: list[str], params: dict[str, Any]) -> EmbedResult:
    raise NotImplementedError(_NOT_BUILT_YET)
