"""Anthropic provider — built in M9. Invoke the `claude-api` skill before writing this one; do not
write Anthropic SDK code from memory, the way `providers/google.py`'s docstring was built from a
live, verified read of `google-genai` instead of recalled knowledge.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from screening_agent.llm.base import EmbedResult, Message, StructuredResult, TextResult
from screening_agent.llm.registry import ModelSpec

_NOT_BUILT_YET = (
    "providers/anthropic.py is not implemented — see M9 in _internal/PLAN_FOR_SONNET.md"
)


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
