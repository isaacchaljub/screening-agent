"""Vendor-neutral request/response types, and the `Protocol` every `providers/*.py` module
implements as a set of module-level functions (`complete_text`, `complete_structured`, `embed`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from screening_agent.llm.registry import ModelSpec


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class TextResult:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class StructuredResult:
    data: BaseModel
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class EmbedResult:
    vectors: list[list[float]]
    model: str


class SchemaError(RuntimeError):
    """A structured-output call returned something that didn't validate. Per R5 this is retried
    against the *same* model with the parse error appended — it must never trigger a fallback."""


class Provider(Protocol):
    def complete_text(
        self, spec: ModelSpec, *, messages: list[Message], params: dict[str, Any]
    ) -> TextResult: ...

    def complete_structured(
        self,
        spec: ModelSpec,
        *,
        messages: list[Message],
        schema: type[BaseModel],
        params: dict[str, Any],
    ) -> StructuredResult: ...

    def embed(
        self, spec: ModelSpec, *, texts: list[str], params: dict[str, Any]
    ) -> EmbedResult: ...
