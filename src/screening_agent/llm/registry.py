"""`ModelSpec` and the extract/compose/embed role table (§5).

Model *roles* and *prices* are decisions, made in `PLAN_FOR_SONNET.md` §5, and are taken as given
here. Model *API shapes* are verified separately, per provider, before that provider is written —
see `providers/google.py`'s docstring for what was confirmed and how. The Anthropic id below is
provisional until M9 pins it via the `claude-api` skill; the OpenAI embedding id is provisional
until M9 as well (M6's RAG build uses the Google embedding model in dev and never needs it).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    vendor: str
    model_id: str
    supports_strict_schema: bool = True
    base_url: str | None = None  # non-default endpoint, e.g. Groq's OpenAI-compatible one

    @property
    def full_name(self) -> str:
        return f"{self.vendor}:{self.model_id}"

    @classmethod
    def parse(cls, model: str, **kwargs: object) -> ModelSpec:
        vendor, sep, model_id = model.partition(":")
        if not sep or not vendor or not model_id:
            raise ValueError(f"model spec must be 'vendor:model-id', got {model!r}")
        return cls(vendor=vendor, model_id=model_id, **kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Role:
    primary: ModelSpec
    backup: ModelSpec | None = None
    dev_override: ModelSpec | None = None


# Confirmed live against the free tier via ai.google.dev/gemini-api/docs/{models,pricing} on
# 2026-08-24 — both extract and compose share one dev-override model per §5.
GEMINI_DEV_MODEL = ModelSpec.parse("google:gemini-3.5-flash-lite")
GEMINI_EMBED_MODEL = ModelSpec.parse("google:gemini-embedding-001")

ROLES: dict[str, Role] = {
    "extract": Role(
        primary=ModelSpec.parse("openai:gpt-5.6-luna"),
        backup=ModelSpec.parse("anthropic:claude-haiku-4-5"),
        dev_override=GEMINI_DEV_MODEL,
    ),
    "compose": Role(
        primary=ModelSpec.parse("openai:gpt-5.6-terra"),
        backup=ModelSpec.parse("anthropic:claude-sonnet-5"),
        dev_override=GEMINI_DEV_MODEL,
    ),
    "embed": Role(
        primary=ModelSpec.parse("openai:text-embedding-3-large"),  # provisional, verify at M9
        backup=None,
        dev_override=GEMINI_EMBED_MODEL,
    ),
}


def resolve(role: str, *, app_env: str) -> ModelSpec:
    """The primary model for `role`, unless `app_env == "dev"` and a dev override exists — then
    that. `--model vendor:id` (CLI/eval sweeps) bypasses this and builds a `ModelSpec` directly."""
    entry = ROLES[role]
    if app_env == "dev" and entry.dev_override is not None:
        return entry.dev_override
    return entry.primary
