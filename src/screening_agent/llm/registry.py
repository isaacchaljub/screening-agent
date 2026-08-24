"""`ModelSpec` and the extract/compose/embed role table (§5).

Model *roles* and *prices* are decisions, made in `PLAN_FOR_SONNET.md` §5 and revised in M9, and
are taken as given here. Model *API shapes* are verified separately, per provider, before that
provider is written — see each `providers/*.py` module's docstring for what was confirmed and how.

**Role table, as of M9 (2026-08-24, superseding §5's original OpenAI-primary table):** Anthropic
is the default (primary) for extract/compose — verified live, `ANTHROPIC_API_KEY` present, cost
accepted for dev/demo use. Google (free-tier, dev-only per R7) is the backup, covering a transport
failure in `dev`; outside `dev` a failed Anthropic call has no fallback, since R7 forbids falling
back onto a free-tier vendor there. Groq is deliberately *not* in this table at all — it's reached
only via `LLMClient(model="groq:...")` for eval sweeps (M8), never as a live primary or backup, to
keep its free-tier usage predictable. Real OpenAI (GPT-5.6 `sol`/`terra`/`luna`, confirmed live via
WebFetch on 2026-08-24 to be real, current model ids) now has a working adapter —
`providers/openai.py` implements the Responses API (`client.responses.create`/`.parse`), the
materially different shape from Chat Completions/Groq that blocked it at M9 — reachable via
`LLMClient(model="openai:...")`. It is still deliberately *not* in this role table: that adapter
is read out of the installed SDK's signatures, not live-verified against a real response (see its
own docstring), so swapping it in for Anthropic as primary/backup is a separate call.
`OPENAI_API_KEY` is present; revisit once §2's live-verification step has actually run.
"""

from __future__ import annotations

from dataclasses import dataclass

# Confirmed live against console.groq.com/docs/openai on 2026-08-24: Groq's endpoint is
# "mostly compatible with OpenAI's client libraries" at this base URL. Kept here, not hardcoded in
# providers/chat_completions.py (§5: "make the base URL part of ModelSpec, not a hardcoded
# constant") — the provider module only ever reads `spec.base_url`.
_VENDOR_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
}


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
        kwargs.setdefault("base_url", _VENDOR_BASE_URLS.get(vendor))
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

# M8 eval sweeps: confirmed live against console.groq.com/docs/models on 2026-08-24. gpt-oss-120b,
# not the smaller/cheaper 20b, because it's currently the only production Groq model class that
# supports strict JSON-schema structured output (console.groq.com/docs/structured-outputs) — the
# extract role needs that, so the choice isn't really discretionary. `--model groq:<id>` (LLMClient
# override) selects this rather than it living in ROLES/resolve() — a sweep runs one model against
# both roles, unlike the normal primary/backup-per-role table.
GROQ_EVAL_MODEL = ModelSpec.parse("groq:openai/gpt-oss-120b")

ROLES: dict[str, Role] = {
    "extract": Role(
        primary=ModelSpec.parse("anthropic:claude-haiku-4-5"),  # fast/cheap — a factual pull
        backup=GEMINI_DEV_MODEL,  # dev-only per R7; no fallback for a live failure outside dev
    ),
    "compose": Role(
        primary=ModelSpec.parse("anthropic:claude-sonnet-5"),  # stronger — candidate-facing tone
        backup=GEMINI_DEV_MODEL,
    ),
    "embed": Role(
        # Neither Anthropic nor Groq has an embeddings endpoint (§5/M8/M9) — Google is the only
        # implemented path, so it's the primary outright rather than a "dev override" of some
        # unbuilt paid primary. R7 still gates it to `dev` (it's free-tier); embed simply has no
        # non-dev story until a paid embeddings vendor is implemented.
        primary=GEMINI_EMBED_MODEL,
        backup=None,
    ),
}


def resolve(role: str, *, app_env: str) -> ModelSpec:
    """The primary model for `role`, unless `app_env == "dev"` and a dev override exists — then
    that. `--model vendor:id` (CLI/eval sweeps) bypasses this and builds a `ModelSpec` directly."""
    entry = ROLES[role]
    if app_env == "dev" and entry.dev_override is not None:
        return entry.dev_override
    return entry.primary
