"""`ModelSpec` and the extract/compose/embed role table (§5).

Model *roles* and *prices* are decisions, taken as given here. Model *API shapes* are verified
separately, per provider, before that provider is written — see each `providers/*.py` module's
docstring for what was confirmed and how.

**Role table:** Anthropic is the primary for extract/compose. Google (free-tier, dev-only per R7)
is the backup, covering a transport failure in `dev`; outside `dev` a failed Anthropic call has no
fallback, since R7 forbids falling back onto a free-tier vendor there. Groq and real OpenAI
(`providers/chat_completions.py`, `providers/openai.py`) both have working adapters but are
deliberately absent from this role table — reachable only via `LLMClient(model="vendor:...")`,
Groq for eval sweeps and OpenAI because the account behind `OPENAI_API_KEY` has no billing credits
(see README "Model choice" and `providers/openai.py`'s docstring). Neither is a live primary or
backup.
"""

from __future__ import annotations

from dataclasses import dataclass

# Groq's endpoint is "mostly compatible with OpenAI's client libraries" at this base URL. Kept
# here, not hardcoded in providers/chat_completions.py (§5: "make the base URL part of ModelSpec,
# not a hardcoded constant") — the provider module only ever reads `spec.base_url`.
_VENDOR_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
}


# Reasoning controls differ *within* the Anthropic vendor, by model generation — so they belong
# on `ModelSpec`, next to `supports_strict_schema`, rather than in `params.py`'s per-vendor
# builder (which only sees the vendor):
#
#   claude-sonnet-5  → accepts `thinking={"type": "adaptive"}` and `output_config={"effort": ...}`
#   claude-haiku-4-5 → 400s on both (predates adaptive thinking)
#
# The split is generational (Claude 4.6+ has adaptive thinking; earlier models used the now-removed
# `budget_tokens`), so this is an *allowlist*: an unrecognised model id gets `False` and we simply
# omit both params, which every model accepts. Guessing the other way would 400 at runtime.
_ADAPTIVE_THINKING_MODELS: frozenset[str] = frozenset(
    {
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-fable-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
    }
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    vendor: str
    model_id: str
    supports_strict_schema: bool = True
    base_url: str | None = None  # non-default endpoint, e.g. Groq's OpenAI-compatible one
    # Anthropic 4.6+ only — see `_ADAPTIVE_THINKING_MODELS` above and `params._build_anthropic`.
    supports_adaptive_thinking: bool = False

    @property
    def full_name(self) -> str:
        return f"{self.vendor}:{self.model_id}"

    @classmethod
    def parse(cls, model: str, **kwargs: object) -> ModelSpec:
        vendor, sep, model_id = model.partition(":")
        if not sep or not vendor or not model_id:
            raise ValueError(f"model spec must be 'vendor:model-id', got {model!r}")
        kwargs.setdefault("base_url", _VENDOR_BASE_URLS.get(vendor))
        # Derived here rather than passed in at each `ROLES` call site on purpose: an eval sweep
        # builds its spec straight from a `--model vendor:id` string (`LLMClient(model=...)`), so
        # a capability set only in `ROLES` would silently not apply to the very runs that measure
        # these models. Still overridable by an explicit kwarg.
        kwargs.setdefault(
            "supports_adaptive_thinking",
            vendor == "anthropic" and model_id in _ADAPTIVE_THINKING_MODELS,
        )
        return cls(vendor=vendor, model_id=model_id, **kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Role:
    primary: ModelSpec
    backup: ModelSpec | None = None
    dev_override: ModelSpec | None = None


# Both extract and compose share one dev-override model per §5.
GEMINI_DEV_MODEL = ModelSpec.parse("google:gemini-3.5-flash-lite")
GEMINI_EMBED_MODEL = ModelSpec.parse("google:gemini-embedding-001")

# Embeddings run in-process (`providers/local.py`), not against a vendor. Calibrated against this
# exact FAQ — see that module's docstring for the model comparison and `rag/retrieve.py` for the
# resulting relevance floor. This is the one role with no hosted dependency, which is deliberate:
# it was previously the only role with no story outside `dev`, because the sole implemented path
# was Google's free tier and R7 refuses free-tier vendors for candidate data.
LOCAL_EMBED_MODEL = ModelSpec.parse("local:intfloat/multilingual-e5-small")

# For eval sweeps. gpt-oss-120b, not the smaller/cheaper 20b, because it's currently the only
# production Groq model class that supports strict JSON-schema structured output — the extract
# role needs that, so the choice isn't really discretionary. `--model groq:<id>` (LLMClient
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
        # In-process, no vendor, no key, no quota — and legal in `prod`, which the hosted
        # alternative was not. Neither Anthropic nor Groq has an embeddings endpoint at all, and
        # Google's is free-tier, so R7 (free tiers may train on submitted prompts) refused it for
        # candidate data outside `dev`. Running the model locally removes the vendor rather than
        # arguing about its terms. Affordable precisely because embeddings are NOT on the per-turn
        # hot path: once per chunk at index time, and once per candidate *question* at query time
        # — not once per turn, since `extract.py` only yields a `faq_question` when one was asked.
        primary=LOCAL_EMBED_MODEL,
        # ⚠️ `backup=None` is a correctness requirement here, not an omission — and it is the one
        # place in this package where R5's "fall back on transport failure" must NOT apply.
        # An index is built with one specific embedding model, so its vectors only mean anything
        # against *that* model's vector space. Falling back to another would either raise on a
        # dimension mismatch (384 here vs Gemini's 3072) or, far worse with a matching dimension,
        # silently return neighbours computed across two unrelated spaces — retrieval that looks
        # like it worked and is noise. A failed embed must fail; `llm/retry.py` still retries it.
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
