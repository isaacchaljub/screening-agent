"""`LLMClient` — the only thing the rest of the app imports from `llm/` (§5).

Resolves a role to a `ModelSpec` (`registry`), builds vendor params (`params`), dispatches to the
right `providers/*` module, and wraps the call in retry + fallback. Validates every configured
role's model against R7 at construction time — "fail at startup, not at 2am mid-conversation."
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from screening_agent import config
from screening_agent.llm import fallback, registry
from screening_agent.llm import params as params_mod
from screening_agent.llm.base import EmbedResult, Message, Provider, StructuredResult, TextResult
from screening_agent.llm.providers import anthropic as anthropic_provider
from screening_agent.llm.providers import chat_completions as chat_completions_provider
from screening_agent.llm.providers import google as google_provider
from screening_agent.llm.providers import local as local_provider
from screening_agent.llm.providers import openai as openai_provider

_PROVIDERS: dict[str, Provider] = {
    "google": google_provider,
    "openai": openai_provider,  # Responses API (providers/openai.py)
    "anthropic": anthropic_provider,
    "groq": chat_completions_provider,  # OpenAI-compatible Chat Completions, different base_url/key
    "local": local_provider,  # in-process sentence-transformers; embeddings only, no network
}


ROLES_REQUIRING_STARTUP_CHECK = ("extract", "compose", "embed")


class LLMClient:
    def __init__(self, *, app_env: str | None = None, model: str | None = None) -> None:
        """`model` ("vendor:model-id", e.g. "groq:openai/gpt-oss-120b") forces extract/compose
        onto one model, bypassing the normal per-role primary/backup/dev-override table entirely
        — an eval sweep is testing one model end to end, not exercising the fallback ladder,
        so there's no backup for a forced model either (a transport failure just fails the sweep
        run, which is the right signal for "rerun it," not a silent vendor swap mid-sweep).
        `embed` is deliberately never forced — see `_resolve_with_backup`."""
        self.app_env = app_env or config.APP_ENV
        self._forced_spec = registry.ModelSpec.parse(model) if model is not None else None
        for role in ROLES_REQUIRING_STARTUP_CHECK:
            if self._forced_spec is not None and role != "embed":
                spec = self._forced_spec
            else:
                spec = registry.resolve(role, app_env=self.app_env)
            config.assert_model_allowed(spec.full_name)

    def _provider_for(self, spec: registry.ModelSpec) -> Provider:
        try:
            return _PROVIDERS[spec.vendor]
        except KeyError:
            raise NotImplementedError(
                f"no provider registered for vendor {spec.vendor!r}"
            ) from None

    def _resolve_with_backup(
        self, role: str
    ) -> tuple[registry.ModelSpec, registry.ModelSpec | None]:
        # A forced `model` (eval sweeps) overrides extract/compose — the two roles a sweep is
        # actually testing — but never embed: §5 dedicates embeddings to Google regardless of
        # which model is under test (Groq, an eval-sweep vendor, has no embedding endpoint at
        # all — forcing it would break the FAQ-interruption scenario, since RAG retrieval embeds
        # through this same client).
        if self._forced_spec is not None and role != "embed":
            return self._forced_spec, None
        entry = registry.ROLES[role]
        primary = registry.resolve(role, app_env=self.app_env)
        # The dev override has no cross-vendor backup of its own (§5) — only fall back when
        # we're actually running the primary/backup pair.
        backup = entry.backup if primary is entry.primary else None
        return primary, backup

    def complete_text(
        self, role: str, *, messages: list[Message], system: str | None = None, **overrides: Any
    ) -> TextResult:
        primary, backup = self._resolve_with_backup(role)

        def call(spec: registry.ModelSpec) -> TextResult:
            neutral = params_mod.NeutralParams(system=system, **overrides)
            vendor_params = params_mod.build_params(spec, neutral)
            return self._provider_for(spec).complete_text(
                spec, messages=messages, params=vendor_params
            )

        return fallback.call_with_fallback(call, primary=primary, backup=backup)

    def complete_structured(
        self,
        role: str,
        *,
        messages: list[Message],
        schema: type[BaseModel],
        system: str | None = None,
        **overrides: Any,
    ) -> StructuredResult:
        primary, backup = self._resolve_with_backup(role)

        def call(spec: registry.ModelSpec) -> StructuredResult:
            neutral = params_mod.NeutralParams(system=system, response_schema=schema, **overrides)
            vendor_params = params_mod.build_params(spec, neutral)
            return self._provider_for(spec).complete_structured(
                spec, messages=messages, schema=schema, params=vendor_params
            )

        return fallback.call_with_fallback(call, primary=primary, backup=backup)

    def embed(self, texts: list[str], *, task_type: str | None = None) -> EmbedResult:
        primary, backup = self._resolve_with_backup("embed")

        def call(spec: registry.ModelSpec) -> EmbedResult:
            neutral = params_mod.NeutralParams(for_embedding=True, embed_task_type=task_type)
            vendor_params = params_mod.build_params(spec, neutral)
            return self._provider_for(spec).embed(spec, texts=texts, params=vendor_params)

        return fallback.call_with_fallback(call, primary=primary, backup=backup)
