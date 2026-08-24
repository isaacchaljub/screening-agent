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
from screening_agent.llm.providers import google as google_provider
from screening_agent.llm.providers import openai as openai_provider

_PROVIDERS: dict[str, Provider] = {
    "google": google_provider,
    "openai": openai_provider,
    "anthropic": anthropic_provider,
}


ROLES_REQUIRING_STARTUP_CHECK = ("extract", "compose", "embed")


class LLMClient:
    def __init__(self, *, app_env: str | None = None) -> None:
        self.app_env = app_env or config.APP_ENV
        for role in ROLES_REQUIRING_STARTUP_CHECK:
            config.assert_model_allowed(registry.resolve(role, app_env=self.app_env).full_name)

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

    def embed(self, texts: list[str]) -> EmbedResult:
        primary, backup = self._resolve_with_backup("embed")

        def call(spec: registry.ModelSpec) -> EmbedResult:
            vendor_params = params_mod.build_params(spec, params_mod.NeutralParams())
            return self._provider_for(spec).embed(spec, texts=texts, params=vendor_params)

        return fallback.call_with_fallback(call, primary=primary, backup=backup)
