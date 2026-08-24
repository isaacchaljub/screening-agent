"""`build_params()` (§5): translates one vendor-neutral request into the exact kwargs a vendor
SDK expects, reconciling the output-token cap name, where the system prompt goes, the
reasoning/thinking parameter, the structured-output mechanism, and whether sampling parameters
are top-level or nested. `NeutralParams` is frozen and every builder returns a fresh `dict` — R6:
`ModelSpec` is frozen, `build_params()` never mutates a caller's dict, popping keys out of a
shared one works once per process and then quietly stops working.

Only `google` is implemented — confirmed against `google-genai==1.60.0` (installed; see
`providers/google.py`). OpenAI and Anthropic branches wait for M9's verification pass (§2) rather
than guess a shape that would just be rewritten.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from screening_agent.llm.registry import ModelSpec

DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.3


@dataclass(frozen=True, slots=True)
class NeutralParams:
    system: str | None = None
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    response_schema: type[BaseModel] | None = None
    thinking: bool = False


def _build_google(neutral: NeutralParams) -> dict[str, Any]:
    params: dict[str, Any] = {
        "max_output_tokens": neutral.max_output_tokens,
        "temperature": neutral.temperature,
    }
    if neutral.system is not None:
        params["system_instruction"] = neutral.system
    if neutral.response_schema is not None:
        params["response_mime_type"] = "application/json"
        params["response_schema"] = neutral.response_schema
    if neutral.thinking:
        params["thinking_config"] = {"thinking_level": "low"}
    return params


_BUILDERS: dict[str, Callable[[NeutralParams], dict[str, Any]]] = {"google": _build_google}


def build_params(spec: ModelSpec, neutral: NeutralParams) -> dict[str, Any]:
    try:
        builder = _BUILDERS[spec.vendor]
    except KeyError:
        raise NotImplementedError(
            f"build_params for vendor {spec.vendor!r} is unverified — see M9"
        ) from None
    return builder(neutral)
