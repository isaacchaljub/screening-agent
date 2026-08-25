"""Runtime configuration: environment, service zones, tone constants.

Loaded once at import time. Real process environment variables always win over `.env` —
`load_dotenv()` does not override variables already set, which is what lets
`APP_ENV=prod python -c ...` override a `.env` that says `APP_ENV=dev` (see the R7 guard test
in `_internal/PLAN_FOR_SONNET.md` M0).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

VALID_ENVS = ("dev", "demo", "prod")

APP_ENV: str = os.getenv("APP_ENV", "dev")
if APP_ENV not in VALID_ENVS:
    raise ValueError(f"APP_ENV must be one of {VALID_ENVS}, got {APP_ENV!r}")

# --- vendor keys — read here only; nothing else in the app should call os.getenv for these ---
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or None
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY") or None
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY") or None
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY") or None


# --- service zones (§4.3) ---


@dataclass(frozen=True, slots=True)
class Zone:
    id: str
    display_name: str
    country: str  # "ES" | "MX"
    timezone: str
    aliases: tuple[str, ...] = field(default_factory=tuple)


ZONES: tuple[Zone, ...] = (
    Zone("madrid", "Madrid", "ES", "Europe/Madrid", ("MAD",)),
    Zone("barcelona", "Barcelona", "ES", "Europe/Madrid", ("Barna", "BCN")),
    Zone("valencia", "Valencia", "ES", "Europe/Madrid", ()),
    Zone("sevilla", "Sevilla", "ES", "Europe/Madrid", ("Seville",)),
    Zone("malaga", "Málaga", "ES", "Europe/Madrid", ("Malaga",)),
    Zone(
        "cdmx",
        "Ciudad de México",
        "MX",
        "America/Mexico_City",
        ("CDMX", "Mexico City", "DF", "Ciudad de Mexico"),
    ),
    Zone("guadalajara", "Guadalajara", "MX", "America/Mexico_City", ("Gdl",)),
    Zone("monterrey", "Monterrey", "MX", "America/Mexico_City", ("MTY",)),
    Zone("puebla", "Puebla", "MX", "America/Mexico_City", ()),
    Zone("queretaro", "Querétaro", "MX", "America/Mexico_City", ("Queretaro",)),
)


# --- tone configuration (§4.6) — the compose prompt and the test suite both read this ---


@dataclass(frozen=True, slots=True)
class ToneConfig:
    max_words: int = 25
    one_question_per_message: bool = True
    no_greeting_after_first: bool = True
    no_bullet_lists: bool = True
    spanish_register: str = "tú"
    acknowledge_before_asking: bool = True


TONE = ToneConfig()


# --- R7: free-tier models are dev-only ---

# Note what is deliberately absent: "local" (llm/providers/local.py). R7 exists because free
# tiers permit the vendor to train on submitted prompts — it is a data-egress rule, not a cost
# rule. A model running in-process sends nothing anywhere, so it is the one embedding path that
# is allowed outside `dev`, and adding it here would ban the option that actually solves R7.
FREE_TIER_VENDORS: frozenset[str] = frozenset({"google", "groq"})


class FreeTierModelError(RuntimeError):
    """Raised when a free-tier model is selected outside APP_ENV=dev."""


def assert_model_allowed(model: str) -> None:
    """Raise `FreeTierModelError` if `model` ("vendor:model-id") is free-tier and
    APP_ENV is not "dev". Google's and Groq's free tiers may use prompts to improve
    their products, so this is a privacy control, not just a cost one."""
    vendor = model.split(":", 1)[0]
    if vendor in FREE_TIER_VENDORS and APP_ENV != "dev":
        raise FreeTierModelError(
            f"{model!r} is a free-tier model and cannot be used outside APP_ENV=dev "
            f"(current APP_ENV={APP_ENV!r}). Free-tier prompts may be used by the vendor "
            "to improve their products, which is not acceptable for demo/prod candidate data."
        )
