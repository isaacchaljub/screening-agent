"""The extract half of R2's two-call turn: strict schema → partial profile + detected language.

`extract()` never validates or decides anything — it only pulls out, verbatim where possible, what
the candidate's *latest* message says about each field. `validators.py` (pure Python) is what
turns that raw text into an accepted value or a rejection reason; this module must stay ignorant
of §4.4's rules the same way `stages.py` is ignorant of them, just from the other direction: R1 is
about who decides *flow*, but R3 ("never guess a field") means extraction has to be conservative
independent of R1 — leaving a field null costs one message, a wrong field corrupts the handoff
silently.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from screening_agent.llm.base import Message, SchemaError
from screening_agent.llm.client import LLMClient
from screening_agent.models import Language

# R5 / llm/retry.py's own docstring: a schema-validation failure retries the *same* model with
# the parse error appended, never a vendor fallback — a different vendor won't fix a bad schema,
# it'll just bill you.
MAX_SCHEMA_RETRIES = 2

_SYSTEM_PROMPT = """\
You extract structured facts from ONE candidate message in a delivery-driver screening chat.

Rules:
- Only extract something explicitly and unambiguously stated in the CANDIDATE'S LATEST message —
  not something implied, guessed, or already established earlier in the conversation.
- If a field isn't clearly present in the latest message, leave it null. Never guess or infer.
- Keep text fields close to the candidate's own words and language — do not translate or
  normalize them (e.g. keep "un par de años", keep "Sevilla" or "Seville" as typed). A later,
  separate step handles normalization and validation.
- BUT strip conversational wrapping — extract the value alone, not the sentence around it.
  If someone writes "me llamo Carla Núñez" or "my name is Carla Núñez", full_name is "Carla
  Núñez" — drop "me llamo" / "my name is", keep only the name itself.
- A candidate message can answer several fields at once — extract all of them.
- Detect the language of the candidate's latest message: "es" or "en". If genuinely mixed, pick
  whichever dominates.
- If the candidate asked a question instead of (or alongside) answering — about pay, hours,
  vehicle, documents, equipment, insurance, anything about the job — put that question, standalone
  and in its own words if it relies on earlier context, in faq_question. Null if they asked
  nothing.
"""


class ExtractedFields(BaseModel):
    language: Language | None = None
    full_name: str | None = None
    has_license: str | None = Field(default=None, description="raw yes/no/hedge phrase")
    city: str | None = None
    availability: str | None = Field(
        default=None,
        description="Employment type only: full-time/part-time/weekends, e.g. 'tiempo completo', "
        "'medio tiempo', 'fines de semana'. Never put a time of day here (morning/afternoon/"
        "evening/'por la mañana') — that's preferred_schedule, a separate field.",
    )
    preferred_schedule: str | None = Field(
        default=None,
        description="Time of day only: morning/afternoon/evening/flexible, e.g. 'por la mañana', "
        "'por la tarde'. Never put full-time/part-time/weekends here (e.g. 'tiempo completo') — "
        "that's availability, a separate field.",
    )
    experience_years: str | None = Field(
        default=None,
        description="The raw phrase, even if it's not a number — 'ninguna'/'no experience'/'none' "
        "is a real, valid answer here, not a reason to leave this null.",
    )
    experience_platforms: list[str] = Field(
        default_factory=list,
        description="Always an array — use [] when none were mentioned, never null.",
    )
    start_date: str | None = None
    faq_question: str | None = Field(
        default=None,
        description="A question the candidate asked about the job (pay, hours, vehicle, "
        "documents, equipment, insurance, etc.), standalone/self-contained. Null if none.",
    )


def extract(
    client: LLMClient, *, history: list[Message], candidate_message: str
) -> ExtractedFields:
    messages = [*history, Message(role="user", content=candidate_message)]
    system = _SYSTEM_PROMPT
    last_error: SchemaError | None = None

    for _ in range(MAX_SCHEMA_RETRIES + 1):
        try:
            result = client.complete_structured(
                "extract",
                system=system,
                messages=messages,
                schema=ExtractedFields,
                temperature=0.0,  # factual extraction, not creative writing — same in, same out
            )
        except SchemaError as exc:
            last_error = exc
            system = (
                f"{_SYSTEM_PROMPT}\n\nYour previous response did not validate against the "
                f"schema: {exc}. Return valid JSON that matches the schema exactly this time."
            )
            continue
        assert isinstance(result.data, ExtractedFields)  # schema= guarantees this; narrows mypy
        return result.data

    assert last_error is not None  # loop always sets this before falling through
    raise last_error
