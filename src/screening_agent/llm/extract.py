"""The extract half of R2's two-call turn: strict schema → partial profile + detected language.

`extract()` never validates or decides anything — it only pulls out, verbatim where possible, what
the candidate's *latest* message says about each field. `validators.py` (pure Python, M1) is what
turns that raw text into an accepted value or a rejection reason; this module must stay ignorant
of §4.4's rules the same way `stages.py` is ignorant of them, just from the other direction: R1 is
about who decides *flow*, but R3 ("never guess a field") means extraction has to be conservative
independent of R1 — leaving a field null costs one message, a wrong field corrupts the handoff
silently.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from screening_agent.llm.base import Message
from screening_agent.llm.client import LLMClient
from screening_agent.models import Language

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
"""


class ExtractedFields(BaseModel):
    language: Language | None = None
    full_name: str | None = None
    has_license: str | None = Field(default=None, description="raw yes/no/hedge phrase")
    city: str | None = None
    availability: str | None = Field(
        default=None,
        description="Employment type only: full-time, part-time, or weekends. "
        "Not a time of day — that's preferred_schedule.",
    )
    preferred_schedule: str | None = Field(
        default=None,
        description="Time of day only: morning, afternoon, evening, or flexible. "
        "Not full-time/part-time/weekends — that's availability.",
    )
    experience_years: str | None = None
    experience_platforms: list[str] = Field(default_factory=list)
    start_date: str | None = None


def extract(
    client: LLMClient, *, history: list[Message], candidate_message: str
) -> ExtractedFields:
    messages = [*history, Message(role="user", content=candidate_message)]
    result = client.complete_structured(
        "extract",
        system=_SYSTEM_PROMPT,
        messages=messages,
        schema=ExtractedFields,
        temperature=0.0,  # a factual extraction, not creative writing — same input, same output
    )
    assert isinstance(result.data, ExtractedFields)  # schema= guarantees this; narrows for mypy
    return result.data
