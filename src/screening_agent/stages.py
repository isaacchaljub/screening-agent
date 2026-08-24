"""The stage machine (§4.5). Pure, no I/O, no vendor or network code (R1) — this and
`validators.py` are the only two modules that decide what happens next; the model reads and
writes text, never flow.

`attempts` is a small `dict[str, int]` of orchestration state kept by the caller (`engine.py`,
built in M3) across turns. Most keys are failed-attempt counters, one per `CandidateProfile`
field name. Two keys are booleans encoded as 0/1 rather than counts:

- `"has_license:needs_confirmation"` — set once a hedge on the licence question is detected
  (`validators.validate_has_license` returned `needs_confirmation=True`).
- `"has_license:confirmed"` — set once the candidate has answered the follow-up `Confirm` turn.
- `"has_license:confirm_attempts"` — counts unclear replies to that `Confirm` turn itself; capped
  the same way any other field is (rule 3), so an unresolved hedge reaches a human instead of
  looping on `Confirm` forever.
- `"wrap_up:shown"` — set once the WRAP_UP message has actually been sent.

This dict, not `CandidateProfile`, is where that bookkeeping lives — the profile is also the
handoff payload and stays exactly the ten fields in §4.2.
"""

from __future__ import annotations

from dataclasses import dataclass

from screening_agent.models import CandidateProfile, DisqualifyReason, Stage, Terminal

MAX_ATTEMPTS = 2

# (profile field, stage that asks for it), in §4.1 order. `zone_id`, not `city_raw`, is the
# CITY field's "answered" signal — `city_raw` can be set while still unresolved.
FIELD_ORDER: tuple[tuple[str, Stage], ...] = (
    ("full_name", Stage.NAME),
    ("has_license", Stage.LICENSE),
    ("zone_id", Stage.CITY),
    ("availability", Stage.AVAILABILITY),
    ("preferred_schedule", Stage.SCHEDULE),
    ("experience_years", Stage.EXPERIENCE),
    ("start_date", Stage.START_DATE),
)


@dataclass(frozen=True, slots=True)
class AskStage:
    stage: Stage


@dataclass(frozen=True, slots=True)
class Terminate:
    outcome: Terminal
    reason: DisqualifyReason | str | None = None


@dataclass(frozen=True, slots=True)
class Confirm:
    field: str


Step = AskStage | Terminate | Confirm


FIELD_FOR_STAGE: dict[Stage, str] = {stage: field for field, stage in FIELD_ORDER}


def is_field_empty(profile: CandidateProfile, field: str) -> bool:
    value = getattr(profile, field)
    if isinstance(value, str):
        return not value.strip()
    return value is None


def next_step(profile: CandidateProfile, attempts: dict[str, int]) -> Step:
    # Rule 1 — licence. An explicit "no" is disqualified outright; a hedge ("me lo saco en
    # junio") is confirmed once first, per validators.validate_has_license's needs_confirmation.
    if profile.has_license is False:
        awaiting_confirmation = attempts.get(
            "has_license:needs_confirmation", 0
        ) and not attempts.get("has_license:confirmed", 0)
        if awaiting_confirmation:
            if attempts.get("has_license:confirm_attempts", 0) >= MAX_ATTEMPTS:
                return Terminate(Terminal.NEEDS_HUMAN, "has_license")
            return Confirm("has_license")
        return Terminate(Terminal.DISQUALIFIED, DisqualifyReason.NO_LICENSE)

    # Rule 2 — city given but never resolved to a service zone after two attempts.
    if profile.city_raw and not profile.zone_id and attempts.get("zone_id", 0) >= MAX_ATTEMPTS:
        return Terminate(Terminal.DISQUALIFIED, DisqualifyReason.OUTSIDE_SERVICE_AREA)

    # Rule 3 — any field stuck at the attempt cap goes to a human rather than looping.
    for field, _stage in FIELD_ORDER:
        if attempts.get(field, 0) >= MAX_ATTEMPTS and is_field_empty(profile, field):
            return Terminate(Terminal.NEEDS_HUMAN, field)

    # Rule 4 — ask the first still-empty stage. Fields volunteered early are already filled,
    # so this naturally skips ahead.
    for field, stage in FIELD_ORDER:
        if is_field_empty(profile, field):
            return AskStage(stage)

    # Rule 5 — everything is filled: show WRAP_UP once, then qualify.
    if not attempts.get("wrap_up:shown", 0):
        return AskStage(Stage.WRAP_UP)
    return Terminate(Terminal.QUALIFIED)
