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


@dataclass(frozen=True, slots=True)
class Redirect:
    """One neutral redirect after off-script/inappropriate input (guardrails.py, M5). Carries the
    stage whose question is still outstanding, so compose.py can re-ask it naturally instead of
    just scolding the candidate."""

    stage: Stage


Step = AskStage | Terminate | Confirm | Redirect


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

    # Rule 5 — everything is filled: qualify in this same step. This used to show WRAP_UP as
    # its own `AskStage`, with qualification only on the *next* `next_step()` call — but that
    # left the candidate needing to send one more message after already being told a recruiter
    # would follow up, which most candidates (reasonably) never do, so the conversation just
    # sat un-qualified. process-design.md's stage table treats "confirms what was captured,
    # states next steps" and "-> QUALIFIED" as one step, not two round-trips apart.
    return Terminate(Terminal.QUALIFIED)


def guardrail_step(prior_off_script_count: int, pending_stage: Stage) -> Redirect | Terminate:
    """Called instead of `next_step()` on a turn `guardrails.classify()` flagged — never folded
    into `next_step()` itself, because "off-script" is a property of *this turn's* message, not
    of profile/attempt state, and `next_step()` is re-evaluated on every later turn too (folding
    it in would re-trigger the redirect forever instead of only once). §3 process-design: one
    neutral redirect, then the conversation closes.
    """
    if prior_off_script_count >= 1:
        return Terminate(Terminal.ABANDONED, "off_script")
    return Redirect(pending_stage)
