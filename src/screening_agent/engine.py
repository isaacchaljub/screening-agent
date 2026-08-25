"""The turn loop (R2): extract → validate/advance in plain Python → compose. `stages.py` and
`validators.py`, not this module, decide what happens next."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from datetime import date

from screening_agent import config, guardrails
from screening_agent import validators as v
from screening_agent.llm.base import Message
from screening_agent.llm.client import LLMClient
from screening_agent.llm.compose import FaqContext, compose
from screening_agent.llm.extract import ExtractedFields, extract
from screening_agent.models import (
    Availability,
    CandidateProfile,
    Language,
    Schedule,
    Stage,
    Terminal,
)
from screening_agent.rag.retrieve import FaqHit
from screening_agent.rag.retrieve import retrieve as retrieve_faq
from screening_agent.stages import (
    FIELD_FOR_STAGE,
    AskStage,
    Step,
    Terminate,
    guardrail_step,
    is_field_empty,
    next_step,
)
from screening_agent.store import Store

MAX_ATTEMPTS = 2

logger = logging.getLogger(__name__)


def generate_summary(profile: CandidateProfile) -> str:
    """Built from the stored structured fields only — never re-read from the transcript, so it
    can never disagree with the data a recruiter or ATS actually receives."""
    zone = next((z for z in config.ZONES if z.id == profile.zone_id), None)
    city = zone.display_name if zone else (profile.city_raw or "unknown city")
    platforms = ", ".join(profile.experience_platforms) if profile.experience_platforms else "none"
    if profile.starts_immediately:
        start = "immediately"
    elif profile.start_date:
        start = profile.start_date.isoformat()
    else:
        start = "unspecified"
    years = "unspecified" if profile.experience_years is None else profile.experience_years
    availability = profile.availability.value if profile.availability else "unspecified"
    schedule = profile.preferred_schedule.value if profile.preferred_schedule else "unspecified"
    return (
        f"{profile.full_name or 'Unnamed candidate'} — {city}. "
        f"Licence: {'yes' if profile.has_license else 'no'}. "
        f"Availability: {availability}, {schedule} shift. "
        f"Experience: {years} years ({platforms}). "
        f"Start: {start}."
    )


class Conversation:
    def __init__(
        self,
        *,
        store: Store,
        client: LLMClient,
        conversation_id: str | None = None,
        today: date | None = None,
        faq_retriever: Callable[[str], list[FaqHit]] | None = None,
    ) -> None:
        self.store = store
        self.client = client
        self.id = conversation_id or str(uuid.uuid4())
        self.profile = CandidateProfile()
        self.attempts: dict[str, int] = {}
        self.language = Language.ES
        self.history: list[Message] = []
        self._today = (
            today  # injected in tests; date.today() only ever called here, not in validators
        )
        self._stage = Stage.GREETING
        self._outcome: Terminal | None = None
        self._disqualify_reason: str | None = None
        # Injectable (tests fake it out — no network/Chroma in the offline suite); defaults to
        # the real FAQ retriever (M6) against this conversation's own client/model.
        self._faq_retriever = faq_retriever or (
            lambda query: retrieve_faq(query, client=self.client)
        )
        store.create_conversation(self.id)

    @property
    def finished(self) -> bool:
        return self._outcome is not None

    @property
    def outcome(self) -> Terminal | None:
        return self._outcome

    def _today_date(self) -> date:
        return self._today or date.today()

    def start(self) -> str:
        """The opening GREETING message — no candidate input yet."""
        text = compose(
            self.client,
            step=AskStage(Stage.GREETING),
            history=[],
            language=self.language,
            is_first_message=True,
        )
        self.history.append(Message(role="assistant", content=text))
        self._record(candidate_message=None, agent_message=text)
        return text

    def step(self, candidate_message: str) -> str:
        if self.finished:
            raise RuntimeError("conversation already reached a terminal outcome")

        # Guardrails (M5) run before extraction and before the candidate's message joins
        # `self.history` — no need to spend a model call pulling fields out of a keyboard mash
        # or an insult, and it keeps the redirect-then-close ladder entirely independent of
        # whatever field happens to be pending.
        flag = guardrails.classify(candidate_message)
        if flag is not None:
            logger.info(
                "guardrail triggered (%s) on conversation %s: %s",
                flag,
                self.id,
                guardrails.redact_for_log(candidate_message),
            )
            self.history.append(Message(role="user", content=candidate_message))
            prior_off_script = self.attempts.get("off_script", 0)
            self.attempts["off_script"] = prior_off_script + 1
            agent_step = guardrail_step(prior_off_script, self._stage)
            text = compose(
                self.client, step=agent_step, history=self.history, language=self.language
            )
            self._finalize(agent_step, text, candidate_message)
            return text

        pending_field = FIELD_FOR_STAGE.get(self._stage)
        attempts_before = self.attempts.get(pending_field, 0) if pending_field else None

        # `extract()` appends `candidate_message` to `history` itself (see llm/extract.py) — so
        # `self.history` must NOT already contain it here, or the model sees the candidate's last
        # message twice in a row. Append only after this call. (Regression: this bug pre-dates
        # M5 and was live-verified to corrupt extraction, e.g. "Me llamo Ana García" sent twice
        # extracted as full_name="Ana GarcíaMe llamo Ana García".)
        extracted = extract(self.client, history=self.history, candidate_message=candidate_message)
        self.history.append(Message(role="user", content=candidate_message))
        if extracted.language is not None:
            self.language = extracted.language

        just_captured, validation_reason = self._apply_extraction(extracted)

        # A candidate asking a side question (M6) is not a failed or silent reply — it's a
        # different, legitimate kind of turn, and process-design.md §3 doesn't want it counted
        # toward the 2-attempt cap ("the stage does not advance", not "the stage fails").
        faq_context: FaqContext | None = None
        if extracted.faq_question:
            hits = self._faq_retriever(extracted.faq_question)
            if hits:
                faq_context = FaqContext(question=extracted.faq_question, answer=hits[0].answer)

        # A silent/off-topic reply extracts nothing for the pending field — no capture, no
        # rejection — so `attempts` never moves and rule 3's NEEDS_HUMAN cap never fires. Count
        # it as a failed attempt too — unless it was an FAQ question, see above.
        if (
            pending_field is not None
            and attempts_before is not None
            and is_field_empty(self.profile, pending_field)
            and self.attempts.get(pending_field, 0) == attempts_before
            and extracted.faq_question is None
        ):
            self.attempts[pending_field] = attempts_before + 1
            validation_reason = validation_reason or "didn't get an answer to that"

        agent_step = next_step(self.profile, self.attempts)

        text = compose(
            self.client,
            step=agent_step,
            history=self.history,
            language=self.language,
            validation_reason=validation_reason,
            just_captured=just_captured,
            faq=faq_context,
        )
        self._finalize(agent_step, text, candidate_message)
        return text

    def _finalize(self, agent_step: Step, text: str, candidate_message: str) -> None:
        self.history.append(Message(role="assistant", content=text))

        if isinstance(agent_step, AskStage):
            self._stage = agent_step.stage
        if isinstance(agent_step, Terminate):
            self._outcome = agent_step.outcome
            reason = agent_step.reason
            self._disqualify_reason = reason.value if hasattr(reason, "value") else reason
            if agent_step.outcome == Terminal.QUALIFIED:
                # stages.py Rule 5 qualifies and confirms in the same step now — record the
                # stage as WRAP_UP so a qualified conversation's last stage reads sensibly
                # instead of showing whichever field question happened to come last.
                self._stage = Stage.WRAP_UP

        self._record(candidate_message=candidate_message, agent_message=text)

        if self.finished:
            summary = generate_summary(self.profile)
            self.store.export_json(self.id, summary=summary)

    def _record(self, *, candidate_message: str | None, agent_message: str) -> None:
        self.store.record_turn(
            self.id,
            candidate_message=candidate_message,
            agent_message=agent_message,
            profile=self.profile,
            stage=self._stage,
            outcome=self._outcome,
            disqualify_reason=self._disqualify_reason,
            language=self.language,
        )

    def _apply_extraction(self, extracted: ExtractedFields) -> tuple[list[str], str | None]:
        awaiting_confirmation = (
            self.profile.has_license is False
            and self.attempts.get("has_license:needs_confirmation", 0)
            and not self.attempts.get("has_license:confirmed", 0)
        )
        if awaiting_confirmation:
            self._apply_license_confirmation(extracted)
            return [], None

        just_captured: list[str] = []
        validation_reason: str | None = None

        def fail(field: str, reason: str | None) -> None:
            nonlocal validation_reason
            self.attempts[field] = self.attempts.get(field, 0) + 1
            validation_reason = validation_reason or reason

        if extracted.full_name and self.profile.full_name is None:
            result = v.validate_full_name(extracted.full_name)
            if result.accepted:
                self.profile.full_name = result.value
                just_captured.append("name")
            else:
                fail("full_name", result.reason)

        if extracted.has_license and self.profile.has_license is None:
            result = v.validate_has_license(extracted.has_license)
            if result.accepted:
                self.profile.has_license = result.value
                if result.needs_confirmation:
                    self.attempts["has_license:needs_confirmation"] = 1
                else:
                    just_captured.append("licence")
            else:
                fail("has_license", result.reason)

        if extracted.city and self.profile.zone_id is None:
            self.profile.city_raw = extracted.city
            result = v.validate_city(extracted.city, config.ZONES)
            if result.accepted:
                self.profile.zone_id = result.value
                just_captured.append("city")
            else:
                fail("zone_id", result.reason)

        if extracted.availability and self.profile.availability is None:
            result = v.validate_availability(extracted.availability)
            if result.accepted:
                self.profile.availability = Availability(result.value)
                just_captured.append("availability")
            else:
                fail("availability", result.reason)

        if extracted.preferred_schedule and self.profile.preferred_schedule is None:
            result = v.validate_preferred_schedule(extracted.preferred_schedule)
            if result.accepted:
                self.profile.preferred_schedule = Schedule(result.value)
                just_captured.append("schedule")
            else:
                fail("preferred_schedule", result.reason)

        if extracted.experience_years and self.profile.experience_years is None:
            result = v.validate_experience_years(extracted.experience_years)
            if result.accepted:
                self.profile.experience_years = result.value
                just_captured.append("experience")
            else:
                fail("experience_years", result.reason)

        if extracted.experience_platforms:
            result = v.validate_experience_platforms(extracted.experience_platforms)
            if result.value:
                self.profile.experience_platforms = result.value

        if extracted.start_date and self.profile.start_date is None:
            result = v.validate_start_date(extracted.start_date, self._today_date())
            if result.accepted:
                self.profile.start_date = result.value.date
                self.profile.starts_immediately = result.value.immediate
                just_captured.append("start date")
            else:
                fail("start_date", result.reason)

        return just_captured, validation_reason

    def _apply_license_confirmation(self, extracted: ExtractedFields) -> None:
        result = v.validate_has_license(extracted.has_license) if extracted.has_license else None
        if result is None or not result.accepted:
            self.attempts["has_license:confirm_attempts"] = (
                self.attempts.get("has_license:confirm_attempts", 0) + 1
            )
            return
        if result.value is True:
            self.profile.has_license = True  # the hedge resolved to "actually yes, I do have it"
        self.attempts["has_license:confirmed"] = 1
