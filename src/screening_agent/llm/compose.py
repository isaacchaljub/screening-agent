"""The compose half of R2's two-call turn: stage/state + tone + validation reason → one message.

Like `extract.py`, this module only writes text — it never decides what happens next. The `Step`
it's given (`AskStage` / `Confirm` / `Terminate`) already came out of `stages.next_step()`; compose
just has to realize it in the candidate's language, inside `config.TONE`'s constraints.
"""

from __future__ import annotations

from screening_agent import config
from screening_agent.llm.base import Message
from screening_agent.llm.client import LLMClient
from screening_agent.models import Language, Stage, Terminal
from screening_agent.stages import AskStage, Confirm, Step, Terminate

_STAGE_INSTRUCTIONS: dict[Stage, str] = {
    Stage.GREETING: (
        "Greet the candidate, say you're screening applicants for a delivery-driver role at "
        "Grupo Sazón, mention their answers are saved for this application, invite them to start."
    ),
    Stage.NAME: "Ask for their full name.",
    Stage.LICENSE: (
        "Ask whether they currently hold a valid driver's licence — yes or no "
        '(Spanish: "licencia de conducir").'
    ),
    Stage.CITY: 'Ask which city or area they\'d work in (Spanish: "ciudad" / "zona").',
    Stage.AVAILABILITY: (
        "Ask their availability: full-time, part-time, or weekends "
        '(Spanish: "tiempo completo", "medio tiempo", "fines de semana").'
    ),
    Stage.SCHEDULE: (
        "Ask their preferred shift: morning, afternoon, evening, or flexible "
        '(Spanish: "mañana", "tarde", "noche", "flexible").'
    ),
    Stage.EXPERIENCE: (
        "Ask how many years of delivery experience they have, and on which platforms "
        "(e.g. Glovo, Uber Eats, Rappi) — one message, both questions."
    ),
    Stage.START_DATE: 'Ask when they could start (Spanish: "fecha de inicio").',
    Stage.WRAP_UP: (
        "Briefly confirm you've got everything you need and a recruiter will follow up soon. "
        "Do not ask another question."
    ),
}

_CONFIRM_INSTRUCTIONS: dict[str, str] = {
    "has_license": (
        "The candidate hedged about having a driver's licence (e.g. said they're getting one "
        "soon). Ask one direct yes/no confirmation: do they NOT currently have a valid licence?"
    ),
}

_OUTCOME_INSTRUCTIONS: dict[Terminal, str] = {
    Terminal.QUALIFIED: (
        "Thank them warmly and say a recruiter will reach out soon with next steps. Do not "
        "restate every field back to them — they just told you all of it."
    ),
    Terminal.DISQUALIFIED: (
        "Close warmly and briefly. Do not recite the disqualifying reason in detail — they "
        "already know what they said. If it's about not having a licence yet, leave the door "
        "open to reapply once they have it."
    ),
    Terminal.NEEDS_HUMAN: (
        "Let them know a recruiter will follow up personally to sort out the last detail. "
        "Warm and brief — not apologetic, this is a normal handoff, not an error."
    ),
    Terminal.ABANDONED: (
        "This outcome is set by the re-engagement scheduler after silence, not composed live."
    ),
}


def _build_system_prompt(
    *,
    step: Step,
    language: Language,
    validation_reason: str | None,
    just_captured: list[str],
    is_first_message: bool,
) -> str:
    tone = config.TONE
    lines = [
        "You are a recruiting assistant for Grupo Sazón, screening delivery-driver candidates "
        "over a messaging-style chat. Write ONE reply message, nothing else.",
        f"Respond ENTIRELY in {'Spanish' if language == Language.ES else 'English'} — every word "
        "of your own sentence, including field names like availability or shift options. The "
        "only exception is a loanword or platform name (e.g. Glovo, moto) the CANDIDATE "
        "themselves just used — keep those verbatim. Everything else you write is translated, "
        "even if these instructions were given to you in English.",
        f"Under {tone.max_words} words.",
    ]
    if tone.one_question_per_message:
        lines.append("At most one question. Two questions get one answer and lose the other.")
    if not is_first_message:
        if tone.no_greeting_after_first:
            lines.append('No greeting or sign-off — do not say "hi again".')
        if tone.acknowledge_before_asking:
            lines.append(
                "Acknowledge what they just said in a word or two before asking the next thing."
            )
    if tone.no_bullet_lists:
        lines.append("No bullet lists — write like a text message.")
    if language == Language.ES:
        lines.append(
            f'Spanish uses the "{tone.spanish_register}" register — a driver role, not a bank.'
        )
    lines.append(
        "Never invent policy, pay, or contract details — if you're unsure, say you'll check "
        "with the team."
    )

    if validation_reason:
        lines.append(
            f"Their last answer didn't come through clearly ({validation_reason}). "
            "Acknowledge briefly, then re-ask clearly — do not sound like an error message."
        )
    if just_captured:
        lines.append(f"You just captured: {', '.join(just_captured)}. Don't ask about these again.")

    if isinstance(step, AskStage):
        lines.append(_STAGE_INSTRUCTIONS[step.stage])
    elif isinstance(step, Confirm):
        lines.append(
            _CONFIRM_INSTRUCTIONS.get(
                step.field, f"Confirm the {step.field} answer with a direct yes/no question."
            )
        )
    elif isinstance(step, Terminate):
        lines.append(_OUTCOME_INSTRUCTIONS[step.outcome])

    return "\n".join(lines)


def compose(
    client: LLMClient,
    *,
    step: Step,
    history: list[Message],
    language: Language,
    validation_reason: str | None = None,
    just_captured: list[str] | None = None,
    is_first_message: bool = False,
) -> str:
    system = _build_system_prompt(
        step=step,
        language=language,
        validation_reason=validation_reason,
        just_captured=just_captured or [],
        is_first_message=is_first_message,
    )
    seed = history or [Message(role="user", content="(the candidate has not written anything yet)")]
    result = client.complete_text("compose", system=system, messages=seed)
    return result.text.strip()
