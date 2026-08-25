"""The compose half of R2's two-call turn: stage/state + tone + validation reason → one message.

Like `extract.py`, this module only writes text — it never decides what happens next. The `Step`
it's given (`AskStage` / `Confirm` / `Terminate`) already came out of `stages.next_step()`; compose
just has to realize it in the candidate's language, inside `config.TONE`'s constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

from screening_agent import config
from screening_agent.llm.base import Message
from screening_agent.llm.client import LLMClient
from screening_agent.models import Language, Stage, Terminal
from screening_agent.stages import AskStage, Confirm, Redirect, Step, Terminate


@dataclass(frozen=True, slots=True)
class FaqContext:
    """One retrieved FAQ hit to weave into the reply — process-design.md §3: "the question
    is answered from the FAQ in one sentence, then the outstanding stage question is re-asked in
    the same message. The stage does not advance." The re-ask half of that is free: `agent_step`
    is still whatever `next_step()`/the pending stage already was, since a question-only turn
    extracts no field and so doesn't advance it."""

    question: str
    answer: str


_STAGE_INSTRUCTIONS: dict[Stage, str] = {
    Stage.GREETING: (
        'Greet the candidate. Introduce yourself as "el asistente de selección de Grupo Sazón" '
        '(in Spanish) or "the screening assistant at Grupo Sazón" (in English) — always with '
        'the article, never bare "asistente"/"assistant". Say you\'re screening applicants for '
        "the delivery-driver role, mention their answers are saved for this application, then ask "
        "for their full name — don't ask a separate 'shall we start?' question first."
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
        "Close the conversation politely and briefly — one or two sentences, neutral, not "
        "scolding. Don't re-ask anything and don't explain why you're ending it. (This same "
        "outcome is also reached silently by the re-engagement scheduler after a candidate goes "
        "quiet — its own nudge messages are composed separately, in reengage/compose_nudge.)"
    ),
}


def _build_system_prompt(
    *,
    step: Step,
    language: Language,
    validation_reason: str | None,
    just_captured: list[str],
    is_first_message: bool,
    faq: FaqContext | None = None,
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
    if faq is not None:
        lines.append(
            f'They also asked: "{faq.question}". Answer that in one sentence, using ONLY this '
            f'fact — never add or guess beyond it: "{faq.answer}". Then, in the same message, '
            "continue with the instruction below."
        )

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
    elif isinstance(step, Redirect):
        lines.append(
            "Their last message wasn't a usable answer — off-topic, nonsensical, or "
            "inappropriate. Don't call that out or lecture them. Acknowledge briefly and "
            "neutrally, then naturally re-ask this: " + _STAGE_INSTRUCTIONS[step.stage]
        )

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
    faq: FaqContext | None = None,
) -> str:
    system = _build_system_prompt(
        step=step,
        language=language,
        validation_reason=validation_reason,
        just_captured=just_captured or [],
        is_first_message=is_first_message,
        faq=faq,
    )
    seed = history or [Message(role="user", content="(the candidate has not written anything yet)")]
    result = client.complete_text("compose", system=system, messages=seed)
    return result.text.strip()


# --- re-engagement nudges — a separate, smaller prompt: there's no `Step`, no validation
# reason, no prior turn to acknowledge, since the candidate hasn't replied to anything yet. ------

_NUDGE_INSTRUCTIONS: dict[int, str] = {
    0: (
        "It's been a while since they last replied mid-application. Check in warmly and "
        "briefly — are they still there / interested in continuing? One short, low-pressure "
        "question, nothing more."
    ),
    1: (
        "They've gone quiet for about a day. Lead with something they'd actually want to "
        "know — pay and shift length — using only the facts given below, then invite them "
        "back to finish applying."
    ),
    2: (
        "Final follow-up before the application closes. Kindly let them know that if you "
        "don't hear back, you'll close this application — but they're welcome to start again "
        "anytime. No guilt-tripping, no pressure."
    ),
}


def compose_nudge(
    client: LLMClient,
    *,
    nudge_index: int,
    language: Language,
    faq_facts: list[str] | None = None,
) -> str:
    tone = config.TONE
    lines = [
        "You are a recruiting assistant for Grupo Sazón. You're re-opening a delivery-driver "
        "screening conversation that paused because the candidate stopped replying. Write ONE "
        "short outbound message, nothing else — they have not replied to anything yet, so this "
        "is not a reply, it's you reaching out again.",
        f"Respond ENTIRELY in {'Spanish' if language == Language.ES else 'English'}.",
        f"Under {tone.max_words} words.",
        'No greeting like "hi again" and no sign-off.',
        "Never invent pay or policy details beyond what's given below.",
    ]
    # Shares `config.TONE` with the reply path rather than hardcoding tone rules separately —
    # needed to avoid register drift (an earlier version that didn't share it drifted into
    # Rioplatense voseo, wrong for both markets this client serves).
    if tone.one_question_per_message:
        lines.append("At most one question.")
    if tone.no_bullet_lists:
        lines.append("No bullet lists — write like a text message.")
    if language == Language.ES:
        lines.append(
            f'Spanish uses the "{tone.spanish_register}" register — never "usted", and never '
            'voseo ("vos"/"tenés"/"podés"). Neutral Spain/Mexico Spanish.'
        )
    # `acknowledge_before_asking` is deliberately NOT applied: there is no previous candidate
    # message to acknowledge — that is what makes this a nudge rather than a reply.
    lines.append(_NUDGE_INSTRUCTIONS[nudge_index])
    if faq_facts:
        lines.append(
            "Facts you may draw on, verbatim — don't add to them: " + " | ".join(faq_facts)
        )
    system = "\n".join(lines)
    seed = [Message(role="user", content="(the candidate has gone quiet)")]
    result = client.complete_text("compose", system=system, messages=seed)
    return result.text.strip()
