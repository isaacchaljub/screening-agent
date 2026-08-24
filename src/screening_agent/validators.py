"""Pure, rule-based field validators (§4.4). No I/O, no model calls, no vendor or network code (R1).

Each function takes the candidate's raw text for one field and returns a `FieldResult`. `reason`
is written for a human — it is fed to the compose model as context, not shown verbatim, but should
already read like something a person would say (§process-design.md "invalid answers").

Bilingual matching is done with small curated ES/EN keyword and phrase lists rather than a model —
this module has to work with zero network access, and the vocabulary a driver-screening chat
actually sees is small and predictable enough that this holds up.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from screening_agent.config import Zone
from screening_agent.models import Availability, Schedule

MAX_ZONE_SUGGESTIONS = 2


@dataclass(frozen=True, slots=True)
class FieldResult:
    accepted: bool
    value: Any = None
    reason: str | None = None
    needs_confirmation: bool = False


@dataclass(frozen=True, slots=True)
class StartDateAnswer:
    date: date
    immediate: bool


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _norm(text: str) -> str:
    return _strip_accents(text or "").strip().lower()


# --- full_name -----------------------------------------------------------------------------

_NAME_TOKEN_RE = re.compile(r"^[a-záéíóúñüA-ZÁÉÍÓÚÑÜ'-]+$")
_NAME_PLACEHOLDERS = {
    "no",
    "n/a",
    "na",
    "asdf",
    "none",
    "ninguno",
    "ninguna",
    "x",
    "xx",
    "xxx",
    "test",
    "prueba",
    "no se",
    "no lo se",
    "not sure",
    "i dont know",
    "prefiero no decir",
}
# Extraction is asked to return the name alone, but a weak model unreliably strips a
# self-introduction preamble (e.g. returns "me llamo Ana García" instead of "Ana García") — even
# at temperature 0, since this is a prompt-adherence gap, not sampling noise. Every other
# validator in this module matches by substring, which is naturally robust to that; full_name's
# strict tokenization isn't, so strip a *recognized* preamble here rather than reject a perfectly
# good answer for it.
_NAME_PREAMBLE_RE = re.compile(
    r"^(me\s+llamo|mi\s+nombre\s+es|soy|my\s+name('?s|\s+is)|i\s*'?\s*am|call\s+me)\s+",
    re.IGNORECASE,
)
# Still-present after stripping means extraction leaked something this module doesn't recognize
# as a preamble — none of these are plausible name tokens, so reject rather than guess (R3).
_NAME_STOPWORDS = {
    "me",
    "llamo",
    "soy",
    "mi",
    "nombre",
    "es",
    "mucho",
    "gusto",
    "hola",
    "my",
    "name",
    "is",
    "im",
    "i'm",
    "hi",
    "hello",
    "call",
}


def validate_full_name(raw: str) -> FieldResult:
    text = (raw or "").strip()
    if not text:
        return FieldResult(False, None, "didn't catch a name there")
    stripped = _NAME_PREAMBLE_RE.sub("", text, count=1).strip()
    if stripped:
        text = stripped
    if _norm(text) in _NAME_PLACEHOLDERS:
        return FieldResult(False, None, "that doesn't look like a name")
    tokens = text.split()
    if len(tokens) < 2:
        return FieldResult(False, None, "need a first and last name")
    if len(tokens) > 4:
        return FieldResult(False, None, "that reads like a sentence, not just a name")
    for token in tokens:
        cleaned = token.strip("-'")
        if len(cleaned) < 2 or not _NAME_TOKEN_RE.match(token):
            return FieldResult(False, None, "that doesn't look like a full name")
        if _norm(cleaned) in _NAME_STOPWORDS:
            return FieldResult(False, None, "that reads like a sentence, not just a name")
    return FieldResult(True, " ".join(tokens), None)


# --- has_license -----------------------------------------------------------------------------

_LICENSE_HEDGE_PHRASES = (
    "me lo saco",
    "voy a sacar",
    "estoy sacando",
    "en tramite",
    "pronto",
    "proximamente",
    "todavia no",
    "aun no",
    "taking the test",
    "getting my license",
    "getting it",
    "working on it",
    "in progress",
    "in the process",
    "not yet but",
    "planning to",
    "gonna get",
    "going to get",
    "soon",
)
_LICENSE_YES_WORDS = {"si", "yes", "yeah", "yep", "claro", "tengo", "sure"}


def validate_has_license(raw: str) -> FieldResult:
    text = _norm(raw)
    if not text:
        return FieldResult(False, None, "need a yes or no on the licence")
    if any(phrase in text for phrase in _LICENSE_HEDGE_PHRASES):
        return FieldResult(True, False, "hedged on licence status", needs_confirmation=True)
    if re.search(r"\bno\b", text) and "no se" not in text and "no lo se" not in text:
        return FieldResult(True, False, None)
    if any(re.search(rf"\b{re.escape(word)}\b", text) for word in _LICENSE_YES_WORDS):
        return FieldResult(True, True, None)
    return FieldResult(False, None, "couldn't tell yes or no on the licence")


# --- city ------------------------------------------------------------------------------------


def _zone_names(zone: Zone) -> list[str]:
    return [zone.display_name, *zone.aliases]


def _nearest_zone_names(text: str, zones: Sequence[Zone], limit: int) -> list[str]:
    scored: list[tuple[float, Zone]] = []
    for zone in zones:
        best = max(
            difflib.SequenceMatcher(None, text, _norm(name)).ratio() for name in _zone_names(zone)
        )
        scored.append((best, zone))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if not scored or scored[0][0] < 0.45:
        return []
    guessed_country = scored[0][1].country
    same_country = [zone.display_name for score, zone in scored if zone.country == guessed_country]
    return same_country[:limit]


def validate_city(raw: str, zones: Sequence[Zone]) -> FieldResult:
    text = _norm(raw)
    if not text:
        return FieldResult(False, None, "didn't catch a city there")

    for zone in zones:
        if text in {_norm(name) for name in _zone_names(zone)}:
            return FieldResult(True, zone.id, None)

    for zone in zones:
        candidates = {_norm(name) for name in _zone_names(zone) if len(name) > 2}
        if any(candidate in text or text in candidate for candidate in candidates):
            return FieldResult(True, zone.id, None)

    nearest = _nearest_zone_names(text, zones, MAX_ZONE_SUGGESTIONS)
    if nearest:
        options = " or ".join(nearest)
        reason = f"didn't recognize that city — did you mean {options}?"
    else:
        listed = ", ".join(zone.display_name for zone in zones)
        reason = f"didn't recognize that city — we're in {listed}"
    return FieldResult(False, None, reason)


# --- availability / preferred_schedule --------------------------------------------------------

_AVAILABILITY_PHRASES: dict[Availability, tuple[str, ...]] = {
    Availability.FULL_TIME: ("full time", "full-time", "tiempo completo", "jornada completa"),
    Availability.PART_TIME: (
        "part time",
        "part-time",
        "medio tiempo",
        "media jornada",
        "tiempo parcial",
    ),
    Availability.WEEKENDS: ("weekends", "weekend", "fines de semana", "fin de semana"),
}

_SCHEDULE_PHRASES: dict[Schedule, tuple[str, ...]] = {
    Schedule.MORNING: ("morning", "manana", "por la manana"),
    Schedule.AFTERNOON: ("afternoon", "tarde", "por la tarde"),
    Schedule.EVENING: ("evening", "night", "noche", "por la noche"),
    Schedule.FLEXIBLE: ("flexible", "any", "cualquiera", "indiferente", "lo que sea", "whatever"),
}


def validate_availability(raw: str) -> FieldResult:
    text = _norm(raw)
    if not text:
        return FieldResult(False, None, "need full-time, part-time, or weekends")
    for value, phrases in _AVAILABILITY_PHRASES.items():
        if any(phrase in text for phrase in phrases):
            return FieldResult(True, value, None)
    return FieldResult(False, None, "need full-time, part-time, or weekends")


def validate_preferred_schedule(raw: str) -> FieldResult:
    text = _norm(raw)
    if not text:
        return FieldResult(False, None, "need morning, afternoon, evening, or flexible")
    for value, phrases in _SCHEDULE_PHRASES.items():
        if any(phrase in text for phrase in phrases):
            return FieldResult(True, value, None)
    return FieldResult(False, None, "need morning, afternoon, evening, or flexible")


# --- experience_years / experience_platforms ---------------------------------------------------

# "no tengo experiencia" ("I don't have experience") is arguably the *most* natural way to say
# this in Spanish — live-verified missing here (M8 eval sweep): extraction correctly captured the
# phrase, but this list didn't recognize it, so it fell through to "couldn't find a number of
# years" and eventually NEEDS_HUMAN, on a candidate who should have qualified with 0 years.
_NONE_EXPERIENCE_WORDS = (
    "none",
    "no experience",
    "sin experiencia",
    "no tengo experiencia",
    "no experiencia",
    "ninguna",
    "ninguno",
)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|a|to|hasta)\s*(\d+(?:\.\d+)?)")
_WORD_NUMBERS = {
    "zero": 0,
    "cero": 0,
    "one": 1,
    "un": 1,
    "una": 1,
    "uno": 1,
    "two": 2,
    "dos": 2,
    "three": 3,
    "tres": 3,
    "four": 4,
    "cuatro": 4,
    "five": 5,
    "cinco": 5,
    "six": 6,
    "seis": 6,
}


def validate_experience_years(raw: str) -> FieldResult:
    text = _norm(raw)
    if not text:
        return FieldResult(False, None, "how many years, roughly?")
    if any(word in text for word in _NONE_EXPERIENCE_WORDS):
        return FieldResult(True, 0.0, None)
    range_match = _RANGE_RE.search(text)
    if range_match:
        return FieldResult(True, float(range_match.group(1)), None)
    number_match = _NUMBER_RE.search(text)
    if number_match:
        value = float(number_match.group())
        if value < 0:
            return FieldResult(False, None, "years can't be negative")
        return FieldResult(True, value, None)
    if "par de" in text or "couple of" in text or "a couple" in text:
        return FieldResult(True, 2.0, None)
    for word, value in _WORD_NUMBERS.items():
        if re.search(rf"\b{word}\b", text):
            return FieldResult(True, float(value), None)
    return FieldResult(False, None, "couldn't find a number of years")


_KNOWN_PLATFORMS: dict[str, str] = {
    "glovo": "Glovo",
    "uber eats": "Uber Eats",
    "ubereats": "Uber Eats",
    "uber": "Uber Eats",
    "just eat": "Just Eat",
    "justeat": "Just Eat",
    "rappi": "Rappi",
    "didi food": "DiDi Food",
    "didi": "DiDi Food",
    "deliveroo": "Deliveroo",
    "amazon flex": "Amazon Flex",
    "amazon": "Amazon Flex",
}


def validate_experience_platforms(raw: str | list[str]) -> FieldResult:
    items = raw if isinstance(raw, list) else re.split(r",| y | and |/", raw or "")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _norm(item)
        if not text:
            continue
        canonical = next((name for key, name in _KNOWN_PLATFORMS.items() if key in text), None)
        value = canonical or item.strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            normalized.append(value)
    return FieldResult(True, normalized, None)


# --- start_date --------------------------------------------------------------------------------

_IMMEDIATE_PHRASES = (
    "ya",
    "hoy",
    "ahora mismo",
    "ahora",
    "de inmediato",
    "inmediatamente",
    "cuanto antes",
    "lo antes posible",
    "asap",
    "now",
    "today",
    "right away",
    "immediately",
    "as soon as possible",
)
_WEEKDAYS: dict[str, int] = {
    "monday": 0,
    "lunes": 0,
    "tuesday": 1,
    "martes": 1,
    "wednesday": 2,
    "miercoles": 2,
    "thursday": 3,
    "jueves": 3,
    "friday": 4,
    "viernes": 4,
    "saturday": 5,
    "sabado": 5,
    "sunday": 6,
    "domingo": 6,
}


def _next_weekday(today: date, target_idx: int) -> date:
    days_ahead = (target_idx - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)


def validate_start_date(raw: str, today: date) -> FieldResult:
    text = _norm(raw)
    if not text:
        return FieldResult(False, None, "when could you start?")

    if any(phrase in text for phrase in _IMMEDIATE_PHRASES):
        return FieldResult(True, StartDateAnswer(today, True), None)

    if re.search(r"\bmanana\b", text):
        return FieldResult(True, StartDateAnswer(today + timedelta(days=1), False), None)

    for name, idx in _WEEKDAYS.items():
        if name in text:
            return FieldResult(True, StartDateAnswer(_next_weekday(today, idx), False), None)

    iso_match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if iso_match:
        try:
            candidate = date.fromisoformat(iso_match.group())
        except ValueError:
            return FieldResult(False, None, "that date doesn't look valid")
        if candidate < today:
            return FieldResult(False, None, "that date has already passed")
        return FieldResult(True, StartDateAnswer(candidate, False), None)

    dmy_match = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", text)
    if dmy_match:
        day, month, year = (int(group) for group in dmy_match.groups())
        try:
            candidate = date(year, month, day)
        except ValueError:
            return FieldResult(False, None, "that date doesn't look valid")
        if candidate < today:
            return FieldResult(False, None, "that date has already passed")
        return FieldResult(True, StartDateAnswer(candidate, False), None)

    return FieldResult(False, None, "didn't catch a date there")
