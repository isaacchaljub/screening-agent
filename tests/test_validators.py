from datetime import date

import pytest

from screening_agent.config import ZONES
from screening_agent.models import Availability, Schedule
from screening_agent.validators import (
    StartDateAnswer,
    validate_availability,
    validate_experience_platforms,
    validate_experience_years,
    validate_full_name,
    validate_has_license,
    validate_preferred_schedule,
    validate_start_date,
)
from screening_agent.validators import validate_city as _validate_city

TODAY = date(2026, 8, 24)  # a Monday


def validate_city(raw: str):
    return _validate_city(raw, ZONES)


# --- full_name ---------------------------------------------------------------------------------


def test_full_name_accepts_two_tokens():
    result = validate_full_name("Ana García")
    assert result.accepted
    assert result.value == "Ana García"


def test_full_name_accepts_accented_hyphenated():
    result = validate_full_name("María José García-Pérez")
    assert result.accepted


def test_full_name_rejects_single_token():
    assert not validate_full_name("Ana").accepted


def test_full_name_rejects_empty():
    assert not validate_full_name("").accepted


def test_full_name_rejects_placeholder_no():
    assert not validate_full_name("no").accepted


def test_full_name_rejects_asdf():
    assert not validate_full_name("asdf").accepted


def test_full_name_strips_leaked_spanish_preamble():
    result = validate_full_name("me llamo Ana García")
    assert result.accepted
    assert result.value == "Ana García"


def test_full_name_strips_leaked_english_preamble():
    result = validate_full_name("my name is Ana García")
    assert result.accepted
    assert result.value == "Ana García"


def test_full_name_rejects_stopwords_that_survive_stripping():
    # Not a recognized preamble shape, so nothing gets stripped — and "llamo" isn't a name.
    assert not validate_full_name("Ana García llamo yo").accepted


def test_full_name_rejects_long_sentence():
    assert not validate_full_name("Ana García García García García García").accepted


def test_full_name_rejects_single_letter_initials():
    assert not validate_full_name("J K").accepted


# --- has_license ---------------------------------------------------------------------------------


def test_license_yes_spanish():
    result = validate_has_license("sí, tengo licencia")
    assert result.accepted and result.value is True


def test_license_yes_english():
    result = validate_has_license("yes I do")
    assert result.accepted and result.value is True


def test_license_no_spanish():
    result = validate_has_license("no tengo")
    assert result.accepted and result.value is False
    assert not result.needs_confirmation


def test_license_no_english():
    result = validate_has_license("no I don't have one")
    assert result.accepted and result.value is False


def test_license_hedge_spanish_needs_confirmation():
    result = validate_has_license("me lo saco en junio")
    assert result.accepted
    assert result.value is False
    assert result.needs_confirmation


def test_license_hedge_english_needs_confirmation():
    result = validate_has_license("I'm taking the test next month")
    assert result.accepted
    assert result.value is False
    assert result.needs_confirmation


def test_license_unclear_not_accepted():
    result = validate_has_license("maybe idk what that means")
    assert not result.accepted


# --- city ---------------------------------------------------------------------------------------


def test_city_exact_match():
    assert validate_city("Madrid").value == "madrid"


def test_city_alias_cdmx():
    assert validate_city("CDMX").value == "cdmx"


def test_city_accent_insensitive():
    assert validate_city("malaga").value == "malaga"  # matches "Málaga"


def test_city_alias_seville_for_sevilla():
    assert validate_city("Seville").value == "sevilla"


def test_city_case_insensitive_alias():
    assert validate_city("cdmx").value == "cdmx"


def test_city_unresolved_suggests_nearest():
    result = validate_city("Sevila")  # typo, close to Sevilla
    assert not result.accepted
    assert result.reason is not None and "Sevilla" in result.reason


def test_city_totally_unknown_lists_all():
    result = validate_city("Timbuktu")
    assert not result.accepted
    assert "Madrid" in result.reason


# --- availability / preferred_schedule -----------------------------------------------------------


def test_availability_full_time_spanish():
    assert validate_availability("tiempo completo").value == Availability.FULL_TIME


def test_availability_part_time_english():
    assert validate_availability("part time please").value == Availability.PART_TIME


def test_availability_weekends():
    assert validate_availability("solo fines de semana").value == Availability.WEEKENDS


def test_availability_invalid():
    assert not validate_availability("no se").accepted


def test_schedule_morning_spanish():
    assert validate_preferred_schedule("por la mañana").value == Schedule.MORNING


def test_schedule_evening_english():
    assert validate_preferred_schedule("evenings work best").value == Schedule.EVENING


def test_schedule_flexible():
    assert validate_preferred_schedule("me da igual, flexible").value == Schedule.FLEXIBLE


def test_schedule_invalid():
    assert not validate_preferred_schedule("purple").accepted


# --- experience_years / experience_platforms -----------------------------------------------------


def test_experience_years_plain_number():
    assert validate_experience_years("3 years").value == 3.0


def test_experience_years_none_spanish():
    assert validate_experience_years("ninguna").value == 0.0


def test_experience_years_no_tengo_experiencia():
    # Regression: the most natural way to say "I have no
    # experience" in Spanish was missing from the recognized none-phrases.
    assert validate_experience_years("no tengo experiencia").value == 0.0


def test_experience_years_none_english():
    assert validate_experience_years("none").value == 0.0


def test_experience_years_range_takes_lower_bound():
    assert validate_experience_years("2-4 years").value == 2.0


def test_experience_years_couple_of_years():
    assert validate_experience_years("un par de años").value == 2.0


def test_experience_years_invalid():
    assert not validate_experience_years("a while, dunno").accepted


def test_experience_platforms_normalizes_known_names():
    result = validate_experience_platforms("Glovo and Uber Eats")
    assert result.accepted
    assert result.value == ["Glovo", "Uber Eats"]


def test_experience_platforms_keeps_unknown_verbatim():
    result = validate_experience_platforms("Foodpanda")
    assert result.value == ["Foodpanda"]


def test_experience_platforms_empty_is_valid():
    result = validate_experience_platforms("")
    assert result.accepted
    assert result.value == []


# --- start_date ------------------------------------------------------------------------------


def test_start_date_iso_future_accepted():
    result = validate_start_date("2026-09-15", TODAY)
    assert result.accepted
    assert isinstance(result.value, StartDateAnswer)
    assert result.value.date == date(2026, 9, 15)
    assert not result.value.immediate


def test_start_date_past_rejected():
    result = validate_start_date("2020-01-01", TODAY)
    assert not result.accepted


def test_start_date_tomorrow_spanish_relative_to_injected_today():
    result = validate_start_date("mañana", TODAY)
    assert result.accepted
    assert result.value.date == date(2026, 8, 25)


def test_start_date_next_weekday_relative_to_injected_today():
    # TODAY is a Monday; "next Friday" should land four days out.
    result = validate_start_date("next friday", TODAY)
    assert result.accepted
    assert result.value.date == date(2026, 8, 28)


def test_start_date_immediate_spanish():
    result = validate_start_date("ya", TODAY)
    assert result.accepted
    assert result.value.immediate
    assert result.value.date == TODAY


def test_start_date_immediate_english():
    result = validate_start_date("I can start today", TODAY)
    assert result.accepted
    assert result.value.immediate


def test_start_date_unparseable_rejected():
    result = validate_start_date("whenever works I guess", TODAY)
    assert not result.accepted


# --- start_date: natural-language dates ---------------------------------------------------------
# Regression. Every eval scenario happened to phrase the start date in ISO format
# ("2026-09-15"), so the suite was green while the validator could not parse the way a candidate
# actually types one. Caught by a live sample conversation: "Puedo empezar el 15 de septiembre"
# was rejected twice and escalated a clean happy path to NEEDS_HUMAN.


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("el 15 de septiembre", date(2026, 9, 15)),
        ("Puedo empezar el 15 de septiembre", date(2026, 9, 15)),
        ("15 septiembre 2026", date(2026, 9, 15)),
        ("15 de septiembre de 2026", date(2026, 9, 15)),
        ("1 de octubre", date(2026, 10, 1)),
        ("setiembre 30", date(2026, 9, 30)),
        ("September 15", date(2026, 9, 15)),
        ("Sept 15th, 2026", date(2026, 9, 15)),
        ("I can start October 1", date(2026, 10, 1)),
    ],
)
def test_start_date_accepts_month_names_in_both_languages(raw, expected):
    result = validate_start_date(raw, date(2026, 8, 25))
    assert result.accepted, result.reason
    assert result.value.date == expected
    assert result.value.immediate is False


def test_start_date_month_without_a_year_rolls_forward_rather_than_reading_as_past():
    # "el 3 de enero" said in August means next January. Reading it as this year would put it
    # seven months in the past and reject a perfectly good answer.
    result = validate_start_date("el 3 de enero", date(2026, 8, 25))
    assert result.accepted
    assert result.value.date == date(2027, 1, 3)


def test_start_date_explicit_past_year_is_still_rejected():
    result = validate_start_date("15 de septiembre de 2020", date(2026, 8, 25))
    assert not result.accepted
    assert "passed" in result.reason


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("en 2 semanas", date(2026, 9, 8)),
        ("in 3 days", date(2026, 8, 28)),
        ("en un mes", date(2026, 9, 25)),
        ("la proxima semana", date(2026, 9, 1)),
        ("next week", date(2026, 9, 1)),
    ],
)
def test_start_date_accepts_relative_spans(raw, expected):
    result = validate_start_date(raw, date(2026, 8, 25))
    assert result.accepted, result.reason
    assert result.value.date == expected


def test_start_date_month_arithmetic_clamps_to_a_valid_day():
    # 31 Jan + 1 month must be 28 Feb, not an invalid 31 Feb.
    result = validate_start_date("en un mes", date(2027, 1, 31))
    assert result.accepted
    assert result.value.date == date(2027, 2, 28)


def test_start_date_numeric_formats_still_win_over_month_names():
    # An explicit numeric date is unambiguous and must not be reinterpreted by the looser
    # month-name pass that runs after it.
    result = validate_start_date("15/09/2026", date(2026, 8, 25))
    assert result.accepted
    assert result.value.date == date(2026, 9, 15)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # ⚠️ These are what `extract.py` ACTUALLY hands this validator. Its prompt says "strip
        # conversational wrapping — extract the value alone, not the sentence around it", so
        # "I can start in 3 weeks" arrives here as "3 weeks". Traced live. Testing this function
        # with whole sentences hid a contradiction between the two rules for a full eval sweep.
        ("3 weeks", date(2026, 9, 15)),
        ("3 semanas", date(2026, 9, 15)),
        ("two weeks", date(2026, 9, 8)),
        ("tres semanas", date(2026, 9, 15)),
        ("un mes", date(2026, 9, 25)),
        ("5 days", date(2026, 8, 30)),
    ],
)
def test_start_date_accepts_a_bare_relative_span_without_its_preposition(raw, expected):
    result = validate_start_date(raw, date(2026, 8, 25))
    assert result.accepted, result.reason
    assert result.value.date == expected


@pytest.mark.parametrize("raw", ["hace 3 semanas", "3 weeks ago"])
def test_start_date_rejects_a_past_relative_reference(raw):
    # Reading "3 weeks ago" as a future offset would silently book someone in the past.
    result = validate_start_date(raw, date(2026, 8, 25))
    assert not result.accepted


# --- availability: elliptical answers ------------------------------------------------------------
# Regression, found live: the agent asks "¿tiempo completo, medio tiempo o fines de semana?" and a
# real person answers "completo" — echoing back only the distinguishing word. The list held only
# the full two-word phrases, so every short form was rejected, while _SCHEDULE_PHRASES had always
# accepted the bare "tarde"/"noche". Same module, two different standards.


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("completo", Availability.FULL_TIME),
        ("completa", Availability.FULL_TIME),
        ("full", Availability.FULL_TIME),
        ("medio", Availability.PART_TIME),
        ("media", Availability.PART_TIME),
        ("parcial", Availability.PART_TIME),
        ("part", Availability.PART_TIME),
        ("finde", Availability.WEEKENDS),
        ("findes", Availability.WEEKENDS),
        ("solo sabados y domingos", Availability.WEEKENDS),
    ],
)
def test_availability_accepts_the_short_form_of_each_option(raw, expected):
    result = validate_availability(raw)
    assert result.accepted, result.reason
    assert result.value == expected


@pytest.mark.parametrize("raw", ["promedio", "aparte", "complemento", "semana", "tengo un coche"])
def test_availability_short_forms_match_on_word_boundaries_not_substrings(raw):
    # The whole reason matching moved from `in` to a word-boundary regex: "medio" lives inside
    # "promedio" and "part" inside "aparte", so substring matching plus short tokens would have
    # silently filed unrelated answers as a valid availability.
    assert not validate_availability(raw).accepted


def test_availability_full_phrases_still_work():
    for raw, expected in (
        ("tiempo completo", Availability.FULL_TIME),
        ("jornada completa", Availability.FULL_TIME),
        ("medio tiempo", Availability.PART_TIME),
        ("tiempo parcial", Availability.PART_TIME),
        ("fines de semana", Availability.WEEKENDS),
        ("full-time", Availability.FULL_TIME),
        ("part time", Availability.PART_TIME),
        ("weekends", Availability.WEEKENDS),
    ):
        assert validate_availability(raw).value == expected


# --- experience_years: units ---------------------------------------------------------------------
# ⚠️ The worst bug found in this module, because it produced a *wrong value* rather than a
# rejection: "6 meses" matched the bare-number regex and was stored as 6.0 YEARS. A recruiter would
# have been handed a six-year veteran who has been driving for six months. R3's whole point.


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("6 meses", 0.5),
        ("6 months", 0.5),
        ("18 meses", 1.5),
        ("medio año", 0.5),
        ("half a year", 0.5),
        ("2 años y medio", 2.5),
    ],
)
def test_experience_years_converts_sub_year_durations_instead_of_reading_the_bare_number(
    raw, expected
):
    result = validate_experience_years(raw)
    assert result.accepted, result.reason
    assert result.value == pytest.approx(expected)


def test_experience_years_still_reads_plain_years():
    for raw, expected in (("3 años", 3.0), ("3", 3.0), ("0.5", 0.5), ("10 years", 10.0)):
        assert validate_experience_years(raw).value == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw", ["nada", "recién empiezo", "soy nuevo", "primera vez", "just starting", "first time"]
)
def test_experience_years_treats_more_beginner_phrasings_as_zero(raw):
    result = validate_experience_years(raw)
    assert result.accepted and result.value == 0.0


@pytest.mark.parametrize("raw", ["un par", "un par de años", "a couple of years"])
def test_experience_years_reads_a_pair_as_two_not_one(raw):
    # "un par" must be checked before _WORD_NUMBERS, which would read its "un" as 1.
    assert validate_experience_years(raw).value == pytest.approx(2.0)


# --- has_license: more natural yes/no ------------------------------------------------------------


@pytest.mark.parametrize("raw", ["por supuesto", "sip", "afirmativo", "claro", "desde luego"])
def test_has_license_accepts_more_affirmatives(raw):
    result = validate_has_license(raw)
    assert result.accepted and result.value is True


@pytest.mark.parametrize("raw", ["nope", "negativo", "qué va", "nunca"])
def test_has_license_accepts_negatives_without_the_word_no(raw):
    result = validate_has_license(raw)
    assert result.accepted and result.value is False


def test_has_license_hedge_still_wins_over_a_plain_negative():
    result = validate_has_license("todavía no, me lo saco en junio")
    assert result.accepted and result.value is False and result.needs_confirmation
