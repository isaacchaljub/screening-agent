from datetime import date

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
