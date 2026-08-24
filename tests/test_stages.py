from datetime import date

from screening_agent.models import Availability, CandidateProfile, DisqualifyReason, Stage, Terminal
from screening_agent.stages import AskStage, Confirm, Terminate, next_step


def profile(**kwargs) -> CandidateProfile:
    return CandidateProfile(**kwargs)


# --- rule 1: licence ------------------------------------------------------------------------


def test_explicit_no_disqualifies_immediately():
    step = next_step(profile(has_license=False), {})
    assert isinstance(step, Terminate)
    assert step.outcome == Terminal.DISQUALIFIED
    assert step.reason == DisqualifyReason.NO_LICENSE


def test_hedged_no_asks_for_confirmation_first():
    attempts = {"has_license:needs_confirmation": 1}
    step = next_step(profile(has_license=False), attempts)
    assert step == Confirm("has_license")


def test_hedged_no_disqualifies_once_confirmed():
    attempts = {"has_license:needs_confirmation": 1, "has_license:confirmed": 1}
    step = next_step(profile(has_license=False), attempts)
    assert isinstance(step, Terminate)
    assert step.outcome == Terminal.DISQUALIFIED
    assert step.reason == DisqualifyReason.NO_LICENSE


def test_hedged_no_needs_human_after_unclear_confirmation_replies():
    attempts = {"has_license:needs_confirmation": 1, "has_license:confirm_attempts": 2}
    step = next_step(profile(has_license=False), attempts)
    assert isinstance(step, Terminate)
    assert step.outcome == Terminal.NEEDS_HUMAN
    assert step.reason == "has_license"


def test_has_license_true_does_not_disqualify():
    step = next_step(profile(has_license=True), {})
    assert not (isinstance(step, Terminate) and step.outcome == Terminal.DISQUALIFIED)


def test_has_license_none_falls_through_to_ask():
    step = next_step(profile(), {})
    assert step == AskStage(Stage.NAME)


# --- rule 2: city / service zone ---------------------------------------------------------------


def test_city_unresolved_under_attempt_cap_reasks():
    p = profile(full_name="Ana García", has_license=True, city_raw="Timbuktu")
    step = next_step(p, {"zone_id": 1})
    assert step == AskStage(Stage.CITY)


def test_city_unresolved_at_attempt_cap_disqualifies():
    p = profile(full_name="Ana García", has_license=True, city_raw="Timbuktu")
    step = next_step(p, {"zone_id": 2})
    assert isinstance(step, Terminate)
    assert step.outcome == Terminal.DISQUALIFIED
    assert step.reason == DisqualifyReason.OUTSIDE_SERVICE_AREA


def test_city_never_attempted_is_not_a_disqualify():
    p = profile(full_name="Ana García", has_license=True)
    step = next_step(p, {})
    assert step == AskStage(Stage.CITY)


# --- rule 3: needs human after two failed attempts ------------------------------------------


def test_name_stuck_at_cap_needs_human():
    step = next_step(profile(), {"full_name": 2})
    assert isinstance(step, Terminate)
    assert step.outcome == Terminal.NEEDS_HUMAN
    assert step.reason == "full_name"


def test_availability_stuck_at_cap_needs_human():
    p = profile(full_name="Ana García", has_license=True, zone_id="madrid")
    step = next_step(p, {"availability": 2})
    assert isinstance(step, Terminate)
    assert step.outcome == Terminal.NEEDS_HUMAN
    assert step.reason == "availability"


def test_one_failed_attempt_does_not_trigger_needs_human():
    step = next_step(profile(), {"full_name": 1})
    assert step == AskStage(Stage.NAME)


# --- rule 4: ask the first empty stage, honoring fields volunteered early -----------------------


def test_fresh_profile_asks_name():
    assert next_step(profile(), {}) == AskStage(Stage.NAME)


def test_three_fields_volunteered_at_once_skips_to_availability():
    p = profile(full_name="Ana García", has_license=True, zone_id="sevilla")
    assert next_step(p, {}) == AskStage(Stage.AVAILABILITY)


def test_zero_experience_years_counts_as_filled():
    p = profile(
        full_name="Ana García",
        has_license=True,
        zone_id="sevilla",
        availability=Availability.FULL_TIME,
        preferred_schedule="morning",
        experience_years=0.0,
    )
    assert next_step(p, {}) == AskStage(Stage.START_DATE)


def test_empty_platforms_does_not_block_progress():
    p = profile(
        full_name="Ana García",
        has_license=True,
        zone_id="sevilla",
        availability=Availability.FULL_TIME,
        preferred_schedule="morning",
        experience_years=0.0,
        experience_platforms=[],
    )
    assert next_step(p, {}) == AskStage(Stage.START_DATE)


# --- rule 5: wrap-up then qualify -----------------------------------------------------------


def _full_profile() -> CandidateProfile:
    return profile(
        full_name="Ana García",
        has_license=True,
        zone_id="sevilla",
        availability=Availability.FULL_TIME,
        preferred_schedule="morning",
        experience_years=2.0,
        start_date=date(2026, 9, 1),
    )


def test_all_fields_filled_shows_wrap_up_once():
    assert next_step(_full_profile(), {}) == AskStage(Stage.WRAP_UP)


def test_wrap_up_shown_then_qualifies():
    step = next_step(_full_profile(), {"wrap_up:shown": 1})
    assert isinstance(step, Terminate)
    assert step.outcome == Terminal.QUALIFIED
