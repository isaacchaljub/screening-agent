from datetime import UTC, datetime, timedelta

from screening_agent.config import ZONES
from screening_agent.models import Terminal
from screening_agent.reengage.policy import RUNG_DELAYS, NudgeDecision, next_nudge

MADRID = next(z for z in ZONES if z.id == "madrid")
CDMX = next(z for z in ZONES if z.id == "cdmx")

# A fixed, known-daytime anchor in both zones — Europe/Madrid is UTC+2 in August (CEST), so
# 09:00 UTC is 11:00 in Madrid and 03:00 in Ciudad de México the same instant. Tests pick whichever
# zone/offset they need daytime or nighttime in, rather than depending on when the suite happens
# to run.
NOON_UTC = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)  # 14:00 Madrid, 06:00 CDMX — daytime in both


# --- gating: zone and last_candidate_activity ------------------------------------------------


def test_never_nudges_before_the_city_is_known():
    decision = next_nudge(
        now=NOON_UTC,
        last_candidate_activity=NOON_UTC - timedelta(days=5),
        nudge_count=0,
        zone=None,
    )
    assert decision == NudgeDecision(send=False)


def test_never_nudges_without_a_recorded_candidate_activity():
    decision = next_nudge(now=NOON_UTC, last_candidate_activity=None, nudge_count=0, zone=MADRID)
    assert decision == NudgeDecision(send=False)


# --- the ladder itself -------------------------------------------------------------------------


def test_first_rung_not_due_before_45_minutes():
    decision = next_nudge(
        now=NOON_UTC,
        last_candidate_activity=NOON_UTC - timedelta(minutes=44),
        nudge_count=0,
        zone=MADRID,
    )
    assert decision == NudgeDecision(send=False)


def test_first_rung_fires_at_45_minutes():
    decision = next_nudge(
        now=NOON_UTC,
        last_candidate_activity=NOON_UTC - RUNG_DELAYS[0],
        nudge_count=0,
        zone=MADRID,
    )
    assert decision.send is True
    assert decision.nudge_index == 0
    assert decision.also_terminate is None


def test_second_rung_fires_at_one_day_not_before():
    too_soon = next_nudge(
        now=NOON_UTC,
        last_candidate_activity=NOON_UTC - timedelta(hours=23),
        nudge_count=1,
        zone=MADRID,
    )
    assert too_soon == NudgeDecision(send=False)

    due = next_nudge(
        now=NOON_UTC, last_candidate_activity=NOON_UTC - RUNG_DELAYS[1], nudge_count=1, zone=MADRID
    )
    assert due.send is True
    assert due.nudge_index == 1
    assert due.also_terminate is None


def test_third_rung_fires_at_three_days_and_also_closes_the_conversation():
    decision = next_nudge(
        now=NOON_UTC, last_candidate_activity=NOON_UTC - RUNG_DELAYS[2], nudge_count=2, zone=MADRID
    )
    assert decision.send is True
    assert decision.nudge_index == 2
    assert decision.also_terminate == Terminal.ABANDONED


def test_three_nudge_cap_stops_the_ladder():
    decision = next_nudge(
        now=NOON_UTC,
        last_candidate_activity=NOON_UTC - timedelta(days=30),
        nudge_count=3,
        zone=MADRID,
    )
    assert decision == NudgeDecision(send=False)


# --- quiet hours, both timezones ---------------------------------------------------------------


def test_quiet_hours_withhold_a_due_nudge_in_madrid():
    # 03:00 Madrid (CEST, UTC+2) == 01:00 UTC.
    night_in_madrid = datetime(2026, 8, 25, 1, 0, tzinfo=UTC)
    decision = next_nudge(
        now=night_in_madrid,
        last_candidate_activity=night_in_madrid - RUNG_DELAYS[0],
        nudge_count=0,
        zone=MADRID,
    )
    assert decision == NudgeDecision(send=False)


def test_quiet_hours_withhold_a_due_nudge_in_cdmx():
    # 03:00 Ciudad de México (UTC-6) == 09:00 UTC.
    night_in_cdmx = datetime(2026, 8, 25, 9, 0, tzinfo=UTC)
    decision = next_nudge(
        now=night_in_cdmx,
        last_candidate_activity=night_in_cdmx - RUNG_DELAYS[0],
        nudge_count=0,
        zone=CDMX,
    )
    assert decision == NudgeDecision(send=False)


def test_same_instant_is_daytime_in_madrid_and_nighttime_in_cdmx():
    # NOON_UTC is 14:00 in Madrid (daytime) but 06:00 in CDMX (still quiet hours) — proves the
    # gate is timezone-aware per zone, not a single global clock check.
    madrid_decision = next_nudge(
        now=NOON_UTC, last_candidate_activity=NOON_UTC - RUNG_DELAYS[0], nudge_count=0, zone=MADRID
    )
    cdmx_decision = next_nudge(
        now=NOON_UTC, last_candidate_activity=NOON_UTC - RUNG_DELAYS[0], nudge_count=0, zone=CDMX
    )
    assert madrid_decision.send is True
    assert cdmx_decision.send is False


def test_a_due_nudge_fires_once_waking_hours_arrive():
    # Same elapsed time as the CDMX quiet-hours case above, but at a daytime instant instead.
    daytime_in_cdmx = datetime(2026, 8, 25, 18, 0, tzinfo=UTC)  # 12:00 CDMX
    decision = next_nudge(
        now=daytime_in_cdmx,
        last_candidate_activity=daytime_in_cdmx - RUNG_DELAYS[0] - timedelta(hours=6),
        nudge_count=0,
        zone=CDMX,
    )
    assert decision.send is True
