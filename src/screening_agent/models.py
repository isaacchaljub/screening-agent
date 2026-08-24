"""Domain enums and the candidate profile (§4.1, §4.2).

Pure data. No I/O, no model calls — see `stages.py` and `validators.py` for the same rule (R1).
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class Stage(StrEnum):
    GREETING = "greeting"
    NAME = "name"
    LICENSE = "license"
    CITY = "city"
    AVAILABILITY = "availability"
    SCHEDULE = "schedule"
    EXPERIENCE = "experience"
    START_DATE = "start_date"
    WRAP_UP = "wrap_up"


class Terminal(StrEnum):
    QUALIFIED = "qualified"
    DISQUALIFIED = "disqualified"
    NEEDS_HUMAN = "needs_human"
    ABANDONED = "abandoned"


class Availability(StrEnum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    WEEKENDS = "weekends"


class Schedule(StrEnum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    FLEXIBLE = "flexible"


class Language(StrEnum):
    ES = "es"
    EN = "en"


class DisqualifyReason(StrEnum):
    NO_LICENSE = "no_license"
    OUTSIDE_SERVICE_AREA = "outside_service_area"


class CandidateProfile(BaseModel):
    """Every field starts empty and is filled in as the conversation progresses.
    This is also, unmodified, the structured payload handed to the recruiter."""

    full_name: str | None = None
    has_license: bool | None = None
    city_raw: str | None = None
    zone_id: str | None = None
    availability: Availability | None = None
    preferred_schedule: Schedule | None = None
    experience_years: float | None = None
    experience_platforms: list[str] = Field(default_factory=list)
    start_date: date | None = None
    starts_immediately: bool = False
