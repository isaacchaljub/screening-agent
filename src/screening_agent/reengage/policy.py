"""The re-engagement ladder — pure, no I/O, no clock reads, no sleeping. Every input
(`now`, `last_candidate_activity`, `nudge_count`, `zone`) is injected, the same discipline as
`stages.next_step()`: this module decides *whether and which* nudge fires, `reengage/scheduler.py`
is the only thing that actually touches a clock or the store.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from screening_agent.config import Zone
from screening_agent.models import Terminal

# The ladder (process-design.md §3): ~45 min ("still there?"), 1 day (value-led — pay and shift
# length), 3 days (final note — the application will close). Index into this tuple == the
# candidate's `nudge_count` *before* this decision, i.e. which rung is next.
RUNG_DELAYS: tuple[timedelta, ...] = (
    timedelta(minutes=45),
    timedelta(days=1),
    timedelta(days=3),
)

# "Waking hours" in the candidate's own zone (process-design.md §3: "only send inside waking
# hours in the candidate's own country"). A message at 3am reads as careless, not eager, however
# well-timed the ladder itself is.
WAKING_HOURS_START = 8
WAKING_HOURS_END = 21  # exclusive


@dataclass(frozen=True, slots=True)
class NudgeDecision:
    send: bool
    nudge_index: int | None = None
    # Set on the *last* rung — that message doubles as the closing note ("if I don't hear back,
    # this application will close"), so sending it and closing the conversation happen together.
    also_terminate: Terminal | None = None


def _in_waking_hours(now: datetime, zone: Zone) -> bool:
    local_hour = now.astimezone(ZoneInfo(zone.timezone)).hour
    return WAKING_HOURS_START <= local_hour < WAKING_HOURS_END


def next_nudge(
    *,
    now: datetime,
    last_candidate_activity: datetime | None,
    nudge_count: int,
    zone: Zone | None,
) -> NudgeDecision:
    """`zone` is `None` until the CITY stage resolves it — per process-design.md §3 ("never
    before the city is known"), that alone withholds every nudge; it's also the only way to know
    which timezone's waking hours apply, so the two rules collapse into one check for free.
    """
    if zone is None or last_candidate_activity is None:
        return NudgeDecision(send=False)
    if nudge_count >= len(RUNG_DELAYS):
        return NudgeDecision(send=False)  # ladder already exhausted and closed

    elapsed = now - last_candidate_activity
    if elapsed < RUNG_DELAYS[nudge_count]:
        return NudgeDecision(send=False)
    if not _in_waking_hours(now, zone):
        return NudgeDecision(send=False)  # due, but wait for a waking hour — try again next tick

    is_last_rung = nudge_count == len(RUNG_DELAYS) - 1
    return NudgeDecision(
        send=True,
        nudge_index=nudge_count,
        also_terminate=Terminal.ABANDONED if is_last_rung else None,
    )
