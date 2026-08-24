"""APScheduler wiring for the re-engagement ladder (M7). Scans non-terminal conversations on a
timer, asks `policy.next_nudge()` (pure) what to do, and only when told to, composes and persists
a nudge. All the actual timing logic lives in `policy.py` — this module's only job is "read the
store, call the policy, write the result back."

**Celery path, documented but not built** (this project uses APScheduler per
`_internal/PLAN_FOR_SONNET.md` §6, M7): at real scale, replace `BackgroundScheduler` with a Celery
beat entry (`reengage.sweep`, every `SWEEP_INTERVAL_SECONDS`) whose task body calls `run_once()`.
Nothing about `run_once()`, `Store.list_active()`, or `Store.record_nudge()` would need to change —
only what triggers the sweep and which process runs it. APScheduler's in-process timer is a fine
fit for one dev/demo server; it stops being enough the moment there's more than one worker process
(each would run its own duplicate sweep) or the process restarts on a schedule, which is exactly
where Celery beat (one scheduled dispatch, workers pull the task) earns its keep.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from apscheduler.schedulers.background import BackgroundScheduler

from screening_agent import config
from screening_agent.config import Zone
from screening_agent.llm.client import LLMClient
from screening_agent.llm.compose import compose_nudge
from screening_agent.models import Language
from screening_agent.rag.retrieve import retrieve
from screening_agent.reengage.policy import next_nudge
from screening_agent.store import Store

logger = logging.getLogger(__name__)

SWEEP_INTERVAL_SECONDS = 300  # production cadence; the fast-clock demo drives run_once directly


def _zone_for(zone_id: str | None) -> Zone | None:
    if zone_id is None:
        return None
    return next((z for z in config.ZONES if z.id == zone_id), None)


def _faq_facts_for_nudge(nudge_index: int, *, client: LLMClient, language: Language) -> list[str]:
    """Only the "value-led" 1-day rung (process-design.md §3) draws on the FAQ — the other two
    are pure check-ins/closures with nothing to look up."""
    if nudge_index != 1:
        return []
    query = (
        "¿cuánto pagan y cómo son los turnos?"
        if language == Language.ES
        else "how much do you pay and what are the shifts like?"
    )
    return [hit.answer for hit in retrieve(query, client=client, top_k=2)]


def run_once(*, store: Store, client: LLMClient, now: datetime | None = None) -> int:
    """One sweep over every non-terminal conversation. Returns how many nudges were sent. `now`
    is injectable so both the fast-clock demo (`reengage/demo.py`) and tests can move time
    forward without sleeping — production code just omits it and gets the real clock."""
    now = now or datetime.now(UTC)
    sent = 0
    for conv in store.list_active():
        decision = next_nudge(
            now=now,
            last_candidate_activity=conv.last_candidate_activity,
            nudge_count=conv.nudge_count,
            zone=_zone_for(conv.zone_id),
        )
        if not decision.send:
            continue

        language = Language(conv.language) if conv.language else Language.ES
        facts = _faq_facts_for_nudge(decision.nudge_index, client=client, language=language)
        message = compose_nudge(
            client, nudge_index=decision.nudge_index, language=language, faq_facts=facts
        )
        store.record_nudge(
            conv.id,
            message=message,
            nudge_index=decision.nudge_index,
            outcome=decision.also_terminate,
        )
        logger.info(
            "sent nudge %s to conversation %s%s",
            decision.nudge_index,
            conv.id,
            " (closing as ABANDONED)" if decision.also_terminate else "",
        )
        sent += 1
    return sent


def start(*, store: Store | None = None, client: LLMClient | None = None) -> BackgroundScheduler:
    """Wires `run_once()` into an in-process timer for the dev/demo server. Not called from
    `api.py` automatically (M4 didn't anticipate it) — a caller opts in explicitly, e.g. a
    startup hook, so the demo script and tests aren't forced to carry a background thread."""
    store = store or Store()
    client = client or LLMClient()
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        lambda: run_once(store=store, client=client),
        "interval",
        seconds=SWEEP_INTERVAL_SECONDS,
        id="reengage_sweep",
    )
    scheduler.start()
    return scheduler
