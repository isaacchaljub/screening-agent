"""Fast-clock demo: proves the re-engagement ladder fires all three nudges in
sequence without actually waiting 3 days, by driving `run_once()` with an injected clock instead
of the real one. Live — each nudge is a real Gemini call.

    python -m screening_agent.reengage.demo

The base instant is fixed (not `datetime.now()`) so the demo is deterministic regardless of what
time it's actually run at — every checkpoint below lands in Madrid's 08:00-21:00 waking-hours
window on purpose.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from screening_agent.llm.client import LLMClient
from screening_agent.models import CandidateProfile, Language, Stage
from screening_agent.reengage.policy import RUNG_DELAYS
from screening_agent.reengage.scheduler import run_once
from screening_agent.store import Store

CONVERSATION_ID = "reengage-demo"
# 09:00 Europe/Madrid on a fixed date, expressed directly in UTC (CEST is UTC+2 in August) so the
# demo doesn't depend on zoneinfo at call time to pick the anchor.
START = datetime(2026, 8, 24, 7, 0, tzinfo=UTC)


def main() -> None:
    # A throwaway database per run, deleted first. The demo uses a fixed CONVERSATION_ID so its
    # output is easy to follow, which means a second run would otherwise die on the primary key
    # ("UNIQUE constraint failed: conversations.id"). A demo you can only run once is a demo that
    # fails the second time you show it to someone.
    db_path = Path("data") / "reengage_demo.db"
    for path in (
        db_path,
        *(db_path.with_name(db_path.name + sfx) for sfx in ("-journal", "-wal", "-shm")),
    ):
        path.unlink(missing_ok=True)  # sidecars too: a stale journal from a crashed run is a lock
    store = Store(db_path=db_path, exports_dir=Path("samples"))
    client = LLMClient()

    store.create_conversation(CONVERSATION_ID)
    profile = CandidateProfile(
        full_name="Marta Ruiz", has_license=True, city_raw="Madrid", zone_id="madrid"
    )
    store.record_turn(
        CONVERSATION_ID,
        candidate_message="Vivo en Madrid",
        agent_message="¡Perfecto, Madrid! ¿Cuál es tu disponibilidad?",
        profile=profile,
        stage=Stage.AVAILABILITY,
        outcome=None,
        disqualify_reason=None,
        language=Language.ES,
        now=START,  # so the ladder measures from this fixed instant, not real wall-clock time
    )
    print(f"[conversation {CONVERSATION_ID}] created, then went quiet at {START.isoformat()}")

    checkpoints = [
        ("t+0 (just went quiet)", START),
        ("t+45min (rung 0 due)", START + RUNG_DELAYS[0] + timedelta(minutes=1)),
        ("t+1 day (rung 1 due)", START + RUNG_DELAYS[1] + timedelta(minutes=1)),
        ("t+3 days (rung 2 due, closes)", START + RUNG_DELAYS[2] + timedelta(minutes=1)),
    ]
    for label, now in checkpoints:
        sent = run_once(store=store, client=client, now=now)
        record = store.get(CONVERSATION_ID)
        print(f"\n[{label}]  now={now.isoformat()}  nudges_sent_this_sweep={sent}")
        print(f"  outcome: {record.outcome or 'still open'}")
        if sent:
            print(f"  message: {record.transcript[-1]['content']!r}")

    print("\n[done]")


if __name__ == "__main__":
    main()
