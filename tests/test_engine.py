"""Engine-level tests with a scripted fake LLMClient — no network, deterministic extraction, so
these exercise engine.py's own logic (which field goes to which validator, attempt bookkeeping,
storage/export) independent of any live model's extraction quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from screening_agent.engine import Conversation
from screening_agent.llm.extract import ExtractedFields
from screening_agent.models import Language, Terminal
from screening_agent.store import Store


@dataclass
class _StructuredResult:
    data: ExtractedFields


@dataclass
class _TextResult:
    text: str


@dataclass
class ScriptedClient:
    """Replays `extractions` in order, one per `step()` call. `complete_text` (compose) never
    needs to be meaningful for these tests — engine.py's control flow doesn't read the reply
    text, only the structured extraction and the pure-Python stage machine do."""

    extractions: list[ExtractedFields] = field(default_factory=list)
    _i: int = 0

    def complete_structured(self, role, *, system=None, messages=None, schema=None, **overrides):
        data = self.extractions[self._i]
        self._i += 1
        return _StructuredResult(data=data)

    def complete_text(self, role, *, system=None, messages=None, **overrides):
        return _TextResult(text=f"[stub {role} #{self._i}]")


def _store(tmp_path: Path) -> Store:
    return Store(db_path=tmp_path / "test.db")


def test_full_conversation_reaches_qualified(tmp_path):
    client = ScriptedClient(
        [
            ExtractedFields(language=Language.ES, full_name="Ana García"),
            ExtractedFields(language=Language.ES, has_license="si"),
            ExtractedFields(language=Language.ES, city="Sevilla"),
            ExtractedFields(language=Language.ES, availability="tiempo completo"),
            ExtractedFields(language=Language.ES, preferred_schedule="por la mañana"),
            ExtractedFields(
                language=Language.ES,
                experience_years="2 años",
                experience_platforms=["Glovo", "Uber Eats"],
            ),
            ExtractedFields(language=Language.ES, start_date="el lunes que viene"),
            ExtractedFields(language=Language.ES),  # closing "gracias" — nothing left to extract
        ]
    )
    conv = Conversation(store=_store(tmp_path), client=client)
    conv.start()

    messages = [
        "Me llamo Ana García",
        "Sí, tengo licencia",
        "Vivo en Sevilla",
        "Tiempo completo",
        "Por la mañana",
        "2 años, en Glovo y Uber Eats",
        "Puedo empezar el lunes que viene",
        "Perfecto, gracias!",
    ]
    for msg in messages:
        conv.step(msg)

    assert conv.finished
    assert conv.outcome == Terminal.QUALIFIED
    assert conv.profile.full_name == "Ana García"
    assert conv.profile.zone_id == "sevilla"
    assert conv.profile.experience_platforms == ["Glovo", "Uber Eats"]

    export_path = tmp_path / "exports" / f"{conv.id}.json"
    assert export_path.exists()


def test_explicit_no_license_reaches_disqualified(tmp_path):
    client = ScriptedClient([ExtractedFields(language=Language.ES, has_license="no")])
    conv = Conversation(store=_store(tmp_path), client=client)
    conv.start()

    conv.step("No, no tengo licencia")

    assert conv.finished
    assert conv.outcome == Terminal.DISQUALIFIED


def test_silence_on_pending_field_escalates_to_needs_human(tmp_path):
    """Regression test: extraction can legitimately return nothing for the field being asked
    about (off-topic or silent reply) without validators.py ever seeing — let alone rejecting —
    that field. Before the engine.py fix, `attempts` never moved in that case, so rule 3's
    NEEDS_HUMAN escalation could never fire and the same question repeated forever.

    Turn 1 replies to the GREETING, which has no pending field yet — NAME is only asked as its
    *output* — so the earliest a silent reply can count against a field is turn 2.
    """
    client = ScriptedClient([ExtractedFields(language=Language.ES)] * 3)
    conv = Conversation(store=_store(tmp_path), client=client)
    conv.start()

    conv.step("hola")  # replies to the greeting; NAME gets asked as a result
    assert not conv.finished
    assert conv.attempts.get("full_name", 0) == 0

    conv.step("no entiendo la pregunta")  # 1st silent reply to NAME
    assert not conv.finished
    assert conv.attempts.get("full_name", 0) == 1

    conv.step("sigo sin entender")  # 2nd silent reply to NAME

    assert conv.finished
    assert conv.outcome == Terminal.NEEDS_HUMAN
    assert conv.attempts.get("full_name", 0) == 2
