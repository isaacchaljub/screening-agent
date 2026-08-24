"""Plays a scripted candidate against a real `Conversation` (real model, real `extract`/`compose`
calls) and scores the result (M8).

Scenarios are pre-scripted message *lists*, not a two-way simulation against a second model
playing the candidate — deliberately: `stages.next_step()` decides flow from validated
`CandidateProfile` fields alone (R1), never from the exact wording of a reply, so a fixed message
order matching §4.1's field order is robust to however the model under test happens to phrase its
questions. That's what makes scoring "did this model extract correctly" meaningful independent of
"did it also write a good question" — the two model calls (R2) are being evaluated somewhat
separately, which is the point.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from screening_agent import config
from screening_agent.engine import Conversation
from screening_agent.llm.client import LLMClient
from screening_agent.store import Store

SCENARIOS_DIR = Path("tests") / "evals" / "scenarios"
DEFAULT_TODAY = date(2026, 8, 24)
_NOT_NULL = "<not_null>"


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    description: str
    messages: list[str]
    expected_outcome: str
    expected_disqualify_reason: str | None = None
    expected_fields: dict[str, Any] = field(default_factory=dict)
    today: str | None = None


def load_scenarios(scenarios_dir: Path = SCENARIOS_DIR) -> list[Scenario]:
    scenarios = []
    for path in sorted(scenarios_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        scenarios.append(Scenario(**data))
    return scenarios


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    scenario: str
    model: str
    expected_outcome: str
    actual_outcome: str | None
    outcome_match: bool
    field_match_ratio: float
    mismatched_fields: list[str]
    length_compliance_ratio: float
    turns_taken: int
    error: str | None = None


def _normalize(value: Any) -> Any:
    if hasattr(value, "value"):  # StrEnum (Availability, Schedule, ...)
        return value.value
    if isinstance(value, date):
        return value.isoformat()
    return value


def _fields_match(actual: Any, expected: Any) -> bool:
    if expected == _NOT_NULL:
        return actual is not None
    normalized = _normalize(actual)
    if isinstance(normalized, list) and isinstance(expected, list):
        return sorted(normalized) == sorted(expected)
    if isinstance(normalized, int | float) and isinstance(expected, int | float):
        return abs(float(normalized) - float(expected)) < 1e-6
    return normalized == expected


def _score_fields(profile: Any, expected_fields: dict[str, Any]) -> tuple[float, list[str]]:
    if not expected_fields:
        return 1.0, []
    mismatched = []
    for name, expected in expected_fields.items():
        actual = getattr(profile, name, None)
        if not _fields_match(actual, expected):
            mismatched.append(f"{name}: expected {expected!r}, got {_normalize(actual)!r}")
    matched = len(expected_fields) - len(mismatched)
    return matched / len(expected_fields), mismatched


def _score_message_lengths(transcript: list[dict[str, str]]) -> float:
    agent_messages = [t["content"] for t in transcript if t["role"] == "agent"]
    if not agent_messages:
        return 1.0
    compliant = sum(1 for m in agent_messages if len(m.split()) <= config.TONE.max_words)
    return compliant / len(agent_messages)


def run_scenario(scenario: Scenario, *, model: str, store: Store) -> ScenarioResult:
    client = LLMClient(model=model)
    today = date.fromisoformat(scenario.today) if scenario.today else DEFAULT_TODAY
    conversation_id = f"eval-{scenario.name}-{model.replace(':', '_').replace('/', '_')}"
    conv = Conversation(store=store, client=client, conversation_id=conversation_id, today=today)

    turns_taken = 0
    error: str | None = None
    try:
        conv.start()
        for message in scenario.messages:
            if conv.finished:
                break
            conv.step(message)
            turns_taken += 1
    except Exception as exc:  # noqa: BLE001 — one bad scenario must not kill the whole sweep
        error = f"{type(exc).__name__}: {exc}"

    record = store.get(conv.id)
    actual_outcome = record.outcome
    outcome_match = error is None and actual_outcome == scenario.expected_outcome
    if error is None and scenario.expected_disqualify_reason is not None:
        outcome_match = (
            outcome_match and record.disqualify_reason == scenario.expected_disqualify_reason
        )

    if error is not None:
        field_ratio, mismatched = 0.0, [error]
        length_ratio = 0.0
    else:
        field_ratio, mismatched = _score_fields(conv.profile, scenario.expected_fields)
        length_ratio = _score_message_lengths(record.transcript)

    return ScenarioResult(
        scenario=scenario.name,
        model=model,
        expected_outcome=scenario.expected_outcome,
        actual_outcome=actual_outcome,
        outcome_match=outcome_match,
        field_match_ratio=field_ratio,
        mismatched_fields=mismatched,
        length_compliance_ratio=length_ratio,
        turns_taken=turns_taken,
        error=error,
    )


def run_all(scenarios: list[Scenario], *, models: list[str]) -> list[ScenarioResult]:
    with tempfile.TemporaryDirectory(prefix="screening_evals_") as tmp:
        tmp_path = Path(tmp)
        store = Store(db_path=tmp_path / "evals.db", exports_dir=tmp_path / "exports")
        return [
            run_scenario(scenario, model=model, store=store)
            for model in models
            for scenario in scenarios
        ]
