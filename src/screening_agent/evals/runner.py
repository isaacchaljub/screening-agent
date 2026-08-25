"""Plays a scripted candidate against a real `Conversation` (real model, real `extract`/`compose`
calls) and scores the result.

Scenarios are pre-scripted message *lists*, not a two-way simulation against a second model
playing the candidate — see `_internal/STUDY_GUIDE.md` "Why scripted messages and not an
agent-vs-agent simulation" for why. `_instrument` wraps `LLMClient.complete_text`/
`complete_structured` at the *instance* level (no `__slots__` on `LLMClient`, so this is a plain
attribute shadow) to measure latency and token usage without touching
`engine.py`/`extract.py`/`compose.py`; `pricing.py` turns the summed tokens into a $ figure per
model that has a known price.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from screening_agent import config
from screening_agent.engine import Conversation
from screening_agent.evals.pricing import cost_usd
from screening_agent.llm.client import LLMClient
from screening_agent.store import Store

SCENARIOS_DIR = Path("tests") / "evals" / "scenarios"

# `--model roles` runs the *real* `registry.ROLES` table (Haiku extracts, Sonnet composes) instead
# of forcing one model onto both jobs. A forced figure is not production's cost, because
# production never routes both calls to one model. This is the mode whose cost number the README
# can honestly call "measured".
ROLES_MODE = "roles"
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
    latency_seconds: float = 0.0  # wall-clock time across every model call in the conversation
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None  # None if any call's model has no published price
    models_used: list[str] = field(default_factory=list)  # >1 in ROLES_MODE, or after a fallback


@dataclass(frozen=True, slots=True)
class CallRecord:
    """One model call, as observed from outside the production code path."""

    model: str  # the model that actually served it — after a fallback, not the one asked for
    latency_seconds: float
    input_tokens: int
    output_tokens: int


def _instrument(client: LLMClient) -> list[CallRecord]:
    """Wraps `client`'s two call methods to time each one and record its model and token usage,
    without touching `engine.py`/`extract.py`/`compose.py` — an eval-only concern, kept out of the
    production call path. Returns the list the wrapper appends to; `run_scenario` sums it after the
    conversation finishes.

    Recording the model *per call* rather than assuming the one under test is what makes
    `ROLES_MODE` measurable: in that mode the two calls in a turn go to different models at
    different prices, and after a fallback either one can be served by a third."""
    calls: list[CallRecord] = []
    original_text = client.complete_text
    original_structured = client.complete_structured

    def record(start: float, result: Any) -> None:
        calls.append(
            CallRecord(
                model=result.model,
                latency_seconds=time.perf_counter() - start,
                input_tokens=result.input_tokens or 0,
                output_tokens=result.output_tokens or 0,
            )
        )

    def timed_text(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = original_text(*args, **kwargs)
        record(start, result)
        return result

    def timed_structured(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = original_structured(*args, **kwargs)
        record(start, result)
        return result

    client.complete_text = timed_text  # type: ignore[method-assign]
    client.complete_structured = timed_structured  # type: ignore[method-assign]
    return calls


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


def _total_cost(calls: list[CallRecord]) -> float | None:
    """Summed per call, since a blended run mixes prices. `None` if *any* call was served by a
    model with no published price in `pricing.py` — a partial total would read as a real one."""
    total = 0.0
    for call in calls:
        priced = cost_usd(
            call.model, input_tokens=call.input_tokens, output_tokens=call.output_tokens
        )
        if priced is None:
            return None
        total += priced
    return total


def run_scenario(scenario: Scenario, *, model: str, store: Store) -> ScenarioResult:
    # ROLES_MODE leaves `LLMClient` unforced, so it resolves each role through `registry.ROLES`
    # exactly as `api.py`/`cli.py` do — the point of the mode is that nothing about the call path
    # differs from production.
    client = LLMClient() if model == ROLES_MODE else LLMClient(model=model)
    calls = _instrument(client)
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

    total_latency = sum(c.latency_seconds for c in calls)
    total_input_tokens = sum(c.input_tokens for c in calls)
    total_output_tokens = sum(c.output_tokens for c in calls)

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
        latency_seconds=total_latency,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        cost_usd=_total_cost(calls),
        models_used=sorted({c.model for c in calls}),
    )


def run_all(
    scenarios: list[Scenario],
    *,
    models: list[str],
    on_model_done: Callable[[list[ScenarioResult]], None] | None = None,
) -> list[ScenarioResult]:
    """Runs every scenario against every model, in model-major order (all scenarios for one model,
    then the next). If `on_model_done` is given, it's called with the full results-so-far list
    after each model's scenarios finish — the CLI uses this to write a partial report to disk
    incrementally, so killing a long sweep only loses the model still in flight, not every model
    that already finished."""
    with tempfile.TemporaryDirectory(prefix="screening_evals_") as tmp:
        tmp_path = Path(tmp)
        store = Store(db_path=tmp_path / "evals.db", exports_dir=tmp_path / "exports")
        results: list[ScenarioResult] = []
        for model in models:
            for scenario in scenarios:
                results.append(run_scenario(scenario, model=model, store=store))
            if on_model_done is not None:
                on_model_done(list(results))
        return results
