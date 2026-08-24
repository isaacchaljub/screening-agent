from dataclasses import dataclass, field

import pytest

from screening_agent.llm.base import SchemaError
from screening_agent.llm.extract import MAX_SCHEMA_RETRIES, ExtractedFields, extract


@dataclass
class _StructuredResult:
    data: ExtractedFields


@dataclass
class FlakyClient:
    """Raises SchemaError `fail_times` times, then returns `final`. Records every `system` prompt
    it was called with, so tests can confirm the parse error actually gets appended for a retry
    rather than the same prompt being repeated blindly."""

    fail_times: int
    final: ExtractedFields
    systems_seen: list[str] = field(default_factory=list)
    calls: int = 0

    def complete_structured(self, role, *, system=None, messages=None, schema=None, **overrides):
        self.systems_seen.append(system)
        self.calls += 1
        if self.calls <= self.fail_times:
            raise SchemaError("experience_platforms: Input should be a valid array, got null")
        return _StructuredResult(data=self.final)


def test_schema_error_retries_same_call_with_parse_error_appended():
    client = FlakyClient(fail_times=1, final=ExtractedFields(full_name="Ana García"))
    result = extract(client, history=[], candidate_message="me llamo Ana García")

    assert result.full_name == "Ana García"
    assert client.calls == 2
    assert "did not validate" not in client.systems_seen[0]
    assert "did not validate" in client.systems_seen[1]
    assert "experience_platforms" in client.systems_seen[1]


def test_schema_error_succeeds_on_the_last_allowed_attempt():
    client = FlakyClient(fail_times=MAX_SCHEMA_RETRIES, final=ExtractedFields(city="Sevilla"))
    result = extract(client, history=[], candidate_message="vivo en Sevilla")

    assert result.city == "Sevilla"
    assert client.calls == MAX_SCHEMA_RETRIES + 1


def test_schema_error_propagates_after_exhausting_retries():
    client = FlakyClient(fail_times=MAX_SCHEMA_RETRIES + 1, final=ExtractedFields())
    with pytest.raises(SchemaError):
        extract(client, history=[], candidate_message="hola")
    assert client.calls == MAX_SCHEMA_RETRIES + 1


def test_schema_error_is_never_retried_against_a_different_model():
    """This is exactly what makes it not a `TransportError`/fallback path (R5) — `extract()` only
    ever calls `client.complete_structured("extract", ...)`, the same role/model every time; there
    is no vendor-swap anywhere in this retry loop."""
    client = FlakyClient(fail_times=1, final=ExtractedFields(full_name="Ana García"))
    extract(client, history=[], candidate_message="me llamo Ana García")
    assert client.calls == 2  # both calls went through the same FlakyClient instance/model
