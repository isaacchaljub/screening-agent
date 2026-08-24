import pytest

from screening_agent.llm.fallback import call_with_fallback
from screening_agent.llm.registry import ModelSpec
from screening_agent.llm.retry import TransportError

PRIMARY = ModelSpec.parse("openai:gpt-5.6-terra")
BACKUP = ModelSpec.parse("anthropic:claude-sonnet-5")

NO_SLEEP = lambda s: None  # noqa: E731 — test-only, avoids real waits during retry


def test_primary_success_never_touches_backup():
    calls = []

    def fn(spec):
        calls.append(spec)
        return "ok"

    result = call_with_fallback(fn, primary=PRIMARY, backup=BACKUP, sleep=NO_SLEEP)
    assert result == "ok"
    assert calls == [PRIMARY]


def test_transport_failure_falls_back():
    calls = []

    def fn(spec):
        calls.append(spec)
        if spec is PRIMARY:
            raise TransportError("down")
        return "served by backup"

    result = call_with_fallback(fn, primary=PRIMARY, backup=BACKUP, sleep=NO_SLEEP)
    assert result == "served by backup"
    assert calls[0] is PRIMARY
    assert calls[-1] is BACKUP


def test_no_backup_configured_reraises():
    def fn(spec):
        raise TransportError("down")

    with pytest.raises(TransportError):
        call_with_fallback(fn, primary=PRIMARY, backup=None, sleep=NO_SLEEP)


def test_schema_error_never_falls_back():
    calls = []

    def fn(spec):
        calls.append(spec)
        raise ValueError("bad schema, not a transport failure")

    with pytest.raises(ValueError):
        call_with_fallback(fn, primary=PRIMARY, backup=BACKUP, sleep=NO_SLEEP)
    assert calls == [PRIMARY]  # never tried the backup


def test_ineligible_backup_is_not_used():
    def fn(spec):
        raise TransportError("down")

    with pytest.raises(TransportError):
        call_with_fallback(
            fn, primary=PRIMARY, backup=BACKUP, is_eligible=lambda spec: False, sleep=NO_SLEEP
        )
