import pytest

from screening_agent.llm.retry import TransportError, call_with_retry


def test_succeeds_without_retry():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    assert call_with_retry(fn, sleep=lambda s: None) == "ok"
    assert len(calls) == 1


def test_retries_then_succeeds():
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TransportError("rate limited")
        return "ok"

    assert call_with_retry(fn, sleep=lambda s: None) == "ok"
    assert attempts["n"] == 3


def test_gives_up_after_max_attempts():
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        raise TransportError("still down")

    with pytest.raises(TransportError):
        call_with_retry(fn, sleep=lambda s: None)
    assert attempts["n"] == 3  # MAX_ATTEMPTS, never more


def test_non_transport_errors_are_not_retried():
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        raise ValueError("schema error, not transport")

    with pytest.raises(ValueError):
        call_with_retry(fn, sleep=lambda s: None)
    assert attempts["n"] == 1


def test_honors_google_style_retry_delay_hint():
    waits = []

    class FakeGoogleError(Exception):
        details = {"error": {"details": [{"retryDelay": "7s"}]}}

    def fn():
        raise TransportError("rate limited", original=FakeGoogleError())

    with pytest.raises(TransportError):
        call_with_retry(fn, sleep=lambda s: waits.append(s))

    assert len(waits) == 2  # slept before attempt 2 and 3, not after the final failure
    assert all(7 <= w <= 7.5 for w in waits)
