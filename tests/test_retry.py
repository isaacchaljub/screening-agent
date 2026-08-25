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


def test_honors_groq_style_retry_delay_hint():
    # Regression (M8): Groq's 429 body reads "Please try again in 1.2075s." — a different
    # phrasing from Google's "retryDelay"/"retry in N seconds", live-verified to fall back to
    # blind exponential backoff (and exhaust all 3 attempts on an 8000 TPM budget) before this
    # pattern was added.
    waits = []

    class FakeGroqError(Exception):
        def __str__(self):
            return (
                "Error code: 429 - {'error': {'message': 'Rate limit reached ... "
                "Please try again in 1.2075s.'}}"
            )

    def fn():
        raise TransportError("rate limited", original=FakeGroqError())

    with pytest.raises(TransportError):
        call_with_retry(fn, sleep=lambda s: waits.append(s))

    assert len(waits) == 2
    assert all(1.2 <= w <= 1.8 for w in waits)


def test_hinted_delay_above_ceiling_fails_fast_instead_of_blocking():
    # Regression: a 429 hinting "try again in 3600.5s" used to be honoured verbatim, sleeping the
    # request thread for a full hour. MAX_DELAY_SECONDS caps the exponential-backoff path but
    # never applied to the hinted path. Above the ceiling, fail immediately (no sleep, no further
    # attempts) rather than hold the connection — this is what lets llm/fallback.py reach a backup
    # vendor instead of blocking.
    waits = []
    attempts = {"n": 0}

    class FakeGoogleError(Exception):
        details = {"error": {"details": [{"retryDelay": "3600.5s"}]}}

    def fn():
        attempts["n"] += 1
        raise TransportError("rate limited", original=FakeGoogleError())

    with pytest.raises(TransportError):
        call_with_retry(fn, sleep=lambda s: waits.append(s))

    assert waits == []  # never slept — a 3600.5s hint is well above MAX_DELAY_SECONDS
    assert attempts["n"] == 1  # gave up after the first failure, not all 3
