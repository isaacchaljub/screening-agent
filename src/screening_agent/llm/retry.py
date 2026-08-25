"""One retry layer (§5, R4). Every vendor SDK's own retry count is set to zero in its provider
module — if it weren't, three retries here times three retries there becomes nine billed calls
for one logical attempt.

Only `TransportError` is retried here (rate limit, timeout, 5xx, connection error — R5). A schema
or 400 error is a different exception (`base.SchemaError`) that this layer never sees; retrying
that against the *same* model with the parse error appended is `llm/extract.py`'s job (M3), not
this one's — a different vendor will not fix a bad schema, it will just bill you.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from typing import TypeVar

MAX_ATTEMPTS = 3
BASE_DELAY_SECONDS = 1.0
MAX_DELAY_SECONDS = 20.0

_RETRY_DELAY_PATTERNS = (
    re.compile(r"retryDelay[\"']?\s*:\s*[\"']?(\d+(?:\.\d+)?)s", re.IGNORECASE),
    re.compile(r"retry in (\d+(?:\.\d+)?)\s*seconds?", re.IGNORECASE),
    # Groq/OpenAI-compatible 429 body (live-verified, M8): "Please try again in 1.2075s." — a
    # different phrasing from Google's, and worth its own pattern rather than assuming Google's
    # covers every vendor: without this, a Groq rate limit falls back to blind exponential
    # backoff, which on an 8000 TPM free-tier budget burns through all 3 retry attempts faster
    # than the window actually refills.
    re.compile(r"try again in (\d+(?:\.\d+)?)s", re.IGNORECASE),
)

T = TypeVar("T")


class TransportError(RuntimeError):
    """A retryable, vendor-agnostic transport failure. `original` carries the vendor's own
    exception so `_wait_seconds` can pull a vendor-specific hint (e.g. Google's retry delay)."""

    def __init__(self, message: str, *, original: Exception | None = None) -> None:
        super().__init__(message)
        self.original = original


def _hinted_delay(exc: TransportError) -> float | None:
    """Parse a vendor's own suggested wait (e.g. Google's `retryDelay`/"retry in Ns") out of the
    original exception's text, so a free-tier 429 doesn't just get hammered with blind backoff."""
    source = exc.original or exc
    text = str(source)
    details = getattr(source, "details", None)
    if details:
        text = f"{text} {details}"
    for pattern in _RETRY_DELAY_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def _wait_seconds(attempt: int, exc: TransportError) -> float | None:
    """`None` means: fail now, don't sleep. A hinted delay above `MAX_DELAY_SECONDS` (e.g. a 429
    hinting "try again in 3600.5s") used to be honoured verbatim, blocking the request thread for
    up to an hour — the exponential-backoff path was already capped at `MAX_DELAY_SECONDS`, but
    the hinted path wasn't. Above the ceiling, holding the connection open buys nothing a real
    caller wants; failing immediately instead lets `llm/fallback.py` try the backup vendor."""
    hinted = _hinted_delay(exc)
    if hinted is not None:
        if hinted > MAX_DELAY_SECONDS:
            return None
        return hinted + random.uniform(0, 0.5)
    backoff = min(MAX_DELAY_SECONDS, BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
    return backoff * (0.5 + random.random())


def call_with_retry(fn: Callable[[], T], *, sleep: Callable[[float], None] = time.sleep) -> T:
    last_exc: TransportError | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except TransportError as exc:
            last_exc = exc
            if attempt == MAX_ATTEMPTS:
                break
            wait = _wait_seconds(attempt, exc)
            if wait is None:
                break
            sleep(wait)
    assert last_exc is not None  # loop always sets this before falling through
    raise last_exc
