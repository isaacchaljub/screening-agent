"""Primary → backup fallback (§5, R5). Only a `TransportError` that survives the retry layer
falls back — a schema/400 failure propagates unchanged, so it can be retried against the *same*
model instead (`llm/extract.py`). Backups are deliberately a different vendor at a matching
tier, so a provider outage degrades the answer instead of ending the demo.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from screening_agent import config
from screening_agent.llm.registry import ModelSpec
from screening_agent.llm.retry import TransportError, call_with_retry

logger = logging.getLogger(__name__)

T = TypeVar("T")


def default_is_eligible(spec: ModelSpec) -> bool:
    """A backup is only eligible if it would itself be allowed to run right now — e.g. R7 blocks
    falling back onto a free-tier model outside `dev`, even as a backup."""
    try:
        config.assert_model_allowed(spec.full_name)
    except config.FreeTierModelError:
        return False
    return True


def call_with_fallback(
    fn: Callable[[ModelSpec], T],
    *,
    primary: ModelSpec,
    backup: ModelSpec | None,
    is_eligible: Callable[[ModelSpec], bool] = default_is_eligible,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    try:
        return call_with_retry(lambda: fn(primary), sleep=sleep)
    except TransportError as exc:
        if backup is None or not is_eligible(backup):
            raise
        logger.warning(
            "falling back %s -> %s after %s: %s",
            primary.full_name,
            backup.full_name,
            type(exc.original or exc).__name__,
            exc,
        )
        return call_with_retry(lambda: fn(backup), sleep=sleep)
