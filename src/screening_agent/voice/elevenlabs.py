"""ElevenLabs Scribe (speech-to-text) — voice input only.

No `language_code` is passed on purpose: Scribe auto-detects it, and this app is bilingual ES/EN
with mid-conversation code-switching — pinning one language here would fight that the same way it
would in `llm/extract.py`. `filename`'s extension is how the API infers the upload's container
format, not just a label — Chrome's `MediaRecorder` (the browser UI's source) defaults to
`audio/webm;codecs=opus`, hence `"recording.webm"` from `api.py`.

Standalone module, not wired into `llm/client.py`'s `LLMClient`/`registry.ROLES`: that machinery
exists to pick between *interchangeable* models for one role, each with a primary and a backup
vendor. There is exactly one STT vendor here and no fallback candidate, so `_resolve_with_backup`
would be pure overhead. `llm/retry.py`'s `TransportError`/`call_with_retry` are reused directly
instead — R4's "one retry layer" is deliberately vendor-agnostic.

Voice output (TTS) is deliberately not implemented — see README "Bonus features" for why.
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable

from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError

from screening_agent import config
from screening_agent.llm.retry import TransportError, call_with_retry

TRANSCRIPTION_MODEL = "scribe_v1"

_client: ElevenLabs | None = None


def _get_client() -> ElevenLabs:
    global _client
    if _client is None:
        if not config.ELEVENLABS_API_KEY:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")
        _client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
    return _client


def _is_transport_error(exc: ApiError) -> bool:
    return exc.status_code is not None and (exc.status_code == 429 or exc.status_code >= 500)


def transcribe(audio: bytes, *, filename: str, sleep: Callable[[float], None] = time.sleep) -> str:
    """Transcribes one recorded utterance. Empty or silent audio comes back as `""` rather than
    raising — the same "candidate didn't answer" case `engine.Conversation.step` already handles
    gracefully for a blank text message, so no special-casing is needed at the call site.

    `sleep` is injectable only so `tests/test_voice.py` can exercise the retry path without real
    backoff delays — same reason `llm/fallback.py::call_with_fallback` takes it, matching R4's one
    retry layer even outside `llm/`."""
    client = _get_client()

    def _call() -> str:
        buffer = io.BytesIO(audio)
        buffer.name = filename
        try:
            response = client.speech_to_text.convert(model_id=TRANSCRIPTION_MODEL, file=buffer)
        except ApiError as exc:
            if _is_transport_error(exc):
                raise TransportError(f"elevenlabs transport error: {exc}", original=exc) from exc
            raise
        return response.text.strip()

    return call_with_retry(_call, sleep=sleep)
