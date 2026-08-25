"""Offline tests for voice/elevenlabs.py's error mapping/retry behaviour (fake client, no
network), plus one `live`-marked end-to-end test against real ElevenLabs Scribe and a real spoken
clip — self-skips without `ELEVENLABS_API_KEY`, same convention as `test_retrieval.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from elevenlabs.core.api_error import ApiError

from screening_agent import config
from screening_agent.llm.retry import TransportError
from screening_agent.voice import elevenlabs as voice_provider

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass
class _FakeTranscript:
    text: str


@dataclass
class _FakeSpeechToText:
    """Raises an `ApiError(status_code)` `fail_times` times, then returns `final_text`."""

    status_code: int
    fail_times: int
    final_text: str
    calls: int = 0
    filenames_seen: list[str] = field(default_factory=list)

    def convert(self, *, model_id, file):
        self.calls += 1
        self.filenames_seen.append(file.name)
        if self.calls <= self.fail_times:
            raise ApiError(status_code=self.status_code, headers=None, body={"detail": "boom"})
        return _FakeTranscript(text=self.final_text)


@dataclass
class _FakeClient:
    speech_to_text: _FakeSpeechToText


def _install(monkeypatch, *, status_code=429, fail_times=0, final_text="ok"):
    fake = _FakeClient(
        speech_to_text=_FakeSpeechToText(
            status_code=status_code, fail_times=fail_times, final_text=final_text
        )
    )
    monkeypatch.setattr(voice_provider, "_get_client", lambda: fake)
    return fake


def test_transcribe_returns_stripped_text(monkeypatch):
    fake = _install(monkeypatch, final_text="  hola, soy Ana  ")
    result = voice_provider.transcribe(b"fake-audio", filename="recording.webm")
    assert result == "hola, soy Ana"
    assert fake.speech_to_text.calls == 1
    assert fake.speech_to_text.filenames_seen == ["recording.webm"]


def test_transcribe_uses_the_configured_model_id(monkeypatch):
    seen = {}

    class _Recorder:
        def convert(self, *, model_id, file):
            seen["model_id"] = model_id
            return _FakeTranscript(text="ok")

    fake_client = _FakeClient(speech_to_text=_Recorder())
    monkeypatch.setattr(voice_provider, "_get_client", lambda: fake_client)
    voice_provider.transcribe(b"x", filename="a.webm")
    assert seen["model_id"] == voice_provider.TRANSCRIPTION_MODEL == "scribe_v1"


def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    fake = _install(monkeypatch, status_code=429, fail_times=2, final_text="ok")
    result = voice_provider.transcribe(b"x", filename="a.webm", sleep=lambda s: None)
    assert result == "ok"
    assert fake.speech_to_text.calls == 3  # 2 failures + 1 success, within MAX_ATTEMPTS


def test_persistent_rate_limit_raises_transport_error(monkeypatch):
    fake = _install(monkeypatch, status_code=429, fail_times=10, final_text="unused")
    with pytest.raises(TransportError):
        voice_provider.transcribe(b"x", filename="a.webm", sleep=lambda s: None)
    assert fake.speech_to_text.calls == 3  # MAX_ATTEMPTS, never more


def test_server_error_is_also_treated_as_transport_error(monkeypatch):
    fake = _install(monkeypatch, status_code=503, fail_times=1, final_text="ok")
    result = voice_provider.transcribe(b"x", filename="a.webm", sleep=lambda s: None)
    assert result == "ok"
    assert fake.speech_to_text.calls == 2


def test_non_transport_error_is_not_retried(monkeypatch):
    # Regression target: the 402 "paid_plan_required" this repo actually hit live against
    # text_to_speech (see voice/elevenlabs.py's docstring) must never trigger a retry loop against
    # a plan restriction that retrying cannot fix.
    fake = _install(monkeypatch, status_code=402, fail_times=10, final_text="unused")
    with pytest.raises(ApiError) as exc_info:
        voice_provider.transcribe(b"x", filename="a.webm", sleep=lambda s: None)
    assert exc_info.value.status_code == 402
    assert fake.speech_to_text.calls == 1


def test_missing_api_key_raises_clearly(monkeypatch):
    monkeypatch.setattr(voice_provider, "_client", None)
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", None)
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        voice_provider.transcribe(b"x", filename="a.webm")


@pytest.mark.live
@pytest.mark.skipif(not config.ELEVENLABS_API_KEY, reason="ELEVENLABS_API_KEY not set")
@pytest.mark.parametrize(
    "fixture_name,expected_words",
    [
        ("voice_sample_en.wav", ("ana", "license")),
        ("voice_sample_es.wav", ("ana", "carnet")),
    ],
)
def test_real_speech_transcribes_correctly(fixture_name, expected_words):
    """Real ElevenLabs Scribe call against real (synthesized-but-genuine) speech in both of this
    app's supported languages — not a synthetic tone. Word-membership, not exact match: Scribe's
    punctuation/casing choices are its own and not the contract this app depends on."""
    audio = (FIXTURES / fixture_name).read_bytes()
    transcript = voice_provider.transcribe(audio, filename=fixture_name)
    lowered = transcript.lower()
    for word in expected_words:
        assert word in lowered, f"{word!r} not found in transcript {transcript!r}"
