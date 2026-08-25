"""Offline tests for api.py's startup FAQ warm-up and its `/api/health` reporting — no network, no
torch. `api._warmup_faq_index` is monkeypatched in every test, so the real `local:` embedding
provider (and its lazy `sentence-transformers` import, see `providers/local.py`) is never reached.
`_get_client()` runs for real — constructing an `LLMClient` is pure/offline (see `llm/client.py`),
which is exactly what lets the R7 free-tier check keep failing startup loudly (api.py's `lifespan`
docstring explains why that call is deliberately outside the try/except this file is testing).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from screening_agent import api


def test_health_reports_faq_index_ready_after_a_successful_warmup(monkeypatch) -> None:
    monkeypatch.setattr(api, "_warmup_faq_index", lambda client: None)

    with TestClient(api.app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "faq_index": "ready"}


def test_health_reports_faq_index_unavailable_when_warmup_fails(monkeypatch) -> None:
    def _boom(client: object) -> None:
        raise RuntimeError("FAQ index missing — data/chroma was never built")

    monkeypatch.setattr(api, "_warmup_faq_index", _boom)

    with TestClient(api.app) as client:
        response = client.get("/api/health")

    # The whole point: a failed warm-up degrades the FAQ feature, it does not take the server down.
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "faq_index": "unavailable"}


def test_warmup_is_called_with_the_process_wide_llm_client(monkeypatch) -> None:
    received: list[object] = []
    monkeypatch.setattr(api, "_warmup_faq_index", lambda client: received.append(client))

    with TestClient(api.app):
        pass

    assert received == [api._get_client()]
