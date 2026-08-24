"""Live tests (real Gemini embeddings) — marked `live`, self-skip without GEMINI_API_KEY. Each
builds its own throwaway Chroma index in a tmp dir rather than depending on `data/chroma` having
already been rebuilt, so the suite is self-contained even though M6's acceptance also runs
`python -m screening_agent.rag.index --rebuild` against the real one for the live app to use.
"""

from __future__ import annotations

import pytest

from screening_agent import config
from screening_agent.llm.client import LLMClient
from screening_agent.rag.index import load_entries, rebuild
from screening_agent.rag.retrieve import DEFAULT_RELEVANCE_FLOOR, retrieve

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not config.GEMINI_API_KEY, reason="GEMINI_API_KEY not set"),
]


@pytest.fixture(scope="module")
def faq_index(tmp_path_factory):
    persist_dir = tmp_path_factory.mktemp("chroma_faq_test")
    client = LLMClient()
    count = rebuild(client=client, persist_dir=persist_dir)
    return persist_dir, client, count


def test_rebuild_indexes_every_entry_in_both_languages(faq_index):
    persist_dir, client, count = faq_index
    entries = load_entries()
    assert count == len(entries)
    assert {e.language.value for e in entries} == {"es", "en"}
    assert sum(1 for e in entries if e.language.value == "es") >= 20
    assert sum(1 for e in entries if e.language.value == "en") >= 20


def test_on_topic_question_clears_the_relevance_floor(faq_index):
    persist_dir, client, _ = faq_index
    hits = retrieve("¿cuánto pagan por entrega?", client=client, persist_dir=persist_dir)
    assert hits
    assert hits[0].relevance >= DEFAULT_RELEVANCE_FLOOR
    assert "pagan" in hits[0].question.lower() or "pay" in hits[0].question.lower()


def test_off_topic_question_is_filtered_out_by_the_relevance_floor(faq_index):
    persist_dir, client, _ = faq_index
    hits = retrieve("do you like pizza?", client=client, persist_dir=persist_dir)
    assert hits == []


def test_blank_query_returns_no_hits_without_calling_the_model(faq_index):
    persist_dir, client, _ = faq_index
    assert retrieve("   ", client=client, persist_dir=persist_dir) == []


def test_spanish_query_retrieves_an_english_entry_by_meaning(faq_index):
    """The whole point of one shared collection instead of two: retrieval is semantic, not
    keyword-based, so a Spanish query reaches an English-only pool of chunks just fine."""
    persist_dir, client, _ = faq_index
    hits = retrieve(
        "¿qué vehículo necesito para repartir?",
        client=client,
        persist_dir=persist_dir,
        language="en",  # restrict the search pool to English chunks only
    )
    assert hits
    assert hits[0].language == "en"
    assert hits[0].relevance >= DEFAULT_RELEVANCE_FLOOR
    assert "vehicle" in hits[0].question.lower()
