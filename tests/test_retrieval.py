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


def test_spanish_query_ranks_the_right_english_entry_by_meaning(faq_index):
    """The whole point of one shared collection instead of two: retrieval is semantic, not
    keyword-based, so a Spanish query picks the correct chunk out of an English-only pool.

    ⚠️ This asserts *ranking*, not that the hit clears `DEFAULT_RELEVANCE_FLOOR` — that assertion
    held with Gemini embeddings and does not with the local model, which is a real and measured
    trade (see `rag/retrieve.py`). Cross-lingual pairs score ~0.82-0.85 on this corpus, overlapping
    the off-topic band, so no absolute floor admits all of them while rejecting junk.

    Measured cross-lingual top-1 accuracy on this FAQ, ES query against the English-only pool:
    e5-small 6/7, e5-base 5/7, e5-large 6/7 — the small model is not the weak link, and the one
    miss is the same on every size: "¿qué vehículo necesito?" ranks "What equipment do you
    provide?" first and "Do I need to provide my own vehicle?" second, which is a near-miss rather
    than a nonsense answer.

    It costs nothing in production: both language files cover the same 20 topics, so a Spanish
    question always has a Spanish chunk scoring higher than the English one, and the same-language
    chunk winning is *better* anyway — the retrieved answer text is then already in the candidate's
    language, so compose doesn't have to translate a fact. The cross-lingual path only ever binds
    for a topic present in one language alone, which this FAQ does not have. Hence: prove the
    semantics work by removing the floor, and let the production floor stay tuned for the
    same-language case it actually serves.
    """
    persist_dir, client, _ = faq_index
    hits = retrieve(
        "¿cuánto pagan por entrega?",
        client=client,
        persist_dir=persist_dir,
        language="en",  # restrict the search pool to English chunks only
        relevance_floor=0.0,  # ranking is the claim here, not threshold clearance — see above
    )
    assert hits
    assert hits[0].language == "en"
    assert "pay" in hits[0].question.lower()


def test_same_language_chunk_outranks_its_cross_lingual_twin(faq_index):
    """The production behaviour the test above is deliberately not asserting: with both languages
    in one collection, a Spanish question resolves to the Spanish chunk, comfortably above the
    floor."""
    persist_dir, client, _ = faq_index
    hits = retrieve("¿cuánto pagan por entrega?", client=client, persist_dir=persist_dir)
    assert hits
    assert hits[0].language == "es"
    assert hits[0].relevance >= DEFAULT_RELEVANCE_FLOOR
