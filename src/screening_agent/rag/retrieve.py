"""Top-k FAQ retrieval with a relevance floor (M6).

Retrieval is by meaning, not keyword — nothing here filters by the query's own language, so a
Spanish query can surface the English-language chunk on the same topic (or vice versa) when it's
the closer semantic match. That's the whole point of embedding both language files into one
collection instead of two.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from screening_agent.llm.client import LLMClient
from screening_agent.rag.index import CHROMA_DIR, get_collection

DEFAULT_TOP_K = 3
# Recalibrated for the local `intfloat/multilingual-e5-small` model (see
# `llm/providers/local.py`). A floor is NOT transferable between embedding models — the previous
# 0.72 was measured against Gemini and is meaningless here, because each model puts its own
# similarities on its own scale. Measured on this exact 40-entry FAQ, 17 on-topic queries (mixed
# ES/EN) against 10 off-topic ones:
#
#     on-topic  : min 0.854   max 0.907     (top-1 retrieved the right chunk 16/17 times)
#     off-topic : max 0.830   min 0.738     ("what's the weather?", "do you like pizza?", ...)
#
# 0.84 sits in that gap. Two honest caveats worth knowing rather than discovering later:
#
# 1. The gap is narrower than Gemini's was (~0.02 vs ~0.06). E5 compresses similarity into a high,
#    tight band, so this threshold discriminates less comfortably than the raw numbers suggest —
#    the *ordering* is reliable, the absolute cut is the fragile part.
# 2. It is biased slightly toward answering. A false positive answers an odd question with a real
#    FAQ fact (mild — compose may only use the retrieved text, never invent). A false negative
#    ignores a genuine question, which is what makes a candidate feel unheard. Given the asymmetry,
#    erring low is right.
#
# The load-bearing gate is upstream anyway: `retrieve()` only ever runs on something `extract.py`
# already classified as a question, so this floor never has to separate questions from answers.
DEFAULT_RELEVANCE_FLOOR = 0.84


@dataclass(frozen=True, slots=True)
class FaqHit:
    question: str
    answer: str
    language: str
    relevance: float  # cosine similarity, roughly [-1, 1]; higher is closer


def warmup(client: LLMClient, *, persist_dir: Path = CHROMA_DIR) -> None:
    """Pays, once and up front, the two costs `retrieve()` would otherwise pay on a live
    candidate's *first* FAQ question: opens (and caches, via `rag.index.get_collection`) the
    Chroma collection handle, and forces the local embedding model to load — `providers/local.py`
    loads `sentence-transformers` lazily, measured at ~6s the first call, ~7ms after. Call this
    once, at process startup (`api.py`'s `lifespan` does); nothing here is cached per-query, so
    calling it more than once just re-pays the (by-then-cheap) warm cost.

    Raises if the collection is missing or the model fails to load — this function does not
    decide whether that's fatal, the caller does. `api.py` treats it as non-fatal: FAQ retrieval
    is a degradable feature, not a privacy control like R7's startup check."""
    get_collection(persist_dir)
    client.embed(["warmup"], task_type="RETRIEVAL_QUERY")


def retrieve(
    query: str,
    *,
    client: LLMClient,
    top_k: int = DEFAULT_TOP_K,
    relevance_floor: float = DEFAULT_RELEVANCE_FLOOR,
    persist_dir: Path = CHROMA_DIR,
    language: str | None = None,
) -> list[FaqHit]:
    """`language`, when given, restricts the search pool to that language's chunks before
    ranking — not used by the live compose path (retrieval there is deliberately
    language-agnostic, see the module docstring), but useful to prove cross-lingual retrieval
    isn't just an accident of which language usually wins (tests/test_retrieval.py)."""
    if not query.strip():
        return []

    collection = get_collection(persist_dir)
    query_vector = client.embed([query], task_type="RETRIEVAL_QUERY").vectors[0]
    where = {"language": language} if language is not None else None
    result = collection.query(query_embeddings=[query_vector], n_results=top_k, where=where)

    hits: list[FaqHit] = []
    metadatas = result["metadatas"][0] if result["metadatas"] else []
    distances = result["distances"][0] if result["distances"] else []
    for metadata, distance in zip(metadatas, distances, strict=True):
        relevance = 1.0 - distance  # chroma's cosine space: distance = 1 - cosine_similarity
        if relevance < relevance_floor:
            continue
        hits.append(
            FaqHit(
                question=metadata["question"],
                answer=metadata["answer"],
                language=metadata["language"],
                relevance=relevance,
            )
        )
    return hits
