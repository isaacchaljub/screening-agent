"""Top-k FAQ retrieval with a relevance floor (M6).

Retrieval is by meaning, not keyword — nothing here filters by the query's own language, so a
Spanish query can surface the English-language chunk on the same topic (or vice versa) when it's
the closer semantic match. That's the whole point of embedding both language files into one
collection instead of two.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb

from screening_agent.llm.client import LLMClient
from screening_agent.rag.index import CHROMA_DIR, COLLECTION_NAME

DEFAULT_TOP_K = 3
# Tuned against live Gemini embeddings on this exact FAQ set (see tests/test_retrieval.py):
# on-topic questions score ~0.75-0.82 cosine similarity against their matching chunk; off-topic
# *questions* ("what's the weather?", "do you like pizza?") score ~0.58-0.69 even at their best
# match — surprisingly high, since everything here shares a job-application register, but well
# short of on-topic. 0.72 sits in the gap. (retrieve() is only ever called on something extract.py
# already decided was a real question — declarative screening answers never reach it, so the floor
# doesn't need to separate those; that gate is extract.py's job.)
DEFAULT_RELEVANCE_FLOOR = 0.72


@dataclass(frozen=True, slots=True)
class FaqHit:
    question: str
    answer: str
    language: str
    relevance: float  # cosine similarity, roughly [-1, 1]; higher is closer


def _collection(persist_dir: Path):
    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    return chroma_client.get_collection(COLLECTION_NAME)


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

    collection = _collection(persist_dir)
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
