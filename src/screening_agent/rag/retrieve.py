"""Top-k FAQ retrieval with a relevance floor.

Retrieval is by meaning, not keyword — nothing here filters by the query's own language, so a
Spanish query can surface the English-language chunk on the same topic (or vice versa) when it's
the closer semantic match. That's the whole point of embedding both language files into one
collection instead of two.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from screening_agent.llm.client import LLMClient
from screening_agent.rag.index import CHROMA_DIR, get_collection

DEFAULT_TOP_K = 3
# Not transferable between embedding models — each puts its own similarities on its own scale, so
# this must be recalibrated if `llm/providers/local.py`'s model ever changes. See README
# "Embeddings run locally" for how 0.84 was calibrated. The load-bearing gate is upstream anyway:
# `retrieve()` only ever runs on something `extract.py` already classified as a question.
DEFAULT_RELEVANCE_FLOOR = 0.84

# How many chroma candidates the interrogative tie-break (below) gets to look at before the result
# is truncated to `top_k` — wider than `top_k` so the right answer has a chance to be re-ranked
# into first place even when it wasn't chroma's closest match by raw cosine distance.
_RERANK_POOL = 8

# "¿Cuánto pagan por entrega?" (how much) and "¿Cuándo me pagan?" (when) sit close enough in this
# embedding model's space that a short, natural candidate query like "cuanto pagan" occasionally
# ranks the wrong one first (measured: 0.880 vs 0.875) — "cuánto"/"cuándo" differ by one letter and
# the model doesn't weight it. Fixing this by re-embedding or swapping models would need
# recalibrating `DEFAULT_RELEVANCE_FLOOR` against the whole FAQ; this is a pure-Python, zero-cost
# tie-break instead (same "no extra model call" approach as `guardrails.classify()`): when the
# query has a recognizable interrogative word, nudge candidates whose own question starts with a
# *matching* one ahead, and ones with a *conflicting* one behind. Candidates with no recognizable
# interrogative (most of the FAQ) are untouched either way.
_INTERROGATIVE_MARKERS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (
            (marker, category)
            for category, markers in {
                "amount": ("cuanto", "cuantos", "cuanta", "cuantas", "how much", "how many"),
                "time": ("cuando", "when"),
                "manner": ("como", "how"),
                "place": ("donde", "where"),
                "identity": ("que", "cual", "cuales", "quien", "what", "which", "who"),
                "reason": ("por que", "porque", "why"),
            }.items()
            for marker in markers
        ),
        key=lambda pair: -len(pair[0]),  # longest marker first: "how much" before "how"
    )
)
_INTERROGATIVE_MATCH_BONUS = 0.05
_INTERROGATIVE_MISMATCH_PENALTY = 0.03


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _interrogative_category(text: str) -> str | None:
    normalized = _strip_accents(text.lower())
    for marker, category in _INTERROGATIVE_MARKERS:
        if re.search(rf"\b{re.escape(marker)}\b", normalized):
            return category
    return None


@dataclass(frozen=True, slots=True)
class FaqHit:
    question: str
    answer: str
    language: str
    relevance: float  # cosine similarity, roughly [-1, 1]; higher is closer — never boosted by
    # the interrogative tie-break, so this still reflects true embedding distance for callers/tests


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
    fetch_k = max(top_k, _RERANK_POOL)
    result = collection.query(query_embeddings=[query_vector], n_results=fetch_k, where=where)

    metadatas = result["metadatas"][0] if result["metadatas"] else []
    distances = result["distances"][0] if result["distances"] else []
    query_category = _interrogative_category(query)

    candidates: list[tuple[float, FaqHit]] = []
    for metadata, distance in zip(metadatas, distances, strict=True):
        relevance = 1.0 - distance  # chroma's cosine space: distance = 1 - cosine_similarity
        if relevance < relevance_floor:
            continue
        rank_score = relevance
        if query_category is not None:
            hit_category = _interrogative_category(metadata["question"])
            if hit_category == query_category:
                rank_score += _INTERROGATIVE_MATCH_BONUS
            elif hit_category is not None:
                rank_score -= _INTERROGATIVE_MISMATCH_PENALTY
        candidates.append(
            (
                rank_score,
                FaqHit(
                    question=metadata["question"],
                    answer=metadata["answer"],
                    language=metadata["language"],
                    relevance=relevance,
                ),
            )
        )
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return [hit for _, hit in candidates[:top_k]]
