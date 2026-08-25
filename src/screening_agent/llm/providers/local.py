"""Local, in-process embeddings via `sentence-transformers` — no vendor, no API key, no network.

**Why this exists at all**, when every other provider in this package wraps a hosted API: embeddings
were the one role with no production story. Anthropic and Groq have no embeddings endpoint, and
Google's is free-tier, which `config.assert_model_allowed` (R7) refuses outside `dev` because free
tiers permit the vendor to train on submitted prompts. Candidate questions are not eligible for
that. Running the model in-process removes the vendor from the picture entirely, so the FAQ path
works in `prod` under the same rule that blocked it before.

**Why the cost is acceptable here, specifically.** Embeddings are not on the per-turn hot path.
They run in exactly two places: once per FAQ entry at index time (`rag/index.py --rebuild`, 40
chunks, offline), and once per *candidate question* at query time — not once per turn, since
`extract.py` only produces a `faq_question` when the candidate actually asked something. A local
forward pass on CPU costs single-digit milliseconds and, unlike the hosted call it replaces, has no
rate limit, no quota, and no failure mode that depends on somebody else's uptime.

**Model: `intfloat/multilingual-e5-small`** — chosen by measurement, not reputation. Calibrated
against this exact 40-entry FAQ with 17 on-topic queries (mixed ES/EN) and 10 off-topic ones:

    model                                     dim  top-1 correct  on-topic min  off-topic max
    intfloat/multilingual-e5-small            384      16/17          0.854         0.830
    intfloat/multilingual-e5-base             768      16/17          0.832         0.798
    paraphrase-multilingual-MiniLM-L12-v2     384      13/17          0.437         0.407

`e5-small` matches the 2x-larger `base` on the metric that actually matters (did the right chunk
come back first) at half the dimensions. `paraphrase-multilingual-MiniLM-L12-v2` — the model used
in a sibling project — is materially worse at *retrieval* here; it is tuned for sentence similarity
rather than question→passage search, which is a different task. Plain `all-MiniLM-L6-v2` was never
a candidate: it is English-only, and this FAQ is deliberately bilingual with cross-lingual matching
as a tested requirement.

**The `query:`/`passage:` prefixes are not decoration.** E5 is trained with them, and they are what
make the embedding *asymmetric* — the same property `task_type=RETRIEVAL_QUERY|RETRIEVAL_DOCUMENT`
bought from Gemini. Dropping them degrades retrieval measurably, so the neutral `task_type` this
package already passes around is translated here rather than discarded.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from screening_agent.llm.base import EmbedResult, Message, StructuredResult, TextResult
from screening_agent.llm.registry import ModelSpec

logger = logging.getLogger(__name__)

# E5's asymmetric prefixes, keyed by the vendor-neutral task type used everywhere else in this
# package. A model without this convention would map both to "" — hence a dict, not two constants.
_TASK_PREFIXES: dict[str, str] = {
    "RETRIEVAL_DOCUMENT": "passage: ",
    "RETRIEVAL_QUERY": "query: ",
}

_models: dict[str, Any] = {}


def _get_model(spec: ModelSpec) -> Any:
    """Loaded once per process and cached. The import is deferred to here rather than done at
    module scope because `client.py` imports every provider eagerly to build its dispatch table —
    importing torch on startup would add seconds to `screening_agent.cli`, the API server, and
    every offline test run, none of which necessarily touch embeddings at all."""
    model = _models.get(spec.model_id)
    if model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("loading local embedding model %s (first call only)", spec.model_id)
        model = SentenceTransformer(spec.model_id)
        _models[spec.model_id] = model
    return model


def embed(spec: ModelSpec, *, texts: list[str], params: dict[str, Any]) -> EmbedResult:
    model = _get_model(spec)
    prefix = _TASK_PREFIXES.get(params.get("task_type") or "", "")
    vectors = model.encode(
        [f"{prefix}{text}" for text in texts],
        normalize_embeddings=True,  # so Chroma's cosine distance is a plain dot product
        show_progress_bar=False,
    )
    return EmbedResult(vectors=[[float(v) for v in row] for row in vectors], model=spec.full_name)


def complete_text(
    spec: ModelSpec, *, messages: list[Message], params: dict[str, Any]
) -> TextResult:
    raise NotImplementedError(
        "the local provider is embeddings-only — generation stays on a hosted vendor "
        "(see registry.ROLES)"
    )


def complete_structured(
    spec: ModelSpec, *, messages: list[Message], schema: type[BaseModel], params: dict[str, Any]
) -> StructuredResult:
    raise NotImplementedError(
        "the local provider is embeddings-only — generation stays on a hosted vendor "
        "(see registry.ROLES)"
    )
