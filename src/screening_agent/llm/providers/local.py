"""Local, in-process embeddings via `sentence-transformers` — no vendor, no API key, no network.
See README "Embeddings run locally, not against a vendor" for why this exists, the model choice,
and its costs.
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
