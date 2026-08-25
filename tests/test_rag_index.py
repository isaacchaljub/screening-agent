"""Offline tests for the Chroma collection cache added to `rag/index.py` (and consumed by
`rag/retrieve.py::warmup`) — no network, no torch. `FakeEmbedClient` stands in for `LLMClient`;
it is never routed through the real `local:` embedding provider, so `sentence-transformers` is
never imported here (see `providers/local.py`'s docstring for why that import must stay deferred).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from screening_agent.rag import index as rag_index
from screening_agent.rag.retrieve import warmup


@dataclass
class _EmbedResult:
    vectors: list[list[float]]
    model: str = "fake:test"


class FakeEmbedClient:
    """Duck-typed like `LLMClient` — `rebuild()` and `warmup()` only ever call `.embed()`."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, texts: list[str], *, task_type: str | None = None) -> _EmbedResult:
        self.calls += 1
        vectors = [[float((hash(t) + i) % 97) for i in range(self.dim)] for t in texts]
        return _EmbedResult(vectors=vectors)


def test_get_collection_is_cached_across_calls(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma"
    rag_index.rebuild(client=FakeEmbedClient(), persist_dir=persist_dir)

    first = rag_index.get_collection(persist_dir)
    second = rag_index.get_collection(persist_dir)
    assert first is second


def test_get_collection_is_keyed_by_persist_dir(tmp_path: Path) -> None:
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    rag_index.rebuild(client=FakeEmbedClient(), persist_dir=dir_a)
    rag_index.rebuild(client=FakeEmbedClient(), persist_dir=dir_b)

    assert rag_index.get_collection(dir_a) is not rag_index.get_collection(dir_b)


def test_rebuild_replaces_the_cached_handle_not_just_the_data(tmp_path: Path) -> None:
    """The bug this guards against: caching a Chroma collection object and then, on rebuild,
    deleting and recreating the underlying collection out from under it — leaving `get_collection`
    handing back a handle onto data chromadb has already deleted."""
    persist_dir = tmp_path / "chroma"
    rag_index.rebuild(client=FakeEmbedClient(), persist_dir=persist_dir)
    stale = rag_index.get_collection(persist_dir)

    rag_index.rebuild(client=FakeEmbedClient(), persist_dir=persist_dir)
    fresh = rag_index.get_collection(persist_dir)

    assert fresh is not stale
    entries = rag_index.load_entries()
    assert fresh.count() == len(entries)


def test_warmup_opens_the_collection_and_embeds_exactly_once(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma"
    rag_index.rebuild(client=FakeEmbedClient(), persist_dir=persist_dir)

    client = FakeEmbedClient()
    warmup(client, persist_dir=persist_dir)

    assert client.calls == 1


def test_warmup_primes_the_cache_so_retrieve_does_not_reopen_the_collection(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma"
    rag_index.rebuild(client=FakeEmbedClient(), persist_dir=persist_dir)

    warmup(FakeEmbedClient(), persist_dir=persist_dir)
    warmed = rag_index.get_collection(persist_dir)

    assert rag_index.get_collection(persist_dir) is warmed


def test_warmup_raises_when_the_index_was_never_built(tmp_path: Path) -> None:
    persist_dir = tmp_path / "never-rebuilt"

    with pytest.raises(Exception):  # noqa: B017 — chromadb's own "no such collection" type
        warmup(FakeEmbedClient(), persist_dir=persist_dir)
