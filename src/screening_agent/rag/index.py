"""Chunk → embed → persist to Chroma.

    python -m screening_agent.rag.index --rebuild

Each `## Question` block in `faq.es.md` / `faq.en.md` is one chunk — short enough on its own that
no further splitting is useful. Embeddings go through the provider layer (`LLMClient.embed`), which
in dev resolves to Google's `gemini-embedding-001` (§5) — this deliberately never requires an
OpenAI key.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from screening_agent.llm.client import LLMClient
from screening_agent.models import Language

FAQ_DIR = Path(__file__).parent
CHROMA_DIR = Path("data") / "chroma"
COLLECTION_NAME = "faq"

_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)

# Collection handles, keyed by `str(persist_dir)`. `PersistentClient(...).get_collection(...)`
# measured at ~60ms — cheap once, but `rag/retrieve.py::retrieve()` used to pay it on every single
# candidate FAQ question. Same pattern as `providers/local.py::_models` /
# `providers/chat_completions.py::_clients`: a plain module-level dict, loaded once per process.
_collections: dict[str, Any] = {}


@dataclass(frozen=True, slots=True)
class FaqEntry:
    id: str
    language: Language
    question: str
    answer: str

    @property
    def text(self) -> str:
        """What gets embedded — question and answer together, so a query phrased like either
        the question or a fact from the answer can still match."""
        return f"{self.question}\n{self.answer}"


def _parse_faq(path: Path, language: Language) -> list[FaqEntry]:
    text = path.read_text(encoding="utf-8")
    headings = list(_HEADING_RE.finditer(text))
    entries = []
    for i, heading in enumerate(headings):
        question = heading.group(1).strip()
        start = heading.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        answer = " ".join(text[start:end].strip().split())
        entries.append(
            FaqEntry(
                id=f"{language.value}-{i:02d}", language=language, question=question, answer=answer
            )
        )
    return entries


def load_entries() -> list[FaqEntry]:
    return [
        *_parse_faq(FAQ_DIR / "faq.es.md", Language.ES),
        *_parse_faq(FAQ_DIR / "faq.en.md", Language.EN),
    ]


def _persistent_client(persist_dir: Path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(persist_dir))


def get_collection(persist_dir: Path = CHROMA_DIR):
    """The `faq` collection handle, cached per `persist_dir` for the life of the process.
    `retrieve()` and `warmup()` (see `rag/retrieve.py`) both call this instead of building their
    own client — that's the whole point of the cache."""
    key = str(persist_dir)
    collection = _collections.get(key)
    if collection is None:
        collection = _persistent_client(persist_dir).get_collection(COLLECTION_NAME)
        _collections[key] = collection
    return collection


def rebuild(*, client: LLMClient | None = None, persist_dir: Path = CHROMA_DIR) -> int:
    """Re-embeds every FAQ entry and replaces the persisted collection wholesale — simpler and
    safer than diffing for a knowledge base this small (~40 chunks)."""
    client = client or LLMClient()
    entries = load_entries()

    texts = [entry.text for entry in entries]
    embeddings = client.embed(texts, task_type="RETRIEVAL_DOCUMENT").vectors

    chroma_client = _persistent_client(persist_dir)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001 — chromadb's "no such collection" error isn't public API
        pass
    collection = chroma_client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[entry.id for entry in entries],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"language": entry.language.value, "question": entry.question, "answer": entry.answer}
            for entry in entries
        ],
    )
    # Overwrite, don't just drop, the cached handle. `_collections` may still hold the handle from
    # *before* `delete_collection` above — that handle now points at data chromadb just deleted.
    # A bare `.pop()` would only fix the next `get_collection()` call by luck of re-fetching;
    # writing the fresh collection in directly is what actually retires the stale one.
    _collections[str(persist_dir)] = collection
    return len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(prog="screening_agent.rag.index")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if not args.rebuild:
        parser.error("only --rebuild is supported right now")

    count = rebuild()
    print(f"indexed {count} FAQ entries into {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
