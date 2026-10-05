"""Vector database of previously collected / ingested articles.

Uses ChromaDB when installed (``pip install news-intelligence[chroma]``) and
otherwise a small JSONL + NumPy store, so the pipeline always has a local
knowledge base. Articles kept by earlier research runs are added automatically
(``NI_AUTO_INGEST``), so the system builds up memory over time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from ..embeddings import Embedder, rescale
from ..models import Document
from ..text import parse_datetime, split_sentences
from .base import Source

COLLECTION = "news_intelligence"


class VectorStore(Protocol):
    def add(self, docs: list[Document]) -> int: ...
    def search(self, query: str, k: int) -> list[tuple[Document, float]]: ...
    def count(self) -> int: ...


def _meta(doc: Document) -> dict:
    return {
        "title": doc.title,
        "url": doc.url,
        "outlet": doc.outlet,
        "domain": doc.domain,
        "provider": doc.provider,
        "published_at": doc.published_at.isoformat() if doc.published_at else "",
    }


def _doc_from(doc_id: str, text: str, meta: dict) -> Document:
    return Document(
        id=doc_id,
        title=meta.get("title") or "(untitled)",
        url=meta.get("url", ""),
        outlet=meta.get("outlet") or meta.get("domain") or "knowledge base",
        domain=meta.get("domain") or "local",
        source_type="vector",
        provider=f"vector:{meta.get('provider', 'ingest')}",
        content=text,
        published_at=parse_datetime(meta.get("published_at")),
    )


def chunk_document(doc: Document, max_chars: int = 1200) -> list[tuple[str, str]]:
    """Split long article bodies into sentence-aligned chunks -> [(chunk_id, text)]."""
    body = doc.content or doc.snippet or doc.title
    if len(body) <= max_chars:
        return [(doc.id, body)]
    chunks, current = [], ""
    for sentence in split_sentences(body):
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return [(f"{doc.id}-{i}", text) for i, text in enumerate(chunks)]


class ChromaVectorStore:
    def __init__(self, path: str, embedder: Embedder):
        import chromadb

        self.embedder = embedder
        client = chromadb.PersistentClient(path=path)
        self.collection = client.get_or_create_collection(
            f"{COLLECTION}-{embedder.name}".replace("_", "-"),
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def add(self, docs: list[Document]) -> int:
        ids, texts, metas = [], [], []
        for doc in docs:
            for chunk_id, text in chunk_document(doc):
                ids.append(chunk_id)
                texts.append(text)
                metas.append(_meta(doc))
        if ids:
            self.collection.upsert(
                ids=ids, documents=texts, metadatas=metas, embeddings=self.embedder.embed(texts).tolist()
            )
        return len(ids)

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        if self.count() == 0:
            return []
        res = self.collection.query(
            query_embeddings=self.embedder.embed([query]).tolist(),
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            (_doc_from(i, text, meta), 1.0 - dist)
            for i, text, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def count(self) -> int:
        return self.collection.count()


class LocalVectorStore:
    """Brute-force cosine search over a JSONL file - fine up to ~100k chunks."""

    def __init__(self, path: str, embedder: Embedder):
        self.embedder = embedder
        self.file = Path(path) / f"{COLLECTION}-{embedder.name}.jsonl"
        self.rows: dict[str, dict] = {}
        if self.file.exists():
            for line in self.file.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows[row["id"]] = row
        self._matrix: np.ndarray | None = None

    def add(self, docs: list[Document]) -> int:
        new = []
        for doc in docs:
            for chunk_id, text in chunk_document(doc):
                new.append({"id": chunk_id, "text": text, "meta": _meta(doc)})
        if not new:
            return 0
        for row, vec in zip(new, self.embedder.embed([r["text"] for r in new])):
            row["vec"] = vec.round(5).tolist()
            self.rows[row["id"]] = row
        self.file.parent.mkdir(parents=True, exist_ok=True)
        with self.file.open("w") as fh:
            for row in self.rows.values():
                fh.write(json.dumps(row) + "\n")
        self._matrix = None
        return len(new)

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        if not self.rows:
            return []
        rows = list(self.rows.values())
        if self._matrix is None:
            self._matrix = np.asarray([r["vec"] for r in rows], dtype=np.float32)
        sims = self._matrix @ self.embedder.embed([query])[0]
        order = np.argsort(-sims)[:k]
        return [(_doc_from(rows[i]["id"], rows[i]["text"], rows[i]["meta"]), float(sims[i])) for i in order]

    def count(self) -> int:
        return len(self.rows)


def build_vector_store(backend: str, path: str, embedder: Embedder) -> VectorStore | None:
    if backend == "none":
        return None
    if backend in ("auto", "chroma"):
        try:
            return ChromaVectorStore(path, embedder)
        except ImportError:
            if backend == "chroma":
                raise
    return LocalVectorStore(path, embedder)


class VectorStoreSource(Source):
    name = "vector_db"
    kind = "vector"

    def __init__(self, store: VectorStore, min_similarity: float = 0.15):
        self.store = store
        self.min_similarity = min_similarity

    async def search(self, query: str, k: int) -> list[Document]:
        floor = getattr(self.store.embedder, "floor", 0.0)
        hits = self.store.search(query, k)
        return [doc for doc, sim in hits if float(rescale(np.float32(sim), floor)) >= self.min_similarity]
