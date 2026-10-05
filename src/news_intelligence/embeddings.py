"""Text embeddings used for grading, deduplication, clustering and the vector DB.

``HashingEmbedder`` is deterministic and fully offline (feature hashing over
stemmed unigrams + bigrams). It is good at near-duplicate detection, which is
most of what the pipeline needs. ``OllamaEmbedder`` (default, e.g.
``nomic-embed-text``) gives true semantic embeddings.

Embedders differ in how similar *unrelated* texts look (≈0 for hashing, ≈0.4
for typical neural models). :func:`similarity` rescales cosine scores by the
embedder's ``floor`` so the pipeline's thresholds mean the same thing for both.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from .config import Settings
from .text import STOPWORDS, stem, tokenize


class Embedder(Protocol):
    name: str
    floor: float

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing-{dim}"
        self.floor = 0.0

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = [stem(t) for t in tokenize(text) if t not in STOPWORDS]
        features = [(t, 1.0) for t in toks] + [(f"{a}_{b}", 0.5) for a, b in zip(toks, toks[1:])]
        for feature, weight in features:
            h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
            vec[h % self.dim] += weight if (h >> 40) & 1 else -weight
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])


class OllamaEmbedder:
    def __init__(self, model: str, base_url: str, client_kwargs: dict | None = None,
                 floor: float = 0.4, batch_size: int = 64):
        from langchain_ollama import OllamaEmbeddings

        self._impl = OllamaEmbeddings(model=model, base_url=base_url, client_kwargs=client_kwargs or {})
        self.name = f"ollama-{model.replace(':', '-').replace('/', '-')}"
        self.floor = floor
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            rows += self._impl.embed_documents(texts[i : i + self.batch_size])
        vecs = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)


class CachedEmbedder:
    """Memoises embeddings; the same snippets get embedded by several nodes."""

    def __init__(self, inner: Embedder, max_items: int = 20_000):
        self.inner = inner
        self.name = inner.name
        self.floor = inner.floor
        self._cache: dict[str, np.ndarray] = {}
        self._max = max_items

    def embed(self, texts: list[str]) -> np.ndarray:
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            if len(self._cache) + len(missing) > self._max:
                self._cache.clear()
            for text, vec in zip(missing, self.inner.embed(missing)):
                self._cache[text] = vec
        if not texts:
            return self.inner.embed([])
        return np.stack([self._cache[t] for t in texts])


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    return a @ b.T  # inputs are L2-normalised


def rescale(sims: np.ndarray, floor: float) -> np.ndarray:
    """Map cosine similarity so ``floor`` (unrelated) -> 0 and 1 (identical) -> 1."""
    if floor <= 0:
        return sims
    return np.clip((sims - floor) / (1.0 - floor), 0.0, 1.0)


def similarity(embedder: Embedder, a: list[str], b: list[str]) -> np.ndarray:
    """Calibrated pairwise similarity matrix between two lists of texts."""
    return rescale(cosine_matrix(embedder.embed(a), embedder.embed(b)), embedder.floor)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "ollama":
        from .llm import client_kwargs

        floor = 0.4 if settings.embedding_floor is None else settings.embedding_floor
        inner: Embedder = OllamaEmbedder(
            settings.embedding_model, settings.ollama_base_url, client_kwargs(settings), floor
        )
    else:
        inner = HashingEmbedder()
        if settings.embedding_floor is not None:
            inner.floor = settings.embedding_floor
    return CachedEmbedder(inner)
