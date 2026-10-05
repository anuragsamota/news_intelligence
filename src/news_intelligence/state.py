"""LangGraph state + the runtime dependencies every node receives."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from .config import Settings
from .embeddings import Embedder
from .llm import LLM
from .models import (
    Claim,
    Contradiction,
    Document,
    Report,
    RetrievalRecord,
    SubQuery,
    TimelineEvent,
)
from .sources import Source
from .sources.vector import VectorStore


class ResearchState(TypedDict, total=False):
    query: str
    iteration: int
    sub_queries: Annotated[list[SubQuery], operator.add]  # every sub-query ever planned
    pending_queries: list[SubQuery]  # sub-queries for the current retrieval round
    documents: Annotated[list[Document], operator.add]  # graded docs from all workers
    retrieval_log: Annotated[list[RetrievalRecord], operator.add]
    corpus: list[Document]  # deduplicated
    ranked: list[Document]
    claims: list[Claim]
    contradictions: list[Contradiction]
    timeline: list[TimelineEvent]
    report: Report
    trace: Annotated[list[str], operator.add]


class WorkerInput(TypedDict):
    query: str
    sub_query: SubQuery
    source_name: str


@dataclass
class Runtime:
    settings: Settings
    sources: list[Source]
    embedder: Embedder
    llm: LLM | None = None
    vector_store: VectorStore | None = None
    notes: list[str] = field(default_factory=list)  # backend fallbacks, surfaced in the report
    _by_name: dict[str, Source] = field(init=False, repr=False)

    def __post_init__(self):
        self._by_name = {s.name: s for s in self.sources}

    def source(self, name: str) -> Source:
        return self._by_name[name]
