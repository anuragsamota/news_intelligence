"""Parallel research workers: one per (sub-query, source), each retrieves then grades."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from ..embeddings import similarity
from ..models import Document, RetrievalRecord, SubQuery
from ..state import Runtime, WorkerInput
from ..text import keywords, truncate

log = logging.getLogger(__name__)


class Grade(BaseModel):
    index: int
    score: float = Field(description="0..1; 0 = unrelated, 1 = directly answers the sub-question")
    reason: str = Field(description="Very short justification")


class Grades(BaseModel):
    grades: list[Grade]


GRADE_SYSTEM = """You grade search results for a news research agent.
For each numbered result decide how useful it is as evidence for the sub-question (and the overall
research question). Score 0-1: 1 = directly reports facts answering it, 0.5 = related context,
0 = off-topic, spam, or a different event with a similar name. Grade every result."""


def heuristic_grades(docs: list[Document], sub_query: SubQuery, query: str, rt: Runtime) -> list[tuple[float, str]]:
    """Blend of keyword coverage and embedding similarity, both in 0..1."""
    targets = [sub_query.search_query, query]
    sims = similarity(rt.embedder, [d.text for d in docs], targets)
    q_terms = [set(keywords(t)) for t in targets]
    out = []
    for doc, sim_row in zip(docs, sims):
        terms = set(keywords(doc.text))
        overlap = max((len(t & terms) / len(t)) if t else 0.0 for t in q_terms)
        score = 0.6 * overlap + 0.4 * max(0.0, float(sim_row.max()))
        out.append((round(min(1.0, score), 3), f"keyword overlap {overlap:.0%}"))
    return out


async def grade(docs: list[Document], sub_query: SubQuery, query: str, rt: Runtime) -> tuple[list[Document], str]:
    if not docs:
        return [], "none"
    scores = None
    method = "heuristic"
    if rt.llm:
        listing = "\n".join(
            f"[{i}] {d.title} — {d.outlet} ({d.published_date or 'n.d.'})\n    {truncate(d.content or d.snippet, 400)}"
            for i, d in enumerate(docs)
        )
        user = f"Research question: {query}\nSub-question: {sub_query.question}\n\nResults:\n{listing}"
        result = await rt.llm.structured(Grades, GRADE_SYSTEM, user)
        if result:
            by_index = {g.index: (min(1.0, max(0.0, g.score)), g.reason) for g in result.grades}
            if len(by_index) >= len(docs) // 2:
                scores = [by_index.get(i, (0.0, "not graded")) for i in range(len(docs))]
                method = "llm"
    if scores is None:
        scores = heuristic_grades(docs, sub_query, query, rt)
    threshold = rt.settings.relevance_threshold if method == "llm" else rt.settings.heuristic_relevance_threshold
    kept = []
    for doc, (score, reason) in zip(docs, scores):
        if score >= threshold:
            kept.append(doc.model_copy(update={"relevance": score, "relevance_reason": reason}))
    return kept, method


async def research_worker(inp: WorkerInput, rt: Runtime) -> dict:
    sub_query, source = inp["sub_query"], rt.source(inp["source_name"])
    record = RetrievalRecord(sub_query_id=sub_query.id, source=source.name, query=sub_query.search_query)
    try:
        docs = await source.search(sub_query.search_query, rt.settings.results_per_source)
    except Exception as exc:  # noqa: BLE001 - one failing source must not sink the run
        log.warning("%s failed for %r: %s", source.name, sub_query.search_query, exc)
        record.error = f"{type(exc).__name__}: {truncate(str(exc), 160)}"
        return {"retrieval_log": [record]}
    docs = [d.model_copy(update={"sub_query_ids": [sub_query.id]}) for d in docs if d.title or d.content]
    kept, method = await grade(docs, sub_query, inp["query"], rt)
    record.found, record.kept = len(docs), len(kept)
    return {
        "documents": kept,
        "retrieval_log": [record],
        "trace": [f"{source.name} ← {sub_query.search_query!r}: {len(kept)}/{len(docs)} relevant ({method} grading)"],
    }
