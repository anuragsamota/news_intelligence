"""Query decomposition (``plan``) and coverage reflection (``reflect``)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..models import Document, SubQuery
from ..state import ResearchState, Runtime
from ..text import keyword_query, keywords, stable_id, truncate


class PlannedQuery(BaseModel):
    question: str = Field(description="A focused sub-question of the research question")
    search_query: str = Field(description="Keyword query (<= 8 words) for news/web search engines")
    source_kinds: list[Literal["news", "web", "vector"]] = Field(
        description="Where to look: 'news' for reporting, 'web' for background/explainers/primary "
        "documents, 'vector' for the local archive of previously collected articles"
    )
    rationale: str = Field(description="Why this sub-question is needed, one sentence")


class Plan(BaseModel):
    sub_queries: list[PlannedQuery]


PLAN_SYSTEM = """You are the planning module of a news research agent.
Decompose the user's research question into at most {n} focused, non-overlapping sub-questions
that together cover: the core facts, the latest developments, background and context, the positions
of key stakeholders, and any disputed or uncertain points. Prefer concrete sub-questions that a
news search can answer. Search queries must be short keyword strings, not sentences."""

FOLLOWUP_SYSTEM = """You are the reflection module of a news research agent.
Some sub-questions returned too little relevant evidence. Propose at most {n} new searches that are
likely to fill those gaps: rephrase with different keywords, use broader or more specific terms,
name the likely entities, or target primary sources. Do not repeat previous search queries."""


def _to_sub_queries(planned: list[PlannedQuery], round_: int, kinds: set[str]) -> list[SubQuery]:
    out = []
    for p in planned:
        wanted = [k for k in p.source_kinds if k in kinds] or sorted(kinds)
        out.append(
            SubQuery(
                id=stable_id(p.search_query, str(round_), size=8),
                question=p.question.strip(),
                search_query=truncate(p.search_query.strip(), 120),
                source_kinds=wanted,
                rationale=p.rationale,
                round=round_,
            )
        )
    return out


def heuristic_plan(query: str, n: int) -> list[PlannedQuery]:
    core = keyword_query(query)
    facets = [
        (query, core, "Core facts of the question"),
        (f"Latest developments: {query}", f"{core} recent developments", "Most recent reporting"),
        (f"Background and context: {query}", f"{core} background explained",
         "Context needed to interpret the news"),
        (f"Stakeholder reactions: {query}", f"{core} reaction response", "Stakeholder positions"),
        (f"Disputed or uncertain points: {query}", f"{core} dispute denies criticism",
         "Disputed points and counter-claims"),
    ]
    return [
        PlannedQuery(question=q, search_query=s, source_kinds=["news", "web", "vector"], rationale=r)
        for q, s, r in facets[: max(1, n)]
    ]


async def plan(state: ResearchState, rt: Runtime) -> dict:
    query = state["query"]
    n = rt.settings.max_sub_queries
    kinds = {s.kind for s in rt.sources} or {"news"}
    planned: list[PlannedQuery] | None = None
    method = "heuristic"
    if rt.llm:
        result = await rt.llm.structured(Plan, PLAN_SYSTEM.format(n=n), f"Research question: {query}")
        if result and result.sub_queries:
            planned, method = result.sub_queries[:n], "llm"
    planned = planned or heuristic_plan(query, n)
    subs = _dedupe_queries(_to_sub_queries(planned, 0, kinds))
    return {
        "iteration": 0,
        "sub_queries": subs,
        "pending_queries": subs,
        "trace": [f"plan ({method}): " + "; ".join(s.search_query for s in subs)],
    }


def _dedupe_queries(subs: list[SubQuery], previous: list[SubQuery] = ()) -> list[SubQuery]:
    seen = {frozenset(keywords(s.search_query)) for s in previous}
    out = []
    for s in subs:
        key = frozenset(keywords(s.search_query))
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def coverage(sub_queries: list[SubQuery], corpus: list[Document]) -> dict[str, int]:
    counts = {s.id: 0 for s in sub_queries}
    for doc in corpus:
        for sq_id in doc.sub_query_ids:
            if sq_id in counts:
                counts[sq_id] += 1
    return counts


async def reflect(state: ResearchState, rt: Runtime) -> dict:
    """Self-check: did every sub-question get enough evidence? If not, search again."""
    iteration = state.get("iteration", 0) + 1
    subs = state.get("sub_queries", [])
    corpus = state.get("corpus", [])
    counts = coverage(subs, corpus)
    current = {s.id for s in state.get("pending_queries", [])}
    gaps = [s for s in subs if s.id in current and counts[s.id] < rt.settings.min_docs_per_sub_query]

    if not gaps or iteration >= rt.settings.max_iterations or not rt.sources:
        why = "coverage ok" if not gaps else "iteration budget reached"
        return {"iteration": iteration, "pending_queries": [], "trace": [f"reflect: {why}"]}

    kinds = {s.kind for s in rt.sources}
    n = min(3, len(gaps) + 1)
    planned: list[PlannedQuery] | None = None
    method = "heuristic"
    if rt.llm:
        covered = "\n".join(f"- {d.title} ({d.outlet})" for d in corpus[:25]) or "(nothing yet)"
        gap_text = "\n".join(f"- {g.question} (searched: {g.search_query!r}, found {counts[g.id]})" for g in gaps)
        user = f"Research question: {state['query']}\n\nGaps:\n{gap_text}\n\nAlready found:\n{covered}"
        result = await rt.llm.structured(Plan, FOLLOWUP_SYSTEM.format(n=n), user)
        if result and result.sub_queries:
            planned, method = result.sub_queries[:n], "llm"
    if planned is None:
        planned = []
        core = keyword_query(state["query"]).split()
        for g in gaps[:n]:
            # Broaden: keep the three most distinctive core terms plus this facet's first extra term.
            extra = [t for t in g.search_query.split() if t not in core][:1]
            planned.append(
                PlannedQuery(question=g.question, search_query=" ".join(core[:3] + extra),
                             source_kinds=g.source_kinds, rationale=f"broadened retry of {g.search_query!r}")
            )
    new = _dedupe_queries(_to_sub_queries(planned, iteration, kinds), subs)
    return {
        "iteration": iteration,
        "sub_queries": new,
        "pending_queries": new,
        "trace": [
            f"reflect ({method}): {len(gaps)} under-covered sub-question(s); "
            f"follow-up: {'; '.join(s.search_query for s in new) or 'none'}"
        ],
    }
