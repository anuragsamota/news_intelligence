"""Semantic deduplication and source ranking."""

from __future__ import annotations

from datetime import datetime, timezone

from ..credibility import credibility
from ..embeddings import similarity
from ..models import Document
from ..state import ResearchState, Runtime
from ..text import canonical_url, truncate

WEIGHTS = {"relevance": 0.45, "credibility": 0.25, "recency": 0.15, "corroboration": 0.15}


def _merge(group: list[Document]) -> Document:
    """Collapse a cluster of near-duplicates into its best representative."""
    best = max(group, key=lambda d: (credibility(d.domain), len(d.content or d.snippet), d.relevance))
    others = [d for d in group if d is not best]
    sub_ids = list(dict.fromkeys(i for d in group for i in d.sub_query_ids))
    dup_urls = list(dict.fromkeys(u for d in others for u in [d.url, *d.duplicate_urls] if u and u != best.url))
    domains = list(
        dict.fromkeys(x for d in group for x in [d.domain, *d.also_reported_by] if x and x != best.domain)
    )
    published = [d.published_at for d in group if d.published_at]
    return best.model_copy(
        update={
            "relevance": max(d.relevance for d in group),
            "sub_query_ids": sub_ids,
            "duplicate_urls": dup_urls,
            "also_reported_by": domains,
            # Earliest sighting is the best estimate of when the story broke.
            "published_at": min(published) if published else None,
            "content": best.content or max((d.content for d in group), key=len),
        }
    )


def deduplicate(docs: list[Document], rt: Runtime) -> list[Document]:
    # 1) exact: same canonical URL (same article found by several workers)
    by_url: dict[str, list[Document]] = {}
    for doc in docs:
        by_url.setdefault(canonical_url(doc.url) if doc.url else doc.id, []).append(doc)
    unique = [_merge(g) if len(g) > 1 else g[0] for g in by_url.values()]
    if len(unique) < 2:
        return unique

    # 2) semantic: syndicated copies / rewrites of the same story
    #    Only near-verbatim copies are merged: independent reports of the same event stay separate
    #    so they can corroborate each other later.
    bodies = [f"{d.title}. {truncate(d.content or d.snippet, 300)}" for d in unique]
    titles = [d.title for d in unique]
    threshold = rt.settings.dedup_threshold
    same = (similarity(rt.embedder, bodies, bodies) >= threshold) | (
        similarity(rt.embedder, titles, titles) >= max(threshold, 0.95)
    )
    order = sorted(range(len(unique)), key=lambda i: -unique[i].relevance)
    assigned: dict[int, int] = {}
    clusters: list[list[int]] = []
    for i in order:
        if i in assigned:
            continue
        cluster = [i]
        assigned[i] = len(clusters)
        for j in order:
            if j not in assigned and same[i, j] and _same_story(unique[i], unique[j]):
                cluster.append(j)
                assigned[j] = len(clusters)
        clusters.append(cluster)
    return [_merge([unique[i] for i in c]) if len(c) > 1 else unique[c[0]] for c in clusters]


def _same_story(a: Document, b: Document) -> bool:
    """Guard against merging two different events that are worded alike (e.g. follow-ups)."""
    if a.published_at and b.published_at and abs((a.published_at - b.published_at).days) > 3:
        return False
    return True


async def dedupe(state: ResearchState, rt: Runtime) -> dict:
    docs = state.get("documents", [])
    corpus = deduplicate(docs, rt)
    return {"corpus": corpus, "trace": [f"dedupe: {len(docs)} graded hits → {len(corpus)} unique stories"]}


def recency_score(published: datetime | None, half_life_days: float, now: datetime | None = None) -> float:
    if published is None:
        return 0.3
    now = now or datetime.now(timezone.utc)
    age_days = max(0.0, (now - published).total_seconds() / 86400)
    return float(0.5 ** (age_days / half_life_days))


def rank_documents(corpus: list[Document], rt: Runtime, now: datetime | None = None) -> list[Document]:
    ranked = []
    for doc in corpus:
        cred = credibility(doc.domain) if doc.source_type != "vector" else max(0.6, credibility(doc.domain))
        independent = {d for d in doc.also_reported_by if d != doc.domain}
        corr = min(1.0, len(independent) / 3)
        rec = recency_score(doc.published_at, rt.settings.recency_half_life_days, now)
        # Small boost for docs that answer several sub-questions at once.
        breadth = min(0.05, 0.02 * (len(doc.sub_query_ids) - 1))
        score = (
            WEIGHTS["relevance"] * doc.relevance
            + WEIGHTS["credibility"] * cred
            + WEIGHTS["recency"] * rec
            + WEIGHTS["corroboration"] * corr
            + breadth
        )
        ranked.append(
            doc.model_copy(update={"credibility": cred, "recency": rec, "corroboration": corr, "score": round(score, 4)})
        )
    ranked.sort(key=lambda d: d.score, reverse=True)
    return ranked


async def rank(state: ResearchState, rt: Runtime) -> dict:
    ranked = rank_documents(state.get("corpus", []), rt)
    top = ", ".join(f"{d.outlet} ({d.score:.2f})" for d in ranked[:3])
    return {"ranked": ranked, "trace": [f"rank: {len(ranked)} stories; top: {top or '-'}"]}

