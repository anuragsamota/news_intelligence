"""Automated timeline from dated claims and story publication dates."""

from __future__ import annotations

from ..embeddings import similarity
from ..models import TimelineEvent
from ..state import ResearchState, Runtime
from ..text import truncate

MAX_EVENTS = 25


def build_events(state: ResearchState, rt: Runtime) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    claimed_docs: set[str] = set()
    # 1) Claims whose text / extraction carries an explicit event date.
    for claim in state.get("claims", []):
        if claim.date:
            events.append(TimelineEvent(date=claim.date, event=claim.text, doc_ids=claim.doc_ids, explicit=True))
            claimed_docs.update(claim.doc_ids)
    # 2) Top stories without a dated claim fall back to their (earliest) publication date.
    for doc in state.get("ranked", [])[: rt.settings.max_report_sources]:
        if doc.id not in claimed_docs and doc.published_at:
            events.append(
                TimelineEvent(date=doc.published_at.date(), event=truncate(doc.title, 200), doc_ids=[doc.id], explicit=False)
            )
    return merge_events(events, rt)


def merge_events(events: list[TimelineEvent], rt: Runtime) -> list[TimelineEvent]:
    """Merge same-day events that describe the same thing; prefer explicitly dated ones."""
    if not events:
        return []
    events = sorted(events, key=lambda e: (e.date, not e.explicit))
    texts = [e.event for e in events]
    sims = similarity(rt.embedder, texts, texts)
    merged: list[tuple[int, TimelineEvent]] = []
    for i, ev in enumerate(events):
        for k, (j, kept) in enumerate(merged):
            if kept.date == ev.date and sims[i, j] >= 0.5:
                doc_ids = list(dict.fromkeys(kept.doc_ids + ev.doc_ids))
                merged[k] = (j, kept.model_copy(update={"doc_ids": doc_ids}))
                break
        else:
            merged.append((i, ev))
    out = [e for _, e in merged]
    if len(out) > MAX_EVENTS:  # keep the best-supported events, still in date order
        keep = sorted(out, key=lambda e: (e.explicit, len(e.doc_ids)), reverse=True)[:MAX_EVENTS]
        out = sorted(keep, key=lambda e: e.date)
    return out


async def build_timeline(state: ResearchState, rt: Runtime) -> dict:
    events = build_events(state, rt)
    explicit = sum(e.explicit for e in events)
    return {"timeline": events, "trace": [f"timeline: {len(events)} events ({explicit} with explicit dates)"]}
