"""Claim extraction, cross-source corroboration and contradiction detection."""

from __future__ import annotations

import asyncio
import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from ..embeddings import similarity
from ..models import Claim, Contradiction, Document
from ..state import ResearchState, Runtime
from ..text import extract_dates, numbers, split_sentences, stable_id, tokenize, truncate

# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------


class ExtractedClaim(BaseModel):
    text: str = Field(description="One atomic, self-contained factual statement (who/what/when)")
    date: str | None = Field(None, description="ISO date (YYYY-MM-DD) the event happened, if stated")


class ExtractedClaims(BaseModel):
    claims: list[ExtractedClaim]


CLAIMS_SYSTEM = """You extract verifiable factual claims from a news article for fact-checking.
Return up to {n} atomic claims that are relevant to the research question. Each claim must be a
single self-contained sentence (resolve pronouns, keep numbers, names and dates exact) stated by the
article itself. Skip opinions, speculation and boilerplate. Give the event date if the text states it."""

_CLAIM_HINT = re.compile(r"\d|%|\b(announc|said|confirm|report|approv|reject|launch|kill|elect|rule|sign|"
                         r"rais|cut|ban|agree|accus|deni|arrest|sue|fine|win|won|lost|increas|decreas|rose|fell)",
                         re.I)


def heuristic_claims(doc: Document, n: int) -> list[ExtractedClaim]:
    sentences = split_sentences(doc.content or doc.snippet) or [doc.title]
    scored = []
    for i, s in enumerate(sentences):
        words = len(s.split())
        if words < 6 or words > 60 or s.endswith("?"):
            continue
        scored.append((-(bool(_CLAIM_HINT.search(s)) * 2 + (i == 0)), i, s))
    picked = [s for _, _, s in sorted(scored)[:n]] or [doc.title]
    return [ExtractedClaim(text=s) for s in picked]


def _parse_date(raw: str | None, text: str) -> date | None:
    if raw:
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            pass
    found = extract_dates(text)
    return found[0] if found else None


async def _claims_for(doc: Document, query: str, rt: Runtime, n: int) -> tuple[list[ExtractedClaim], bool]:
    if rt.llm and (doc.content or doc.snippet):
        user = (
            f"Research question: {query}\n\nArticle: {doc.title} — {doc.outlet}, "
            f"published {doc.published_date or 'unknown'}\n\n{truncate(doc.content or doc.snippet, 4000)}"
        )
        result = await rt.llm.structured(ExtractedClaims, CLAIMS_SYSTEM.format(n=n), user)
        if result and result.claims:
            return result.claims[:n], True
    return heuristic_claims(doc, n), False


# ---------------------------------------------------------------------------
# Heuristic conflict detection (also used to stop contradicting claims merging)
# ---------------------------------------------------------------------------

ANTONYMS = [
    ("rose", "fell"), ("rise", "fall"), ("increase", "decrease"), ("increased", "decreased"),
    ("up", "down"), ("gain", "loss"), ("gained", "lost"), ("won", "lost"), ("win", "lose"),
    ("approved", "rejected"), ("approve", "reject"), ("passed", "failed"), ("confirmed", "denied"),
    ("confirms", "denies"), ("accepted", "rejected"), ("higher", "lower"), ("more", "fewer"),
    ("guilty", "innocent"), ("acquitted", "convicted"), ("open", "closed"), ("allowed", "banned"),
    ("legal", "illegal"), ("support", "oppose"), ("supports", "opposes"), ("true", "false"),
    ("resigned", "remains"), ("alive", "dead"), ("expanded", "shrank"), ("raised", "cut"),
]
_ANTONYM_MAP = {a: b for a, b in ANTONYMS} | {b: a for a, b in ANTONYMS}
NEGATIONS = {"not", "no", "never", "denied", "denies", "deny", "false", "didn't", "won't", "isn't", "wasn't"}


def heuristic_conflict(a: str, b: str, similarity: float) -> str | None:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if similarity >= 0.35:
        for word in ta:
            opposite = _ANTONYM_MAP.get(word)
            if opposite and opposite in tb and word not in tb:
                return f"opposite outcomes reported ('{word}' vs '{opposite}')"
    if similarity >= 0.5 and bool(ta & NEGATIONS) != bool(tb & NEGATIONS):
        return "one source negates what the other asserts"
    na, nb = numbers(a), numbers(b)
    if similarity >= 0.5 and na and nb and not (na & nb):
        return f"different figures reported ({', '.join(sorted(na))} vs {', '.join(sorted(nb))})"
    return None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def extract_claims(state: ResearchState, rt: Runtime) -> dict:
    """Extract claims from the top-ranked stories and cluster them across sources."""
    docs = state.get("ranked", [])[: rt.settings.max_docs_for_claims]
    results = await asyncio.gather(*(_claims_for(d, state["query"], rt, 4) for d in docs))
    raw: list[tuple[ExtractedClaim, Document]] = [(c, d) for (claims, _), d in zip(results, docs) for c in claims]
    used_llm = sum(1 for _, ok in results if ok)
    claims = cluster_claims(raw, rt)
    corroborated = sum(c.status == "corroborated" for c in claims)
    return {
        "claims": claims,
        "trace": [
            f"claims: {len(raw)} extracted from {len(docs)} stories ({used_llm} via LLM) → "
            f"{len(claims)} distinct, {corroborated} corroborated by ≥2 independent outlets"
        ],
    }


def cluster_claims(raw: list[tuple[ExtractedClaim, Document]], rt: Runtime) -> list[Claim]:
    if not raw:
        return []
    texts = [c.text for c, _ in raw]
    sims = similarity(rt.embedder, texts, texts)
    threshold = rt.settings.claim_cluster_threshold
    order = sorted(range(len(raw)), key=lambda i: -raw[i][1].score)  # best source first
    assigned: set[int] = set()
    claims: list[Claim] = []
    for i in order:
        if i in assigned:
            continue
        members = [i]
        assigned.add(i)
        for j in order:
            if j in assigned or sims[i, j] < threshold:
                continue
            if heuristic_conflict(texts[i], texts[j], float(sims[i, j])):
                continue  # keep contradicting statements apart
            members.append(j)
            assigned.add(j)
        docs = list({raw[m][1].id: raw[m][1] for m in members}.values())
        domains = list(dict.fromkeys(d.domain for d in docs))
        claim_date = next(
            (dt for m in members if (dt := _parse_date(raw[m][0].date, raw[m][0].text))), None
        )
        lead = raw[i][1]
        claims.append(
            Claim(
                id=stable_id(texts[i], lead.id, size=10),
                text=texts[i],
                doc_ids=[d.id for d in docs],
                date=claim_date,
                support_domains=domains,
                status="corroborated" if len(domains) >= 2 else "single_source",
                # Half source quality, half independent corroboration (saturating at 3 outlets).
                confidence=round(0.5 * max(d.credibility for d in docs) + 0.5 * min(len(domains), 3) / 3, 3),
            )
        )
    return claims


class PairJudgement(BaseModel):
    pair: int
    relation: Literal["contradiction", "consistent", "unrelated"]
    explanation: str = Field(description="One sentence; for contradictions say exactly what differs")


class PairJudgements(BaseModel):
    judgements: list[PairJudgement]


CONTRADICTION_SYSTEM = """You check pairs of claims taken from different news outlets.
For each numbered pair decide whether the claims CONTRADICT each other (they cannot both be true
about the same event: different figures, outcomes, dates, attributions or a denial), are CONSISTENT
(agree or are compatible), or are UNRELATED (about different things). Differences in detail or
emphasis that can both be true are 'consistent'. Judge every pair."""


async def detect_contradictions(state: ResearchState, rt: Runtime) -> dict:
    claims = state.get("claims", [])
    candidates = candidate_pairs(claims, rt)
    found: list[Contradiction] = []
    method = "heuristic"
    if rt.llm and candidates:
        listing = "\n".join(
            f"[{k}] A: {claims[i].text}\n    B: {claims[j].text}" for k, (i, j, _) in enumerate(candidates)
        )
        result = await rt.llm.structured(PairJudgements, CONTRADICTION_SYSTEM, listing)
        if result:
            method = "llm"
            for jd in result.judgements:
                if jd.relation == "contradiction" and 0 <= jd.pair < len(candidates):
                    i, j, _ = candidates[jd.pair]
                    found.append(Contradiction(claim_a=claims[i].id, claim_b=claims[j].id,
                                               explanation=jd.explanation, method="llm"))
    if method == "heuristic":
        for i, j, sim in candidates:
            why = heuristic_conflict(claims[i].text, claims[j].text, sim)
            if why:
                found.append(Contradiction(claim_a=claims[i].id, claim_b=claims[j].id, explanation=why))

    contested = {c.claim_a for c in found} | {c.claim_b for c in found}
    claims = [c.model_copy(update={"status": "contested"}) if c.id in contested else c for c in claims]
    return {
        "claims": claims,
        "contradictions": found,
        "trace": [f"contradictions ({method}): {len(candidates)} candidate pairs → {len(found)} conflicts"],
    }


def candidate_pairs(claims: list[Claim], rt: Runtime) -> list[tuple[int, int, float]]:
    """Topically related claims backed by *different* documents."""
    if len(claims) < 2:
        return []
    texts = [c.text for c in claims]
    sims = similarity(rt.embedder, texts, texts)
    pairs = []
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            if set(claims[i].doc_ids) & set(claims[j].doc_ids):
                continue
            if sims[i, j] >= 0.3:
                pairs.append((i, j, float(sims[i, j])))
    pairs.sort(key=lambda p: -p[2])
    return pairs[: rt.settings.max_contradiction_pairs]
