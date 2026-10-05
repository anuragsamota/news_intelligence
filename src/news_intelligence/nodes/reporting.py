"""Citation-grounded report writing, citation validation and evidence verification."""

from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel, Field

from ..embeddings import similarity
from ..models import Citation, Claim, Document, Finding, Report, Section
from ..state import ResearchState, Runtime
from ..text import split_sentences, truncate

_CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def select_citations(ranked: list[Document], claims: list[Claim], limit: int) -> list[Citation]:
    """Number the sources: every doc backing a claim, then the best of the rest, in rank order."""
    claim_docs = {d for c in claims for d in c.doc_ids}
    chosen = [d for d in ranked if d.id in claim_docs]
    chosen += [d for d in ranked if d.id not in claim_docs][: max(0, limit - len(chosen))]
    chosen.sort(key=lambda d: -d.score)
    return [
        Citation(n=i, doc_id=d.id, title=d.title, url=d.url, outlet=d.outlet,
                 published_at=d.published_at, credibility=d.credibility, score=d.score)
        for i, d in enumerate(chosen, start=1)
    ]


def cite(doc_ids: list[str], numbers: dict[str, int]) -> str:
    ns = sorted({numbers[d] for d in doc_ids if d in numbers})
    return "".join(f"[{n}]" for n in ns)


def validate_citations(text: str, valid: set[int]) -> tuple[str, list[int]]:
    """Normalise ``[1, 2]`` → ``[1][2]`` and drop references to sources that don't exist."""
    invalid: list[int] = []

    def fix(match: re.Match) -> str:
        nums = [int(x) for x in re.split(r"\s*[,;]\s*", match.group(1))]
        invalid.extend(n for n in nums if n not in valid)
        return "".join(f"[{n}]" for n in nums if n in valid)

    return _CITE.sub(fix, text), invalid


def grounding_ratio(texts: list[str]) -> tuple[float, int]:
    """Share of factual sentences carrying at least one citation."""
    total = cited = 0
    for text in texts:
        for line in text.splitlines():
            line = line.strip().lstrip("-*• ").strip()
            if not line or line.startswith("#"):
                continue
            for sentence in split_sentences(line):
                if len(sentence.split()) < 5:
                    continue
                total += 1
                cited += bool(_CITE.search(sentence))
    return (cited / total if total else 1.0), total - cited


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


class DraftFinding(BaseModel):
    text: str = Field(description="One-sentence finding, without citation markers")
    citations: list[int] = Field(description="Source numbers that directly support the finding")


class DraftSection(BaseModel):
    heading: str
    body: str = Field(description="Markdown. Every factual sentence ends with citations like [2][5]")


class ReportDraft(BaseModel):
    title: str
    summary: str = Field(description="3-5 sentence executive summary with inline citations [n]")
    key_findings: list[DraftFinding]
    sections: list[DraftSection]


WRITER_SYSTEM = """You are the writer of a citation-grounded news research report.
Rules:
- Use ONLY the numbered sources and verified claims provided. Never add outside facts.
- Every factual sentence must end with one or more citations in the form [n] referring to the
  numbered sources. Cite the sources that actually support the sentence.
- Prefer corroborated claims. When a point rests on a single source say so ("according to X").
- Where sources contradict each other, present both sides with their citations; do not pick a winner
  unless the evidence clearly favours one side, and say why.
- Be neutral, specific (names, numbers, dates) and concise. No speculation.
Write a title, a short executive summary, 4-8 key findings, and 3-6 thematic sections."""


def _evidence_brief(state: ResearchState, citations: list[Citation], numbers: dict[str, int], by_id: dict) -> str:
    sources = "\n".join(
        f"[{c.n}] {c.title} — {c.outlet}, {c.published_at.date() if c.published_at else 'n.d.'} "
        f"(credibility {c.credibility:.2f})\n     {truncate(by_id[c.doc_id].content or by_id[c.doc_id].snippet, 500)}"
        for c in citations
    )
    claims = "\n".join(
        f"- ({c.status}, {len(c.support_domains)} outlet(s)) {c.text} {cite(c.doc_ids, numbers)}"
        for c in state.get("claims", [])
    )
    claim_by_id = {c.id: c for c in state.get("claims", [])}
    conflicts = "\n".join(
        f"- {claim_by_id[x.claim_a].text} {cite(claim_by_id[x.claim_a].doc_ids, numbers)} VS "
        f"{claim_by_id[x.claim_b].text} {cite(claim_by_id[x.claim_b].doc_ids, numbers)} — {x.explanation}"
        for x in state.get("contradictions", [])
        if x.claim_a in claim_by_id and x.claim_b in claim_by_id
    )
    subs = "\n".join(f"- {s.question}" for s in state.get("sub_queries", []))
    return (
        f"Research question: {state['query']}\n\nSub-questions investigated:\n{subs}\n\n"
        f"Numbered sources:\n{sources}\n\nExtracted claims:\n{claims or '(none)'}\n\n"
        f"Contradictions:\n{conflicts or '(none detected)'}"
    )


def heuristic_draft(state: ResearchState, numbers: dict[str, int], by_id: dict) -> ReportDraft:
    query = state["query"]
    priority = {"corroborated": 0, "contested": 1, "single_source": 2}
    claims = sorted(state.get("claims", []), key=lambda c: (priority[c.status], -c.confidence))
    findings = [
        DraftFinding(text=c.text, citations=sorted({numbers[d] for d in c.doc_ids if d in numbers}))
        for c in claims[:8]
        if any(d in numbers for d in c.doc_ids)
    ]
    summary = [] if findings else ["No relevant evidence found."]
    for f in findings[:3]:
        summary.append(f"{f.text.rstrip('.')}. {''.join(f'[{n}]' for n in f.citations)}")
    if state.get("contradictions"):
        summary.append(f"Sources disagree on {len(state['contradictions'])} point(s); see the disputed claims below.")

    sections = []
    shown: set[str] = set()
    for sq in state.get("sub_queries", []):
        docs = [d for d in state.get("ranked", []) if sq.id in d.sub_query_ids and d.id in numbers
                and d.id not in shown][:4]
        if not docs:
            continue
        shown.update(d.id for d in docs)
        lines = []
        for d in docs:
            lead = split_sentences(d.content or d.snippet)
            detail = f": {truncate(lead[0], 220)}" if lead and lead[0] != d.title else ""
            when = d.published_date.isoformat() if d.published_date else "n.d."
            lines.append(f"- **{d.title}** — {d.outlet}, {when}{detail} [{numbers[d.id]}]")
        sections.append(DraftSection(heading=sq.question, body="\n".join(lines)))
    return ReportDraft(
        title=truncate(f"Research brief: {query}", 120),
        summary=" ".join(summary),
        key_findings=findings,
        sections=sections,
    )


# ---------------------------------------------------------------------------
# Evidence verification of findings
# ---------------------------------------------------------------------------


class Verification(BaseModel):
    index: int
    supported: bool
    note: str = Field(description="Quote or paraphrase the supporting evidence, or say what is missing")


class Verifications(BaseModel):
    items: list[Verification]


VERIFY_SYSTEM = """You verify that each finding of a research report is supported by the sources it
cites. For each numbered finding, read the cited source excerpts and decide if they actually support
the finding as written (numbers, names, dates and direction must match). Be strict."""


def heuristic_verify(findings: list[Finding], citations: dict[int, Citation], by_id: dict, rt: Runtime) -> None:
    for f in findings:
        passages = []
        for n in f.citations:
            doc = by_id[citations[n].doc_id]
            passages += [(n, s) for s in [doc.title, *split_sentences(doc.content or doc.snippet)]]
        if not passages:
            f.verified, f.verification_note = False, "no citation"
            continue
        sims = similarity(rt.embedder, [f.text], [p for _, p in passages])[0]
        best = int(np.argmax(sims))
        f.verified = bool(sims[best] >= 0.35)
        n, passage = passages[best]
        f.verification_note = (
            f"closest evidence [{n}]: “{truncate(passage, 160)}” (similarity {sims[best]:.2f})"
        )


async def verify_findings(findings: list[Finding], citations: dict[int, Citation], by_id: dict, rt: Runtime) -> str:
    if rt.llm and findings:
        listing = []
        for i, f in enumerate(findings):
            excerpts = "\n".join(
                f"   [{n}] {truncate(by_id[citations[n].doc_id].text, 600)}" for n in f.citations
            )
            listing.append(f"({i}) {f.text}\n{excerpts or '   (no citations)'}")
        result = await rt.llm.structured(Verifications, VERIFY_SYSTEM, "\n\n".join(listing))
        if result:
            for v in result.items:
                if 0 <= v.index < len(findings):
                    findings[v.index].verified = v.supported and bool(findings[v.index].citations)
                    findings[v.index].verification_note = v.note
            return "llm"
    heuristic_verify(findings, citations, by_id, rt)
    return "heuristic"


def finding_status(f: Finding, claims: list[Claim], citations: dict[int, Citation], by_id: dict, rt: Runtime):
    cited_docs = {citations[n].doc_id for n in f.citations}
    related = [c for c in claims if cited_docs & set(c.doc_ids)]
    if related:
        sims = similarity(rt.embedder, [f.text], [c.text for c in related])[0]
        best = int(np.argmax(sims))
        if sims[best] >= 0.4:
            return related[best].status
    domains = {by_id[d].domain for d in cited_docs}
    return "corroborated" if len(domains) >= 2 else "single_source"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def write_report(state: ResearchState, rt: Runtime) -> dict:
    ranked = state.get("ranked", [])
    by_id = {d.id: d for d in ranked}
    claims = state.get("claims", [])
    citations = select_citations(ranked, claims, rt.settings.max_report_sources)
    numbers = {c.doc_id: c.n for c in citations}
    cite_map = {c.n: c for c in citations}
    warnings: list[str] = list(rt.notes)

    draft: ReportDraft | None = None
    method = "heuristic"
    if rt.llm and citations:
        draft = await rt.llm.structured(ReportDraft, WRITER_SYSTEM, _evidence_brief(state, citations, numbers, by_id))
        method = "llm" if draft else "heuristic (LLM failed)"
    if draft is None:
        draft = heuristic_draft(state, numbers, by_id)

    valid = set(cite_map)
    summary, bad = validate_citations(draft.summary, valid)
    sections = []
    for s in draft.sections:
        body, more_bad = validate_citations(s.body, valid)
        bad += more_bad
        sections.append(Section(heading=s.heading, body=body))
    findings = []
    for f in draft.key_findings:
        nums = [n for n in dict.fromkeys(f.citations) if n in valid]
        bad += [n for n in f.citations if n not in valid]
        text = _CITE.sub("", f.text).strip()
        findings.append(Finding(text=text, citations=nums))
    if bad:
        warnings.append(f"Removed {len(bad)} citation(s) to non-existent sources: {sorted(set(bad))}")

    verify_method = await verify_findings(findings, cite_map, by_id, rt)
    for f in findings:
        f.status = finding_status(f, claims, cite_map, by_id, rt) if f.citations else "single_source"
    unverified = [f for f in findings if not f.verified]
    if unverified:
        warnings.append(f"{len(unverified)} key finding(s) could not be verified against their cited sources.")

    ratio, uncited = grounding_ratio([summary] + [s.body for s in sections])
    if uncited:
        warnings.append(f"{uncited} sentence(s) in the narrative carry no citation.")
    if not ranked:
        warnings.append("No relevant sources were found; try rephrasing the question or enabling more sources.")
    errors = [r for r in state.get("retrieval_log", []) if r.error]
    if errors:
        warnings.append(f"{len(errors)} retrieval call(s) failed: " + ", ".join(sorted({r.source for r in errors})))

    providers: dict[str, int] = {}
    for d in ranked:
        providers[d.provider.split(":")[0]] = providers.get(d.provider.split(":")[0], 0) + 1
    report = Report(
        query=state["query"],
        title=draft.title,
        summary=summary,
        key_findings=findings,
        sections=sections,
        timeline=state.get("timeline", []),
        contradictions=state.get("contradictions", []),
        claims=claims,
        citations=citations,
        sub_queries=state.get("sub_queries", []),
        retrieval_log=state.get("retrieval_log", []),
        warnings=warnings,
        stats={
            "writer": method,
            "verifier": verify_method,
            "llm": rt.llm.name if rt.llm else None,
            "embeddings": rt.embedder.name,
            "retrieval_rounds": state.get("iteration", 1),
            "documents_retrieved": sum(r.found for r in state.get("retrieval_log", [])),
            "documents_relevant": len(state.get("documents", [])),
            "unique_stories": len(ranked),
            "outlets": len({d.domain for d in ranked}),
            "by_provider": providers,
            "claims": len(claims),
            "corroborated_claims": sum(c.status == "corroborated" for c in claims),
            "contradictions": len(state.get("contradictions", [])),
            "grounding": round(ratio, 3),
            "verified_findings": f"{len(findings) - len(unverified)}/{len(findings)}",
        },
    )
    return {"report": report, "trace": [f"report ({method}): grounding {ratio:.0%}, {len(citations)} sources cited"]}


async def remember(state: ResearchState, rt: Runtime) -> dict:
    """Persist relevant articles to the vector DB so future runs can retrieve them."""
    if not (rt.vector_store and rt.settings.auto_ingest):
        return {}
    fresh = [d for d in state.get("ranked", []) if d.source_type != "vector"][: rt.settings.max_report_sources]
    added = rt.vector_store.add(fresh)
    return {"trace": [f"remember: stored {added} chunk(s) in the vector DB"]}
