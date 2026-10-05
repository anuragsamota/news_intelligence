"""Markdown rendering of a :class:`~news_intelligence.models.Report`."""

from __future__ import annotations

from .models import Report

STATUS_LABEL = {
    "corroborated": "corroborated",
    "single_source": "single source",
    "contested": "**contested**",
}


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(r: Report) -> str:
    n_of = {c.doc_id: c.n for c in r.citations}
    claim_by_id = {c.id: c for c in r.claims}

    def cites(doc_ids: list[str]) -> str:
        return "".join(f"[{n}]" for n in sorted({n_of[d] for d in doc_ids if d in n_of}))

    s = r.stats
    out = [
        f"# {r.title}",
        "",
        f"> **Question:** {r.query}  ",
        f"> Generated {r.generated_at:%Y-%m-%d %H:%M} UTC · {len(r.citations)} sources · "
        f"{s.get('outlets', 0)} outlets · {s.get('retrieval_rounds', 1)} retrieval round(s) · "
        f"grounding {s.get('grounding', 0):.0%} · findings verified {s.get('verified_findings', '-')}",
        "",
        "## Summary",
        "",
        r.summary,
        "",
    ]
    if r.key_findings:
        out += ["## Key findings", ""]
        for f in r.key_findings:
            refs = "".join(f"[{n}]" for n in f.citations)
            check = "verified" if f.verified else "unverified"
            out.append(f"- {f.text} {refs} — _{STATUS_LABEL[f.status]}, {check}_")
        out.append("")
    for sec in r.sections:
        out += [f"## {sec.heading}", "", sec.body, ""]

    if r.timeline:
        out += ["## Timeline", "", "| Date | Event | Sources |", "|---|---|---|"]
        for e in r.timeline:
            date = e.date.isoformat() + ("" if e.explicit else " *")
            out.append(f"| {date} | {_cell(e.event)} | {cites(e.doc_ids)} |")
        out += ["", "_\\* publication date; no explicit event date in the text._", ""]

    if r.contradictions:
        out += ["## Disputed claims", ""]
        for c in r.contradictions:
            a, b = claim_by_id.get(c.claim_a), claim_by_id.get(c.claim_b)
            if a and b:
                out.append(f"- {a.text} {cites(a.doc_ids)}  \n  **vs.** {b.text} {cites(b.doc_ids)}  \n  _{c.explanation}_")
        out.append("")

    if r.claims:
        out += ["## Evidence verification", "", "| Claim | Status | Independent outlets | Confidence | Sources |",
                "|---|---|---|---|---|"]
        for c in sorted(r.claims, key=lambda c: (-len(c.support_domains), -c.confidence))[:30]:
            out.append(
                f"| {_cell(c.text)} | {STATUS_LABEL[c.status]} | {len(c.support_domains)} | "
                f"{c.confidence:.2f} | {cites(c.doc_ids)} |"
            )
        out.append("")
        checks = [f for f in r.key_findings if f.verification_note]
        if checks:
            out += ["**Finding checks**", ""]
            out += [f"{i}. {'✔' if f.verified else '✘'} {f.verification_note}" for i, f in enumerate(checks, 1)]
            out.append("")

    out += ["## Sources", ""]
    for c in r.citations:
        when = c.published_at.date().isoformat() if c.published_at else "n.d."
        link = f"[{c.title}]({c.url})" if c.url else c.title
        out.append(f"{c.n}. {link} — {c.outlet}, {when} · credibility {c.credibility:.2f} · rank score {c.score:.2f}")
    out.append("")

    out += ["## Research trace", ""]
    log_by_sq: dict[str, list] = {}
    for rec in r.retrieval_log:
        log_by_sq.setdefault(rec.sub_query_id, []).append(rec)
    for sq in r.sub_queries:
        hits = ", ".join(
            f"{rec.source} {'error' if rec.error else f'{rec.kept}/{rec.found}'}" for rec in log_by_sq.get(sq.id, [])
        )
        out.append(f"- _(round {sq.round})_ {sq.question} — `{sq.search_query}` → {hits or 'no sources'}")
    out.append(
        f"\nPipeline: writer={s.get('writer')}, verifier={s.get('verifier')}, "
        f"llm={s.get('llm') or 'none'}, embeddings={s.get('embeddings', '-')}"
    )
    if r.warnings:
        out += ["", "## Warnings", ""] + [f"- {w}" for w in r.warnings]
    return "\n".join(out).rstrip() + "\n"
