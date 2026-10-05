"""Domain models shared by every node of the research graph."""

from __future__ import annotations

from datetime import date as Date, datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

SourceKind = Literal["news", "web", "vector"]
ClaimStatus = Literal["corroborated", "single_source", "contested"]


class SubQuery(BaseModel):
    id: str
    question: str
    search_query: str
    source_kinds: list[SourceKind] = Field(default_factory=lambda: ["news", "web", "vector"])
    rationale: str = ""
    round: int = 0


class Document(BaseModel):
    id: str
    title: str
    url: str
    outlet: str  # display name, e.g. "Reuters"
    domain: str  # publisher domain, used for credibility + independence
    source_type: SourceKind
    provider: str  # retriever that found it, e.g. "gdelt"
    snippet: str = ""
    content: str = ""
    published_at: datetime | None = None
    sub_query_ids: list[str] = Field(default_factory=list)

    # Filled in by grading / dedup / ranking.
    relevance: float = 0.0
    relevance_reason: str = ""
    duplicate_urls: list[str] = Field(default_factory=list)
    also_reported_by: list[str] = Field(default_factory=list)  # other domains
    credibility: float = 0.5
    recency: float = 0.5
    corroboration: float = 0.0
    score: float = 0.0

    @property
    def text(self) -> str:
        body = self.content or self.snippet
        return f"{self.title}. {body}" if body else self.title

    @property
    def published_date(self) -> Date | None:
        return self.published_at.date() if self.published_at else None


class RetrievalRecord(BaseModel):
    sub_query_id: str
    source: str
    query: str
    found: int = 0
    kept: int = 0
    error: str | None = None


class Claim(BaseModel):
    id: str
    text: str
    doc_ids: list[str]
    date: Date | None = None
    support_domains: list[str] = Field(default_factory=list)
    status: ClaimStatus = "single_source"
    confidence: float = 0.0


class Contradiction(BaseModel):
    claim_a: str
    claim_b: str
    explanation: str
    method: Literal["llm", "heuristic"] = "heuristic"


class TimelineEvent(BaseModel):
    date: Date
    event: str
    doc_ids: list[str]
    explicit: bool = True  # date stated in the text vs. publication date


class Citation(BaseModel):
    n: int
    doc_id: str
    title: str
    url: str
    outlet: str
    published_at: datetime | None = None
    credibility: float = 0.5
    score: float = 0.0


class Finding(BaseModel):
    text: str
    citations: list[int] = Field(default_factory=list)
    status: ClaimStatus = "single_source"
    verified: bool = False
    verification_note: str = ""


class Section(BaseModel):
    heading: str
    body: str


class Report(BaseModel):
    query: str
    title: str
    summary: str
    key_findings: list[Finding] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    sub_queries: list[SubQuery] = Field(default_factory=list)
    retrieval_log: list[RetrievalRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_markdown(self) -> str:
        from .render import render_markdown

        return render_markdown(self)
