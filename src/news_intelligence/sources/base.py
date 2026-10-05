from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from ..models import Document, SourceKind
from ..text import canonical_url, domain_of, stable_id, strip_html


class Source(ABC):
    """A retriever the research workers can query."""

    name: str
    kind: SourceKind

    @abstractmethod
    async def search(self, query: str, k: int) -> list[Document]: ...


def make_document(
    *,
    title: str,
    url: str,
    provider: str,
    source_type: SourceKind,
    outlet: str | None = None,
    domain: str | None = None,
    snippet: str = "",
    content: str = "",
    published_at: datetime | None = None,
) -> Document:
    url = url or ""
    domain = (domain or domain_of(url) or "unknown").removeprefix("www.")
    return Document(
        id=stable_id(canonical_url(url) if url else title),
        title=strip_html(title) or "(untitled)",
        url=url,
        outlet=outlet or domain,
        domain=domain,
        source_type=source_type,
        provider=provider,
        snippet=strip_html(snippet),
        content=strip_html(content),
        published_at=published_at,
    )
