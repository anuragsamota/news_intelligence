"""News retrievers: GDELT, Google News RSS (both keyless) and NewsAPI.org."""

from __future__ import annotations

import feedparser
import httpx

from ..models import Document
from ..text import domain_of, parse_datetime
from .base import Source, make_document

USER_AGENT = "news-intelligence/0.1 (+https://github.com/anuragsamota/study_planner)"


class GDELTSource(Source):
    """GDELT DOC 2.0 API - global news index, no key required (titles only)."""

    name = "gdelt"
    kind = "news"
    URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(self, timeout: float = 20.0, timespan: str = "3months", language: str = "english"):
        self.timeout = timeout
        self.timespan = timespan
        self.language = language

    async def search(self, query: str, k: int) -> list[Document]:
        q = query if not self.language else f"{query} sourcelang:{self.language}"
        params = {
            "query": q,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": str(max(1, min(k, 250))),
            "sort": "HybridRel",
            "timespan": self.timespan,
        }
        async with httpx.AsyncClient(timeout=self.timeout, headers={"User-Agent": USER_AGENT}) as client:
            resp = await client.get(self.URL, params=params)
            resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError:  # GDELT answers query errors with plain text
            raise RuntimeError(resp.text.strip()[:200] or "invalid GDELT response") from None
        return [
            make_document(
                title=a.get("title", ""),
                url=a.get("url", ""),
                domain=a.get("domain"),
                provider=self.name,
                source_type="news",
                published_at=parse_datetime(a.get("seendate")),
            )
            for a in payload.get("articles", [])[:k]
        ]


class GoogleNewsRSSSource(Source):
    """Google News search feed - no key required."""

    name = "google_news"
    kind = "news"
    URL = "https://news.google.com/rss/search"

    def __init__(self, timeout: float = 20.0, language: str = "en", country: str = "US"):
        self.timeout = timeout
        self.language = language
        self.country = country

    async def search(self, query: str, k: int) -> list[Document]:
        params = {
            "q": query,
            "hl": f"{self.language}-{self.country}",
            "gl": self.country,
            "ceid": f"{self.country}:{self.language}",
        }
        async with httpx.AsyncClient(
            timeout=self.timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True
        ) as client:
            resp = await client.get(self.URL, params=params)
            resp.raise_for_status()
        return parse_google_news_feed(resp.text, k, self.name)


def parse_google_news_feed(xml: str, k: int, provider: str = "google_news") -> list[Document]:
    feed = feedparser.parse(xml)
    docs = []
    for entry in feed.entries[:k]:
        source = entry.get("source") or {}
        outlet = source.get("title")
        title = entry.get("title", "")
        if outlet and title.endswith(f" - {outlet}"):
            title = title[: -len(outlet) - 3]
        docs.append(
            make_document(
                title=title,
                url=entry.get("link", ""),
                outlet=outlet,
                # The entry link is a news.google.com redirect; the publisher is in <source>.
                domain=domain_of(source.get("href", "")) or None,
                provider=provider,
                source_type="news",
                snippet=entry.get("summary", ""),
                published_at=parse_datetime(entry.get("published_parsed") or entry.get("published")),
            )
        )
    return docs


class NewsAPISource(Source):
    """NewsAPI.org /v2/everything - needs NEWSAPI_KEY, returns descriptions + content."""

    name = "newsapi"
    kind = "news"
    URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str, timeout: float = 20.0, language: str = "en"):
        self.api_key = api_key
        self.timeout = timeout
        self.language = language

    async def search(self, query: str, k: int) -> list[Document]:
        params = {"q": query, "pageSize": str(min(k, 100)), "sortBy": "relevancy", "language": self.language}
        async with httpx.AsyncClient(timeout=self.timeout, headers={"X-Api-Key": self.api_key}) as client:
            resp = await client.get(self.URL, params=params)
            resp.raise_for_status()
        return [
            make_document(
                title=a.get("title") or "",
                url=a.get("url") or "",
                outlet=(a.get("source") or {}).get("name"),
                provider=self.name,
                source_type="news",
                snippet=a.get("description") or "",
                # NewsAPI truncates content with "[+1234 chars]".
                content=(a.get("content") or "").split(" [+")[0],
                published_at=parse_datetime(a.get("publishedAt")),
            )
            for a in resp.json().get("articles", [])
            if a.get("title") and a.get("title") != "[Removed]"
        ]
