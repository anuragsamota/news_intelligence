"""General web search via Tavily (needs TAVILY_API_KEY)."""

from __future__ import annotations

import httpx

from ..models import Document
from ..text import parse_datetime
from .base import Source, make_document


class TavilySource(Source):
    name = "tavily"
    kind = "web"
    URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, timeout: float = 30.0, topic: str = "general"):
        self.api_key = api_key
        self.timeout = timeout
        self.topic = topic

    async def search(self, query: str, k: int) -> list[Document]:
        body = {"query": query, "max_results": min(k, 20), "search_depth": "advanced", "topic": self.topic}
        async with httpx.AsyncClient(
            timeout=self.timeout, headers={"Authorization": f"Bearer {self.api_key}"}
        ) as client:
            resp = await client.post(self.URL, json=body)
            resp.raise_for_status()
        return [
            make_document(
                title=r.get("title") or r.get("url", ""),
                url=r.get("url", ""),
                provider=self.name,
                source_type="web",
                content=r.get("content") or "",
                published_at=parse_datetime(r.get("published_date")),
            )
            for r in resp.json().get("results", [])
        ]
