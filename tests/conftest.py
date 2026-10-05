from __future__ import annotations

from datetime import datetime, timezone

import pytest

from news_intelligence.config import Settings
from news_intelligence.embeddings import CachedEmbedder, HashingEmbedder
from news_intelligence.models import Document
from news_intelligence.sources import Source, make_document
from news_intelligence.state import Runtime


def _dt(day: int) -> datetime:
    return datetime(2026, 3, day, 12, tzinfo=timezone.utc)


# A small fictional news cycle: one story syndicated, corroborated, contradicted, plus noise.
ARTICLES = [
    dict(
        title="Regulators approve Northwind Harbor bank merger",
        url="https://www.reuters.com/business/northwind-harbor-merger-approved?utm_source=x",
        content="Regulators approved the Northwind Harbor bank merger on March 3, 2026. "
        "The deal between Northwind Bank and Harbor Financial is valued at 12 billion dollars. "
        "The combined bank will close 40 branches next year.",
        published_at=_dt(3),
    ),
    dict(  # syndicated copy of the Reuters wire -> deduplicated
        title="Regulators approve Northwind Harbor bank merger",
        url="https://apnews.com/article/northwind-harbor-merger",
        content="Regulators approved the Northwind Harbor bank merger on March 3, 2026. "
        "The deal between Northwind Bank and Harbor Financial is valued at 12 billion dollars.",
        published_at=_dt(3),
    ),
    dict(  # independent corroboration
        title="Northwind and Harbor merger gets green light from regulators",
        url="https://www.bbc.com/news/business-northwind-harbor",
        content="Regulators approved the Northwind Harbor merger on March 3, 2026, the two banks said. "
        "The deal between Northwind Bank and Harbor Financial is valued at 12 billion dollars. "
        "Analysts expect the combined lender to become the fourth-largest retail bank by deposits.",
        published_at=_dt(4),
    ),
    dict(  # contradicts the deal value
        title="Insiders question the Northwind Harbor merger price",
        url="https://medium.com/@insider/northwind-harbor-merger-price",
        content="The deal between Northwind Bank and Harbor Financial is valued at 9 billion dollars, "
        "according to people familiar with the matter.",
        published_at=_dt(5),
    ),
    dict(  # stakeholder reaction
        title="Unions criticise Northwind Harbor merger over branch closures",
        url="https://www.theguardian.com/business/northwind-harbor-unions",
        content="Bank unions criticised the Northwind Harbor merger on March 6, 2026, warning that branch "
        "closures would cost hundreds of jobs. Union leaders called for a review of the merger.",
        published_at=_dt(6),
    ),
    dict(  # off-topic noise
        title="Local bakery wins award for sourdough bread",
        url="https://example-town-news.com/bakery-award",
        content="A family bakery won the regional sourdough competition for the third year running.",
        published_at=_dt(2),
    ),
]


class FakeSource(Source):
    def __init__(self, name="fake_news", kind="news", articles=ARTICLES, fail=False):
        self.name, self.kind, self.articles, self.fail = name, kind, articles, fail
        self.queries: list[str] = []

    async def search(self, query: str, k: int) -> list[Document]:
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("upstream unavailable")
        return [make_document(provider=self.name, source_type=self.kind, **a) for a in self.articles[:k]]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(llm_provider="none", vector_backend="local", vector_db_path=str(tmp_path / "vdb"),
                    results_per_source=10, max_sub_queries=3)


@pytest.fixture
def make_runtime(settings):
    def factory(sources=None, llm=None, vector_store=None, **overrides) -> Runtime:
        s = settings
        for key, value in overrides.items():
            setattr(s, key, value)
        return Runtime(settings=s, sources=sources if sources is not None else [FakeSource()],
                       embedder=CachedEmbedder(HashingEmbedder()), llm=llm, vector_store=vector_store)

    return factory
