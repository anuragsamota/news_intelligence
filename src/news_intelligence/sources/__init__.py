from __future__ import annotations

from ..config import Settings
from .base import Source, make_document
from .news import GDELTSource, GoogleNewsRSSSource, NewsAPISource
from .vector import VectorStore, VectorStoreSource
from .web import TavilySource

__all__ = [
    "Source",
    "make_document",
    "GDELTSource",
    "GoogleNewsRSSSource",
    "NewsAPISource",
    "TavilySource",
    "VectorStoreSource",
    "build_sources",
]

_GDELT_LANGUAGES = {"en": "english", "de": "german", "fr": "french", "es": "spanish", "it": "italian"}


def build_sources(settings: Settings, vector_store: VectorStore | None = None) -> list[Source]:
    t = settings.request_timeout
    sources: list[Source] = []
    if settings.enable_gdelt:
        sources.append(
            GDELTSource(t, settings.gdelt_timespan, _GDELT_LANGUAGES.get(settings.news_language, ""))
        )
    if settings.enable_google_news:
        sources.append(GoogleNewsRSSSource(t, settings.news_language))
    if settings.newsapi_key:
        sources.append(NewsAPISource(settings.newsapi_key, t, settings.news_language))
    if settings.tavily_api_key:
        sources.append(TavilySource(settings.tavily_api_key, t))
    if vector_store is not None:
        sources.append(VectorStoreSource(vector_store))
    return sources
