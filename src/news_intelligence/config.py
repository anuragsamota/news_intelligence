"""Runtime settings, read from environment variables (prefix ``NI_``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields


def _get(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _ollama_host() -> str | None:
    host = _get("OLLAMA_HOST")
    if not host:
        return None
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    if host.startswith("http://0.0.0.0"):  # server bind address, not a client URL
        host = host.replace("0.0.0.0", "localhost", 1)
    return host if host.count(":") > 1 else f"{host}:11434"


@dataclass
class Settings:
    # --- Ollama backend ------------------------------------------------------
    # Local server by default; point at https://ollama.com (+ OLLAMA_API_KEY) for Ollama Cloud.
    ollama_base_url: str = "http://localhost:11434"
    ollama_api_key: str | None = None
    ollama_keep_alive: str = "30m"  # keep models loaded between pipeline steps

    llm_provider: str = "ollama"  # ollama | none (deterministic heuristics only)
    llm_model: str = "qwen2.5:7b"  # any Ollama chat model with good JSON output
    llm_num_ctx: int = 16384  # report writing sends ~20 source excerpts
    llm_timeout: float = 300.0
    llm_concurrency: int = 2  # match OLLAMA_NUM_PARALLEL on the server

    # --- Embeddings / vector DB ----------------------------------------------
    embedding_provider: str = "ollama"  # ollama | hashing (offline, lexical)
    embedding_model: str = "nomic-embed-text"
    # Cosine similarity of unrelated texts for this embedder. Similarities are rescaled so
    # that this maps to 0, which keeps every threshold below embedder-independent.
    # Unset = per-embedder default (hashing 0.0, ollama 0.4).
    embedding_floor: float | None = None
    vector_backend: str = "auto"  # auto | chroma | local | none
    vector_db_path: str = ".news_intelligence/vectordb"
    auto_ingest: bool = True  # store relevant articles for future runs

    # --- Sources ---------------------------------------------------------------
    newsapi_key: str | None = None
    tavily_api_key: str | None = None
    enable_gdelt: bool = True
    enable_google_news: bool = True
    gdelt_timespan: str = "3months"
    news_language: str = "en"
    results_per_source: int = 8
    request_timeout: float = 20.0

    # --- Agent behaviour -------------------------------------------------------
    max_sub_queries: int = 5
    max_iterations: int = 2  # retrieval rounds (1 = no follow-up searches)
    min_docs_per_sub_query: int = 2
    relevance_threshold: float = 0.5  # LLM grader, 0..1
    heuristic_relevance_threshold: float = 0.3
    dedup_threshold: float = 0.9  # near-verbatim (syndicated) copies only
    claim_cluster_threshold: float = 0.7
    max_docs_for_claims: int = 15
    max_contradiction_pairs: int = 30
    max_report_sources: int = 20
    recency_half_life_days: float = 7.0

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        values: dict = {}
        for f in fields(cls):
            raw = _get(f"NI_{f.name.upper()}")
            if raw is None:
                continue
            kind = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "")
            if kind.startswith("bool"):
                values[f.name] = raw.lower() in ("1", "true", "yes", "on")
            elif kind.startswith("int"):
                values[f.name] = int(raw)
            elif kind.startswith("float"):
                values[f.name] = float(raw)
            else:
                values[f.name] = raw
        # Conventional variable names are honoured too.
        values.setdefault("ollama_base_url", _ollama_host())
        values.setdefault("ollama_api_key", _get("OLLAMA_API_KEY"))
        values.setdefault("newsapi_key", _get("NEWSAPI_KEY"))
        values.setdefault("tavily_api_key", _get("TAVILY_API_KEY"))
        values = {k: v for k, v in values.items() if v is not None}
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)
