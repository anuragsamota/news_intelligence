from dataclasses import replace

import pytest
from conftest import FakeSource
from fake_ollama import FakeOllama

from news_intelligence.config import Settings
from news_intelligence.graph import build_runtime, research, resolve_backend
from news_intelligence.llm import OllamaStatus

QUERY = "Was the Northwind Harbor bank merger approved and what is the deal worth?"


@pytest.fixture
def ollama():
    with FakeOllama() as server:
        yield server


def _settings(url: str, **kw) -> Settings:
    base = dict(ollama_base_url=url, vector_backend="none", max_sub_queries=3, embedding_floor=0.0,
                enable_gdelt=False, enable_google_news=False)
    return Settings(**{**base, **kw})


async def test_full_pipeline_through_ollama(ollama):
    rt = build_runtime(_settings(ollama.url))
    assert rt.notes == []
    assert rt.llm.name == "ollama:qwen2.5:7b"
    assert rt.embedder.name == "ollama-nomic-embed-text"
    rt = replace(rt, sources=[FakeSource()])

    report = await research(QUERY, rt)

    # The LLM planned the searches and wrote + verified the report.
    assert [sq.search_query for sq in report.sub_queries[:2]] == [
        "northwind harbor merger approval", "northwind harbor deal value"]
    assert report.stats["writer"] == "llm" and report.stats["verifier"] == "llm"
    assert report.title.startswith("Northwind")
    assert all(f.verified for f in report.key_findings)

    # LLM grading dropped the off-topic story; LLM judge found the price conflict.
    assert not any("bakery" in c.url for c in report.citations)
    assert report.contradictions and all(c.method == "llm" for c in report.contradictions)
    assert any(c.date and c.date.isoformat() == "2026-03-03" for c in report.claims)

    # Citation validation: [1, 2] normalised, hallucinated [99] removed and reported.
    assert "[1][2]" in report.summary and "[99]" not in report.summary
    assert any("[99]" in w for w in report.warnings)

    # Requests were real Ollama API calls with schema-constrained output and our options.
    chats = ollama.chat_requests()
    assert chats and all(isinstance(c["format"], dict) for c in chats)
    assert {c["model"] for c in chats} == {"qwen2.5:7b"}
    assert all(c["options"]["num_ctx"] == 16384 and c["options"]["temperature"] == 0 for c in chats)
    assert any(p == "/api/embed" for p, _ in ollama.requests)


def test_unreachable_ollama_falls_back_to_offline_mode():
    settings, notes = resolve_backend(Settings(), OllamaStatus(False, error="ConnectError"))
    assert settings.llm_provider == "none" and settings.embedding_provider == "hashing"
    assert "not reachable" in notes[0]


def test_missing_models_tell_user_what_to_pull():
    status = OllamaStatus(True, {"nomic-embed-text:latest"})
    settings, notes = resolve_backend(Settings(llm_model="llama3.1:8b"), status)
    assert settings.llm_provider == "none" and settings.embedding_provider == "ollama"
    assert "ollama pull llama3.1:8b" in notes[0]


async def test_fallback_note_appears_in_report(ollama):
    ollama.models = ["nomic-embed-text:latest"]  # chat model not pulled
    rt = replace(build_runtime(_settings(ollama.url)), sources=[FakeSource()])
    assert rt.llm is None
    report = await research(QUERY, rt)
    assert any("ollama pull qwen2.5:7b" in w for w in report.warnings)
    assert report.stats["writer"] == "heuristic"


def test_ollama_host_env_is_honoured(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0")
    assert Settings.from_env().ollama_base_url == "http://localhost:11434"
    monkeypatch.setenv("OLLAMA_HOST", "gpu-box:11500")
    assert Settings.from_env().ollama_base_url == "http://gpu-box:11500"
    monkeypatch.setenv("NI_OLLAMA_BASE_URL", "https://ollama.com")
    assert Settings.from_env().ollama_base_url == "https://ollama.com"
