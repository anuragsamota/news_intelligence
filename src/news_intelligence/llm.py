"""Ollama backend: chat model wrapper and server health check.

Every node calls :meth:`LLM.structured`, which returns ``None`` on any failure
so the node can fall back to its deterministic heuristic. That keeps a run
alive when the model times out or returns malformed output.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TypeVar

import httpx
from pydantic import BaseModel

from .config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLM:
    def __init__(self, chat_model, name: str, concurrency: int = 2):
        self.model = chat_model
        self.name = name
        self._sem = asyncio.Semaphore(concurrency)

    async def structured(self, schema: type[T], system: str, user: str) -> T | None:
        async with self._sem:
            try:
                # Ollama constrains decoding to the JSON schema (``format=<schema>``).
                runnable = self.model.with_structured_output(schema, method="json_schema")
                result = await runnable.ainvoke([("system", system), ("human", user)])
            except Exception as exc:  # noqa: BLE001 - any backend error -> heuristic fallback
                log.warning("LLM call for %s failed: %s", schema.__name__, exc)
                return None
        if isinstance(result, dict):
            try:
                result = schema.model_validate(result)
            except Exception:  # noqa: BLE001
                return None
        return result if isinstance(result, schema) else None


def client_kwargs(settings: Settings) -> dict:
    """Timeout + auth header for the ollama client (Ollama Cloud or an auth proxy)."""
    kwargs: dict = {"timeout": settings.llm_timeout}
    if settings.ollama_api_key:
        kwargs["headers"] = {"Authorization": f"Bearer {settings.ollama_api_key}"}
    return kwargs


@dataclass
class OllamaStatus:
    reachable: bool
    models: set[str] = field(default_factory=set)
    error: str | None = None

    def has(self, model: str) -> bool:
        return model in self.models or (":" not in model and f"{model}:latest" in self.models)


def check_ollama(settings: Settings) -> OllamaStatus:
    """List the models the Ollama server has pulled (``GET /api/tags``)."""
    headers = client_kwargs(settings).get("headers", {})
    try:
        resp = httpx.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", headers=headers, timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
    except Exception as exc:  # noqa: BLE001
        return OllamaStatus(False, error=f"{type(exc).__name__}: {exc}")
    return OllamaStatus(True, {m[k] for m in models for k in ("name", "model") if m.get(k)})


def build_llm(settings: Settings) -> LLM | None:
    if settings.llm_provider == "none":
        return None
    if settings.llm_provider != "ollama":
        raise ValueError(f"Unknown LLM provider {settings.llm_provider!r} (expected 'ollama' or 'none')")
    from langchain_ollama import ChatOllama

    chat = ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        num_ctx=settings.llm_num_ctx,
        keep_alive=settings.ollama_keep_alive,
        reasoning=False,  # keep <think> traces out of structured answers on reasoning models
        client_kwargs=client_kwargs(settings),
    )
    return LLM(chat, f"ollama:{settings.llm_model}", settings.llm_concurrency)
