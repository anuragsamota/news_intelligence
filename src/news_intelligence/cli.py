"""Command line interface: ``news-intelligence research|ingest|sources``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

import httpx

from .config import Settings
from .graph import build_runtime, research
from .models import Document
from .sources import make_document
from .text import parse_datetime, strip_html


def _progress(node: str, delta: dict) -> None:
    for line in delta.get("trace", []):
        print(f"  · {line}", file=sys.stderr)


def _settings(args, **overrides) -> Settings:
    if getattr(args, "offline", False):
        overrides.update(llm_provider="none", embedding_provider="hashing")
    return Settings.from_env(
        ollama_base_url=getattr(args, "ollama_url", None),
        llm_model=getattr(args, "model", None),
        embedding_model=getattr(args, "embed_model", None),
        **overrides,
    )


async def _research(args) -> int:
    rt = build_runtime(_settings(args, max_iterations=args.rounds, max_sub_queries=args.max_sub_queries))
    print(
        f"Researching: {args.query}\n  sources: {', '.join(s.name for s in rt.sources) or 'none'}"
        f" · llm: {rt.llm.name if rt.llm else 'none (heuristic mode)'} · embeddings: {rt.embedder.name}",
        file=sys.stderr,
    )
    for note in rt.notes:
        print(f"  ! {note}", file=sys.stderr)
    report = await research(args.query, rt, on_step=None if args.quiet else _progress)
    markdown = report.to_markdown()
    if args.out:
        Path(args.out).write_text(markdown)
        print(f"Report written to {args.out}", file=sys.stderr)
    else:
        print(markdown)
    if args.json:
        Path(args.json).write_text(report.model_dump_json(indent=2))
        print(f"JSON written to {args.json}", file=sys.stderr)
    return 0


def _load_path(path: Path) -> list[Document]:
    if path.suffix in (".json", ".jsonl"):
        text = path.read_text()
        rows = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(text)
        rows = rows if isinstance(rows, list) else [rows]
        return [
            make_document(
                title=r.get("title", path.stem),
                url=r.get("url", ""),
                outlet=r.get("outlet") or r.get("source"),
                provider="ingest",
                source_type="vector",
                content=r.get("content") or r.get("text") or r.get("description", ""),
                published_at=parse_datetime(r.get("published_at") or r.get("date")),
            )
            for r in rows
        ]
    text = path.read_text(errors="ignore")
    title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.strip()), path.stem)
    return [make_document(title=title, url=path.resolve().as_uri(), outlet=path.name, domain="local",
                          provider="ingest", source_type="vector", content=text)]


async def _load_url(url: str) -> Document:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": "news-intelligence/0.1"})
        resp.raise_for_status()
    html = resp.text
    title = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    body = re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html, flags=re.I | re.S)
    paragraphs = [strip_html(p) for p in re.findall(r"<p[^>]*>(.*?)</p>", body, re.I | re.S)]
    content = " ".join(p for p in paragraphs if len(p.split()) > 5) or strip_html(body)
    return make_document(title=strip_html(title.group(1)) if title else url, url=url,
                         provider="ingest", source_type="vector", content=content)


async def _ingest(args) -> int:
    rt = build_runtime(_settings(args))
    for note in rt.notes:
        print(f"! {note}", file=sys.stderr)
    if rt.vector_store is None:
        print("Vector store disabled (NI_VECTOR_BACKEND=none).", file=sys.stderr)
        return 1
    docs: list[Document] = []
    for item in args.items:
        if item.startswith(("http://", "https://")):
            docs.append(await _load_url(item))
            continue
        path = Path(item)
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for f in files:
            if f.suffix.lower() in (".txt", ".md", ".json", ".jsonl", ".html"):
                docs.extend(_load_path(f))
    added = rt.vector_store.add(docs)
    print(f"Ingested {len(docs)} document(s) as {added} chunk(s); store now holds {rt.vector_store.count()}.")
    return 0


def _status(args) -> int:
    from .llm import check_ollama

    settings = _settings(args)
    status = check_ollama(settings)
    print(f"Ollama  {settings.ollama_base_url}: " + ("reachable" if status.reachable else f"NOT reachable ({status.error})"))
    if status.reachable:
        for role, model in (("chat model", settings.llm_model), ("embeddings", settings.embedding_model)):
            ok = status.has(model)
            print(f"  {role:11} {model:24} {'ok' if ok else f'missing -> ollama pull {model}'}")
    rt = build_runtime(settings)
    print("Sources:")
    for s in rt.sources:
        print(f"  {s.name:12} {s.kind}")
    print(f"Pipeline: llm={rt.llm.name if rt.llm else 'none (heuristic mode)'} · embeddings={rt.embedder.name}")
    if rt.vector_store is not None:
        print(f"Vector store: {type(rt.vector_store).__name__} ({rt.vector_store.count()} chunks)")
    return 0 if status.reachable else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="news-intelligence", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--ollama-url", help="Ollama server URL (default http://localhost:11434 or $OLLAMA_HOST)")
    parser.add_argument("--model", help="Ollama chat model (default qwen2.5:7b)")
    parser.add_argument("--embed-model", help="Ollama embedding model (default nomic-embed-text)")
    parser.add_argument("--offline", action="store_true", help="No Ollama: heuristic pipeline + hashing embeddings")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("research", help="Research a question and write a cited report")
    p.add_argument("query")
    p.add_argument("-o", "--out", help="Write the Markdown report here (default: stdout)")
    p.add_argument("--json", help="Also write the full structured report as JSON")
    p.add_argument("--rounds", type=int, help="Max retrieval rounds (default 2)")
    p.add_argument("--max-sub-queries", type=int)
    p.add_argument("-q", "--quiet", action="store_true", help="Hide progress output")

    i = sub.add_parser("ingest", help="Add files, folders or URLs to the vector DB")
    i.add_argument("items", nargs="+")

    sub.add_parser("status", aliases=["sources"], help="Check Ollama, models, sources and the vector store")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    if args.command == "research":
        return asyncio.run(_research(args))
    if args.command == "ingest":
        return asyncio.run(_ingest(args))
    return _status(args)


if __name__ == "__main__":
    raise SystemExit(main())
