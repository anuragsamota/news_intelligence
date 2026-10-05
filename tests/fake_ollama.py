"""A tiny stand-in for the Ollama HTTP API (/api/tags, /api/embed, /api/chat).

It lets the tests drive the real ``langchain-ollama`` client end to end without
downloading models. Chat answers are scripted per JSON schema (recognised by
the ``format`` the client sends), mimicking a well-behaved local model.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from news_intelligence.embeddings import HashingEmbedder

_EMBEDDER = HashingEmbedder(dim=256)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=\.)\s+", text) if len(s.split()) >= 5]


def scripted_answer(schema_props: set[str], user: str) -> dict:
    if "sub_queries" in schema_props:  # Plan
        return {"sub_queries": [
            {"question": "Did regulators approve the Northwind Harbor merger?",
             "search_query": "northwind harbor merger approval", "source_kinds": ["news"], "rationale": "core facts"},
            {"question": "What is the Northwind Harbor deal worth?",
             "search_query": "northwind harbor deal value", "source_kinds": ["news"], "rationale": "figures"},
        ]}
    if "grades" in schema_props:  # Grades
        items = re.findall(r"^\[(\d+)\] (.*)$", user, re.M)
        return {"grades": [{"index": int(i), "score": 0.0 if "bakery" in t.lower() else 0.9,
                            "reason": "off-topic" if "bakery" in t.lower() else "on topic"} for i, t in items]}
    if "claims" in schema_props:  # ExtractedClaims
        body = user.split("\n\n", 2)[-1]
        out = []
        for s in _sentences(body)[:3]:
            m = re.search(r"March (\d{1,2}), 2026", s)
            out.append({"text": s, "date": f"2026-03-{int(m.group(1)):02d}" if m else None})
        return {"claims": out}
    if "judgements" in schema_props:  # PairJudgements
        pairs = re.findall(r"^\[(\d+)\] A: (.*)\n\s+B: (.*)$", user, re.M)
        out = []
        for k, a, b in pairs:
            figures = {"12 billion", "9 billion"}
            hit = any(f in a for f in figures) and any(f in b for f in figures) and a != b
            out.append({"pair": int(k), "relation": "contradiction" if hit else "consistent",
                        "explanation": "12 billion vs 9 billion deal value" if hit else "compatible"})
        return {"judgements": out}
    if "key_findings" in schema_props:  # ReportDraft
        return {
            "title": "Northwind–Harbor merger: approval and disputed price",
            "summary": "Regulators approved the Northwind Harbor merger on March 3, 2026 [1][2]. "
                       "Most outlets put the deal at 12 billion dollars [1, 2], while one source says 9 billion [4]. "
                       "A citation to a source that does not exist must be removed [99].",
            "key_findings": [
                {"text": "Regulators approved the Northwind Harbor merger on March 3, 2026.", "citations": [1, 2]},
                {"text": "The deal is valued at 12 billion dollars.", "citations": [1, 2]},
            ],
            "sections": [{"heading": "Reaction", "body": "Unions criticised the planned branch closures [3]."}],
        }
    if "items" in schema_props:  # Verifications
        n = len(re.findall(r"^\((\d+)\) ", user, re.M))
        return {"items": [{"index": i, "supported": True, "note": "matches cited text"} for i in range(n)]}
    return {}


class FakeOllama:
    def __init__(self, models=("qwen2.5:7b", "nomic-embed-text:latest")):
        self.models = list(models)
        self.requests: list[tuple[str, dict]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence test output
                pass

            def _send(self, payload, ndjson=False):
                body = (json.dumps(payload) + "\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson" if ndjson else "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                server.requests.append((self.path, {}))
                if self.path == "/api/tags":
                    self._send({"models": [{"name": m, "model": m} for m in server.models]})
                else:
                    self.send_error(404)

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                server.requests.append((self.path, data))
                if self.path == "/api/embed":
                    inputs = data["input"] if isinstance(data["input"], list) else [data["input"]]
                    self._send({"model": data["model"], "embeddings": _EMBEDDER.embed(inputs).tolist()})
                elif self.path == "/api/chat":
                    fmt = data.get("format") or {}
                    props = set(fmt.get("properties", {})) if isinstance(fmt, dict) else set()
                    user = next((m["content"] for m in data["messages"] if m["role"] == "user"), "")
                    self._send({
                        "model": data["model"],
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "message": {"role": "assistant", "content": json.dumps(scripted_answer(props, user))},
                        "done": True,
                        "done_reason": "stop",
                    }, ndjson=bool(data.get("stream")))
                else:
                    self.send_error(404)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()

    def chat_requests(self) -> list[dict]:
        return [d for p, d in self.requests if p == "/api/chat"]
