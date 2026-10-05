# News Intelligence

A multi-source news researcher: an **agentic RAG** system built on **LangGraph**, running on a local **Ollama** backend. Give it a research question and it:

1. **breaks the question down** into focused sub-questions,
2. **searches several sources in parallel**: news APIs, web search and a local vector database,
3. **grades each result for relevance**, **removes duplicate stories**, and **ranks sources** by credibility, recency and how many outlets agree,
4. **checks its own coverage** and runs follow-up searches where it found too little,
5. **pulls out claims** and groups the ones that match across outlets, **flags contradictions** and **builds a timeline**,
6. writes a **report where every claim has a citation**, checks that each finding is backed by the sources it cites, and flags any sentence without a citation.

```
START → plan ─┬─ Send(sub-query × source) ─→ research_worker ─→ dedupe → reflect ─┐
              │   (parallel fan-out)              ▲  (retrieve + grade)           │ gaps?
              │                                   └──────── follow-up searches ───┤
              └─ (no sources) → dedupe                                            ▼ covered
      rank → extract_claims → detect_contradictions → build_timeline → write_report → remember → END
```

| Node | What it does |
|---|---|
| `plan` | The LLM splits the question into up to N sub-questions (facts, latest news, background, stakeholder reactions, disputed points). For each one it writes a keyword search query and picks which kinds of source to use (`news` / `web` / `vector`). |
| `research_worker` | One parallel worker per (sub-query, source) pair, sent out with LangGraph's `Send` API. It runs the search, then **grades relevance** (one batched LLM call, or a keyword + embedding score). A source that fails is logged, and the rest of the run carries on. |
| `dedupe` | First merges results with the same canonical URL. Then **semantic deduplication** merges copies of the same wire story. The most credible copy is kept and the other outlets are recorded. Independent reports of the same event are kept separate. |
| `reflect` | **Self-correction loop.** Counts the relevant stories for each sub-question. If any has too few, it writes new search queries and runs another retrieval round (`NI_MAX_ITERATIONS`). |
| `rank` | **Source ranking:** `0.45·relevance + 0.25·credibility + 0.15·recency + 0.15·corroboration`. Credibility starts from a per-domain prior (`credibility.py`), recency halves every 7 days, and corroboration counts the other outlets that ran the same story. |
| `extract_claims` | Pulls short, single-fact claims out of the top stories (one parallel LLM call per story) and groups matching claims across outlets. A claim is **corroborated** when 2 or more independent domains report it. |
| `detect_contradictions` | Pairs up related claims that come from different sources, and has the LLM judge whether each pair is a contradiction, consistent, or unrelated. The fallback rules look for different figures, opposite words (approved/rejected) and negation. Both claims in a contradiction are marked **contested**. |
| `build_timeline` | Builds a timeline from dates stated in the claims. A story with no stated date falls back to its earliest publication date, marked with `*`. Same-day events that match are merged. |
| `write_report` | The writer may use **only** the numbered sources. After writing, citations are **checked**: citations to sources that don't exist are removed, sentences without a citation are counted (the grounding score), and each key finding is **checked against the text of the sources it cites**. |
| `remember` | Saves the relevant articles to the vector DB, so later research runs can retrieve them. |

## Sources

| Source | Kind | Key needed |
|---|---|---|
| GDELT DOC 2.0 | news | no |
| Google News RSS | news | no |
| NewsAPI.org | news | `NEWSAPI_KEY` |
| Tavily | web search | `TAVILY_API_KEY` |
| Vector DB (ChromaDB, or a built-in JSONL+NumPy store) | vector | no |

To add a new source, subclass `sources.base.Source` with an async `search(query, k)` method and register it in `sources/__init__.py`.

## Ollama backend

Ollama provides both models:

| Role | Default | Used for |
|---|---|---|
| Chat model (`NI_LLM_MODEL`) | `qwen2.5:7b` | Planning, relevance grading, claim extraction, contradiction checks, report writing, finding checks |
| Embedding model (`NI_EMBEDDING_MODEL`) | `nomic-embed-text` | Vector DB search, deduplication, claim matching, timeline merging |

Every LLM call uses Ollama's JSON-schema structured output (`format=<schema>`), so the model has to return valid JSON. Any chat model that handles JSON well will work, for example `qwen2.5:14b`, `llama3.1:8b` or `mistral-nemo`. Bigger models give better reports.

**Fallbacks.** At startup the app asks Ollama which models are pulled (`/api/tags`):

- If Ollama can't be reached, the whole run switches to offline mode: rule-based steps plus an offline lexical embedder.
- If a model is missing, only that part falls back, and you get the exact `ollama pull …` command to run.
- If a single LLM call fails during a run (timeout or bad JSON), only that step falls back to its rule-based version.

The report's warnings and `Pipeline:` line show which mode each step used. Use `--offline` to skip Ollama entirely.

**Remote or cloud.** Set `OLLAMA_HOST` (or `NI_OLLAMA_BASE_URL`) to use Ollama on another machine. For Ollama Cloud, use `https://ollama.com` and set `OLLAMA_API_KEY`.

## Quick start

### Local

```bash
ollama pull qwen2.5:7b && ollama pull nomic-embed-text
pip install -e ".[chroma]"           # chroma is optional; without it a built-in vector store is used
cp .env.example .env && set -a && . ./.env && set +a

news-intelligence status             # check Ollama, models, sources and the vector DB
news-intelligence research "What is happening with EU AI Act enforcement?" -o report.md --json report.json
news-intelligence ingest ./my_articles/ https://example.com/some-article   # add to the vector DB
news-intelligence --model qwen2.5:14b research "..."                       # pick a model per run
```

### Docker (Ollama included)

```bash
cp .env.example .env
docker compose up -d ollama ollama-pull                 # start Ollama and pull the models (first run only)
docker compose run --rm news-intelligence status
docker compose run --rm news-intelligence research "Your question" -o /reports/report.md
```

Reports are saved to `./reports`. The Ollama models and the vector DB are kept in Docker volumes. For an NVIDIA GPU, uncomment the `deploy` block in `docker-compose.yml`.

See [`examples/sample_report.md`](examples/sample_report.md) for the report format. That example was generated in `--offline` mode from the test fixture corpus. In that run, a syndicated AP copy was merged, a $12bn vs $9bn conflict was flagged, the timeline was built, and every finding was checked.

### Python API

```python
import asyncio
from news_intelligence import Settings, build_runtime, research

rt = build_runtime(Settings.from_env(llm_model="qwen2.5:14b", max_iterations=3))
report = asyncio.run(research("Who is winning the EV price war in China?", rt))
print(report.to_markdown())
report.stats  # grounding, verified findings, contradictions, llm, embeddings, ...
```

## Configuration

All settings are in `config.py` and can be overridden with `NI_<FIELD>` environment variables. The ones you'll use most:

| Variable | Default | Meaning |
|---|---|---|
| `NI_MAX_SUB_QUERIES` | 5 | Number of sub-questions the planner writes |
| `NI_MAX_ITERATIONS` | 2 | Maximum retrieval rounds, including reflection follow-ups |
| `NI_RESULTS_PER_SOURCE` | 8 | Results fetched per source for each sub-query |
| `NI_RELEVANCE_THRESHOLD` | 0.5 | Minimum LLM relevance grade to keep a result |
| `NI_LLM_NUM_CTX` | 16384 | Context window requested from Ollama. Report writing sends about 20 source excerpts. |
| `NI_LLM_CONCURRENCY` | 2 | Number of LLM calls sent at once. Match `OLLAMA_NUM_PARALLEL` on the server. |
| `NI_DEDUP_THRESHOLD` | 0.9 | Calibrated similarity at which two results count as the same story (syndicated copies) |
| `NI_EMBEDDING_FLOOR` | 0.4 for Ollama, 0 for hashing | Cosine similarity that unrelated texts get with this embedder. All similarities are rescaled so this maps to 0, so the thresholds work the same with any embedder. |
| `NI_VECTOR_BACKEND` | auto | `chroma` when installed, otherwise `local`. `none` turns the vector DB off. |
| `NI_CREDIBILITY_FILE` | – | JSON `{domain: score}` that overrides or extends the credibility priors |

## Tests

```bash
pip install -e ".[dev]" && pytest -q
```

The tests run fully offline:

- **Pipeline tests** use a fixture news cycle: a wire story, a syndicated copy, an independent report that corroborates it, a blog post that contradicts it, a stakeholder reaction and an off-topic story. They check relevance filtering, deduplication, ranking, corroboration, contradiction detection, the timeline, citation grounding, the reflection loop, error handling and the vector DB.
- **Ollama integration tests** (`tests/fake_ollama.py`) run the real `langchain-ollama` client against a fake server that speaks Ollama's HTTP API (`/api/tags`, `/api/chat`, `/api/embed`). They check schema-constrained requests, LLM-driven planning, grading, judging and writing, removal of made-up citations, and the fallback when a model is missing.

## Limitations

- GDELT returns headlines only. Add NewsAPI or Tavily to get article text, and the claims and evidence checks get much better.
- The offline fallback embedder only matches near-duplicates and close rewording. Keep `nomic-embed-text` pulled to match true paraphrases.
- The default `NI_EMBEDDING_FLOOR=0.4` is a typical value for `nomic-embed-text`. If you switch to a different embedding model and see too many or too few merges, adjust it.
- Small local models (3B and under) often write weaker reports. A 7B–14B model is a good balance for speed and quality.
- Credibility scores are rough starting values, not fact-check verdicts. Adjust them for your domain.
