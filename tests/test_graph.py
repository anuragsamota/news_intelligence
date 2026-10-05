from datetime import date

from conftest import FakeSource

from news_intelligence.graph import build_graph, research
from news_intelligence.nodes.planning import Plan, PlannedQuery
from news_intelligence.sources.vector import LocalVectorStore

QUERY = "Was the Northwind Harbor bank merger approved and what is the deal worth?"


async def test_end_to_end_offline_report(make_runtime, tmp_path):
    news, web = FakeSource(), FakeSource(name="fake_web", kind="web")
    rt = make_runtime(sources=[news, web])
    rt.vector_store = LocalVectorStore(str(tmp_path / "vdb"), rt.embedder)
    report = await research(QUERY, rt)

    # Decomposed into several sub-queries, each sent to both sources in parallel.
    assert len(report.sub_queries) >= 2
    assert len(news.queries) == len(web.queries) >= 2

    cited_urls = {c.url for c in report.citations}
    assert not any("bakery" in u for u in cited_urls), "off-topic story must be graded out"
    assert any("reuters.com" in u for u in cited_urls)
    assert not any("apnews.com" in u for u in cited_urls), "syndicated duplicate must be merged"

    # 12bn (Reuters/BBC) vs 9bn (Medium) is flagged, and the 12bn claim is corroborated elsewhere.
    assert any("12 vs 9" in c.explanation for c in report.contradictions)
    by_text = {c.text: c for c in report.claims}
    twelve = next(c for t, c in by_text.items() if "12 billion" in t)
    nine = next(c for t, c in by_text.items() if "9 billion" in t)
    assert twelve.status == nine.status == "contested"
    assert set(twelve.support_domains) == {"reuters.com", "bbc.com"}
    assert twelve.confidence > nine.confidence
    approval = next(c for t, c in by_text.items() if t.startswith("Regulators approved"))
    assert approval.status == "corroborated" and len(approval.support_domains) == 2

    assert date(2026, 3, 3) in {e.date for e in report.timeline}
    assert [e.date for e in report.timeline] == sorted(e.date for e in report.timeline)

    # Every narrative sentence carries a valid citation; findings are verified against sources.
    assert report.stats["grounding"] == 1.0
    valid = {c.n for c in report.citations}
    assert all(set(f.citations) <= valid and f.citations for f in report.key_findings)
    assert any(f.verified for f in report.key_findings)

    md = report.to_markdown()
    for heading in ("## Key findings", "## Timeline", "## Disputed claims", "## Evidence verification", "## Sources"):
        assert heading in md

    # Relevant stories were remembered in the vector DB for future runs.
    assert rt.vector_store.count() > 0


async def test_reflection_triggers_follow_up_round(make_runtime):
    empty = FakeSource(articles=[])
    rt = make_runtime(sources=[empty], max_iterations=2)
    report = await research(QUERY, rt)
    rounds = {sq.round for sq in report.sub_queries}
    assert rounds == {0, 1}
    assert report.stats["retrieval_rounds"] == 2
    assert any("No relevant sources" in w for w in report.warnings)


async def test_no_sources_still_produces_report(make_runtime):
    report = await research(QUERY, make_runtime(sources=[]))
    assert report.citations == [] and report.summary


class FakeLLM:
    """Answers only the planning call; every other node must fall back to heuristics."""

    name = "fake:planner"

    async def structured(self, schema, system, user):
        if schema is Plan:
            return Plan(sub_queries=[
                PlannedQuery(question="Did regulators approve the merger?", search_query="northwind harbor merger approval",
                             source_kinds=["news"], rationale="core"),
                PlannedQuery(question="How much is the deal worth?", search_query="northwind harbor deal value",
                             source_kinds=["web"], rationale="numbers"),
            ])
        return None


async def test_llm_plan_routes_sub_queries_to_requested_source_kinds(make_runtime):
    news, web = FakeSource(), FakeSource(name="fake_web", kind="web")
    report = await research(QUERY, make_runtime(sources=[news, web], llm=FakeLLM()))
    assert news.queries[0] == "northwind harbor merger approval"
    assert web.queries[0] == "northwind harbor deal value"
    assert report.stats["writer"].startswith("heuristic")
    assert report.citations


def test_graph_compiles_with_expected_nodes(make_runtime):
    nodes = set(build_graph(make_runtime()).get_graph().nodes)
    assert {"plan", "research_worker", "dedupe", "reflect", "rank", "extract_claims",
            "detect_contradictions", "build_timeline", "write_report", "remember"} <= nodes
