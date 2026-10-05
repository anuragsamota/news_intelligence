from conftest import ARTICLES, FakeSource

from news_intelligence.models import SubQuery
from news_intelligence.nodes.curation import deduplicate, rank_documents
from news_intelligence.nodes.reporting import grounding_ratio, validate_citations
from news_intelligence.nodes.retrieval import grade
from news_intelligence.nodes.verification import heuristic_conflict
from news_intelligence.sources import make_document
from news_intelligence.sources.vector import LocalVectorStore, VectorStoreSource

QUERY = "Northwind Harbor bank merger approval"


def _docs(rel=0.8):
    return [make_document(provider="t", source_type="news", **a).model_copy(update={"relevance": rel}) for a in ARTICLES]


async def test_grading_filters_off_topic(make_runtime):
    rt = make_runtime()
    sq = SubQuery(id="s", question=QUERY, search_query="northwind harbor merger")
    kept, method = await grade(_docs(0), sq, QUERY, rt)
    titles = {d.title for d in kept}
    assert method == "heuristic"
    assert "Local bakery wins award for sourdough bread" not in titles
    assert "Regulators approve Northwind Harbor bank merger" in titles


def test_semantic_dedup_merges_syndicated_copies(make_runtime):
    corpus = deduplicate(_docs(), make_runtime())
    merged = [d for d in corpus if d.title == "Regulators approve Northwind Harbor bank merger"]
    assert len(merged) == 1
    assert merged[0].domain == "reuters.com"  # most credible copy wins
    assert "apnews.com" in merged[0].also_reported_by
    assert len(corpus) == len(ARTICLES) - 1


def test_ranking_prefers_credible_corroborated_sources(make_runtime):
    rt = make_runtime()
    ranked = rank_documents(deduplicate(_docs(), rt), rt)
    assert ranked[0].domain == "reuters.com"
    medium = next(d for d in ranked if d.domain == "medium.com")
    assert medium.score < ranked[0].score


def test_heuristic_conflict_detection():
    assert "different figures" in heuristic_conflict(
        "The deal is valued at 12 billion dollars.", "The deal is valued at 9 billion dollars.", 0.8
    )
    assert "opposite" in heuristic_conflict("Shareholders approved the merger.", "Shareholders rejected the merger.", 0.6)
    assert heuristic_conflict("The deal closed.", "Unions protested branch closures.", 0.1) is None


def test_validate_citations_drops_unknown_sources():
    text, bad = validate_citations("Approved on March 3 [1, 2]. Valued at 12bn [7].", {1, 2})
    assert text == "Approved on March 3 [1][2]. Valued at 12bn ."
    assert bad == [7]
    ratio, uncited = grounding_ratio(["The merger was approved by regulators [1]. Unions are not happy about this."])
    assert (ratio, uncited) == (0.5, 1)


async def test_local_vector_store_roundtrip(tmp_path, make_runtime):
    rt = make_runtime()
    store = LocalVectorStore(str(tmp_path), rt.embedder)
    assert store.add(_docs()) == len(ARTICLES)
    reopened = LocalVectorStore(str(tmp_path), rt.embedder)  # persisted to disk
    hits = await VectorStoreSource(reopened).search("sourdough bakery award", 2)
    assert hits[0].title.startswith("Local bakery")
    assert hits[0].source_type == "vector"


async def test_failing_source_does_not_break_worker(make_runtime):
    from news_intelligence.nodes.retrieval import research_worker

    rt = make_runtime(sources=[FakeSource(fail=True)])
    sq = SubQuery(id="s", question=QUERY, search_query="northwind")
    out = await research_worker({"query": QUERY, "sub_query": sq, "source_name": "fake_news"}, rt)
    assert out["retrieval_log"][0].error.startswith("RuntimeError")
    assert "documents" not in out
