from __future__ import annotations

from types import SimpleNamespace

from src.config import AgentConfig
from src.exceptions import ResponseError
from src.search_orchestrator import SearchOrchestrator


class _StubEmbeddingNone:
    def embed(self, text: str):  # noqa: ANN001 - simple stub
        return None


class _StubEmbeddingList:
    def __init__(self, vec):
        self.vec = vec

    def embed(self, text: str):  # noqa: ANN001 - simple stub
        return list(self.vec)


def test_planning_retries_on_context_length_and_applies_rebuild() -> None:
    cfg = AgentConfig()

    def ddg_results(q: str):
        return [{"title": "T", "link": "http://x", "snippet": f"S:{q}"}]

    class PlanningChain:
        def __init__(self):
            self.behavior = [ResponseError("Context length exceeded"), "candidate"]
            self.called = 0

        def invoke(self, inputs):  # noqa: ANN001 - simple stub
            self.called += 1
            val = self.behavior.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

    planning = PlanningChain()

    class QF:
        def __init__(self):
            self.called = 0

        def invoke(self, inputs):  # noqa: ANN001 - simple stub
            self.called += 1
            return "YES"

    calls = {"reduced": 0}

    def reduce(key, label):
        calls["reduced"] += 1

    qf = QF()
    chains = {"planning": planning, "query_filter": qf}

    orch = SearchOrchestrator(
        cfg,
        ddg_results=ddg_results,
        embedding_client=_StubEmbeddingNone(),
        context_similarity=lambda a, b, c: 1.0,
        inputs_builder=lambda *a, **k: {},
        reduce_context_and_rebuild=reduce,
        rebuild_counts={"relevance": 0, "planning": 0, "query_filter": 0},
        char_budget=lambda x: x,
        mark_error=lambda m: m,
    )

    aggregated, kws = orch.run(
        chains=chains,
        should_search=True,
        user_query="q",
        current_datetime="d",
        current_year="y",
        current_month="m",
        current_day="dd",
        conversation_text="c",
        prior_responses_text="p",
        question_embedding=None,
        topic_embedding_current=None,
        topic_keywords=set(),
        primary_search_query="q",
    )

    assert calls["reduced"] == 1
    assert planning.called >= 2
    assert getattr(qf, "called", 0) >= 1


def test_query_filter_retries_on_context_length_then_accepts() -> None:
    cfg = AgentConfig()

    def ddg_results(q: str):
        return [{"title": "T", "link": "http://x", "snippet": f"S:{q}"}]

    class Planning:
        def invoke(self, inputs):  # noqa: ANN001 - simple stub
            return "candidate"

    class QF:
        def __init__(self):
            self.behavior = [ResponseError("Context length exceeded"), "YES"]
            self.called = 0

        def invoke(self, inputs):  # noqa: ANN001 - simple stub
            self.called += 1
            val = self.behavior.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

    qf = QF()
    calls = {"reduced": 0}

    def reduce(key, label):
        calls["reduced"] += 1

    chains = {"planning": Planning(), "query_filter": qf}

    orch = SearchOrchestrator(
        cfg,
        ddg_results=ddg_results,
        embedding_client=_StubEmbeddingNone(),
        context_similarity=lambda a, b, c: 1.0,
        inputs_builder=lambda *a, **k: {},
        reduce_context_and_rebuild=reduce,
        rebuild_counts={"relevance": 0, "planning": 0, "query_filter": 0},
        char_budget=lambda x: x,
        mark_error=lambda m: m,
    )

    aggregated, kws = orch.run(
        chains=chains,
        should_search=True,
        user_query="q",
        current_datetime="d",
        current_year="y",
        current_month="m",
        current_day="dd",
        conversation_text="c",
        prior_responses_text="p",
        question_embedding=None,
        topic_embedding_current=None,
        topic_keywords=set(),
        primary_search_query="q",
    )

    assert calls["reduced"] == 1
    assert qf.called >= 1


def test_fill_planning_retry_fails_and_stops(monkeypatch) -> None:
    cfg = AgentConfig(max_fill_attempts=2)

    aggregated: list[str] = []

    # planning raises context length then raises again on retry -> should break
    class Planning:
        def __init__(self):
            self.called = 0

        def invoke(self, inputs):  # noqa: ANN001 - simple stub
            self.called += 1
            if self.called == 1:
                raise ResponseError("Context length exceeded")
            raise ResponseError("Still too big")

    calls = {"reduced": 0}

    def reduce(key, label):
        calls["reduced"] += 1

    orch = SearchOrchestrator(
        cfg,
        ddg_results=lambda q: [],
        embedding_client=_StubEmbeddingNone(),
        context_similarity=lambda a, b, c: 1.0,
        inputs_builder=lambda *a, **k: {},
        reduce_context_and_rebuild=reduce,
        rebuild_counts={"relevance": 0, "planning": 0, "query_filter": 0},
        char_budget=lambda x: x,
        mark_error=lambda m: m,
    )

    # call private method directly to exercise fill loop
    orch._fill_pending_queries(
        chains={"planning": Planning(), "query_filter": SimpleNamespace(invoke=lambda i: "NO")},
        pending_queries=[],
        seen_query_norms=set(),
        aggregated_results=aggregated,
        question_embedding=None,
        topic_embedding_current=None,
        conversation_text="c",
        user_query="q",
        prior_responses_text="p",
        current_datetime="d",
        current_year="y",
        current_month="m",
        current_day="dd",
    )

    assert calls["reduced"] == 1
    assert aggregated == []


def test_parse_suggestions_strips_and_limits() -> None:
    from src.search_orchestrator import SearchOrchestrator

    raw = "- first\n* second\n 'third'\nfourth"
    out = SearchOrchestrator._parse_suggestions(raw, limit=2)
    assert out == ["first", "second"]


def test_parse_suggestions_returns_empty_on_none_marker() -> None:
    from src.search_orchestrator import SearchOrchestrator

    raw = "- first\nNone\nsecond"
    out = SearchOrchestrator._parse_suggestions(raw, limit=5)
    assert out == []


def test_low_similarity_skips_suggestion() -> None:
    cfg = AgentConfig(embedding_query_similarity_threshold=0.9)

    def ddg_results(q: str):
        return [{"title": "T", "link": "http://x", "snippet": f"S:{q}"}]

    class Planning:
        def invoke(self, inputs):  # noqa: ANN001
            return "candidate"

    class QF:
        def invoke(self, inputs):  # noqa: ANN001
            return "YES"

    # embedding returns a vector, but similarity function returns low value
    embedding = _StubEmbeddingList([0.1, 0.2])

    orch = SearchOrchestrator(
        cfg,
        ddg_results=ddg_results,
        embedding_client=embedding,
        context_similarity=lambda a, b, c: 0.1,
        inputs_builder=lambda *a, **k: {},
        reduce_context_and_rebuild=lambda k, label: None,
        rebuild_counts={"relevance": 0, "planning": 0, "query_filter": 0},
        char_budget=lambda x: x,
        mark_error=lambda m: m,
    )

    aggregated, kws = orch.run(
        chains={"planning": Planning(), "query_filter": QF()},
        should_search=True,
        user_query="q",
        current_datetime="d",
        current_year="y",
        current_month="m",
        current_day="dd",
        conversation_text="c",
        prior_responses_text="p",
        question_embedding=[1.0],
        topic_embedding_current=None,
        topic_keywords=set(),
        primary_search_query="q",
    )

    # candidate should be skipped due to low similarity; no results for candidate
    assert not any("S:candidate" in r for r in aggregated)
