from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.search_orchestrator import SearchOrchestrator, SearchAbort
from src.config import AgentConfig
from src.exceptions import ResponseError


class _StubEmbeddingClient:
    def embed(self, text: str):
        return [0.1, 0.2]


class DummyEmbedding:
    def __init__(self, vec):
        self.vec = vec

    def embed(self, text: str):
        return list(self.vec)


class _StubEmbeddingNone:
    def embed(self, text: str):
        return None


def _inputs_builder(*args, **kwargs):
    return {}


def test_search_orchestrator_raises_on_result_filter_model_missing() -> None:
    """Integration test: SearchOrchestrator raises SearchAbort when result filter model is missing."""
    cfg = AgentConfig()

    def ddg_results(q: str):
        return [{"title": "T", "link": "http://x", "snippet": "S"}]

    class BadChain:
        def invoke(self, inputs):
            raise ResponseError("Model Not Found: Robot model not found")

    chains = {"result_filter": BadChain()}

    orch = SearchOrchestrator(
        cfg,
        ddg_results=ddg_results,
        embedding_client=_StubEmbeddingClient(),
        context_similarity=lambda a, b, c: 0.0,
        inputs_builder=_inputs_builder,
        reduce_context_and_rebuild=lambda key, label: None,
        rebuild_counts={"relevance": 0, "planning": 0, "query_filter": 0},
        char_budget=lambda x: x,
        mark_error=lambda m: m,
    )

    with pytest.raises(SearchAbort):
        orch.run(
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
            topic_keywords={"x"},
            primary_search_query="q",
        )


def test_result_included_when_similarity_exceeds_threshold() -> None:
    """Integration test: Results are included when embedding similarity exceeds threshold."""
    cfg = AgentConfig(embedding_result_similarity_threshold=0.1)

    def ddg_results(q: str):
        return [{"title": "T", "link": "http://x", "snippet": "S"}]

    chains = {
        "result_filter": SimpleNamespace(invoke=lambda inputs: "NO"),
        "planning": SimpleNamespace(invoke=lambda inputs: "NONE"),
        "query_filter": SimpleNamespace(invoke=lambda inputs: "NO"),
    }

    orch = SearchOrchestrator(
        cfg,
        ddg_results=ddg_results,
        embedding_client=DummyEmbedding([1.0]),
        context_similarity=lambda a, b, c: 0.2,
        inputs_builder=lambda *a, **k: {},
        reduce_context_and_rebuild=lambda key, label: None,
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
        question_embedding=[1.0],
        topic_embedding_current=None,
        topic_keywords=set(),
        primary_search_query="q",
    )

    assert aggregated


def test_result_accepted_by_result_filter() -> None:
    """Integration test: Results are accepted when result filter returns YES."""
    cfg = AgentConfig(embedding_result_similarity_threshold=0.5)

    def ddg_results(q: str):
        return [{"title": "T", "link": "http://x", "snippet": "S"}]

    class RF:
        def __init__(self):
            self.called = 0

        def invoke(self, inputs):
            self.called += 1
            return "YES"

    rf = RF()
    chains = {
        "result_filter": rf,
        "planning": SimpleNamespace(invoke=lambda inputs: "NONE"),
        "query_filter": SimpleNamespace(invoke=lambda inputs: "NO"),
    }

    orch = SearchOrchestrator(
        cfg,
        ddg_results=ddg_results,
        embedding_client=DummyEmbedding([0.0]),
        context_similarity=lambda a, b, c: 0.0,
        inputs_builder=lambda *a, **k: {},
        reduce_context_and_rebuild=lambda key, label: None,
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
        topic_keywords={"some"},
        primary_search_query="q",
    )

    assert aggregated
    assert rf.called >= 1


def test_relevance_retry_on_context_length_then_accepts(monkeypatch) -> None:
    """Integration test: SearchOrchestrator retries on context length error and succeeds."""
    cfg = AgentConfig(embedding_result_similarity_threshold=0.5)

    def ddg_results(q: str):
        return [{"title": "T", "link": "http://x", "snippet": "S"}]

    class RF:
        def __init__(self):
            self.behavior = [ResponseError("Context length exceeded"), "YES"]

        def invoke(self, inputs):
            val = self.behavior.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

    rf = RF()
    calls = {"reduced": 0}

    def reduce(key, label):
        calls["reduced"] += 1

    chains = {
        "result_filter": rf,
        "planning": SimpleNamespace(invoke=lambda inputs: "NONE"),
        "query_filter": SimpleNamespace(invoke=lambda inputs: "NO"),
    }

    orch = SearchOrchestrator(
        cfg,
        ddg_results=ddg_results,
        embedding_client=DummyEmbedding([0.0]),
        context_similarity=lambda a, b, c: 0.0,
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
        topic_keywords={"some"},
        primary_search_query="q",
    )

    assert aggregated
    assert calls["reduced"] == 1


def test_planning_retries_on_context_length_and_applies_rebuild() -> None:
    """Integration test: Planning chain retries on context length error and succeeds."""
    cfg = AgentConfig()

    def ddg_results(q: str):
        return [{"title": "T", "link": "http://x", "snippet": f"S:{q}"}]

    class PlanningChain:
        def __init__(self):
            self.behavior = [ResponseError("Context length exceeded"), "candidate"]
            self.called = 0

        def invoke(self, inputs):
            self.called += 1
            val = self.behavior.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

    planning = PlanningChain()

    class QF:
        def __init__(self):
            self.called = 0

        def invoke(self, inputs):
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
    assert qf.called >= 1


def test_query_filter_retries_on_context_length_then_accepts() -> None:
    """Integration test: Query filter retries on context length error and succeeds."""
    cfg = AgentConfig()

    def ddg_results(q: str):
        return [{"title": "T", "link": "http://x", "snippet": f"S:{q}"}]

    class Planning:
        def invoke(self, inputs):
            return "candidate"

    class QF:
        def __init__(self):
            self.behavior = [ResponseError("Context length exceeded"), "YES"]
            self.called = 0

        def invoke(self, inputs):
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


def test_low_similarity_skips_suggestion() -> None:
    """Integration test: Low similarity causes query suggestion to be skipped."""
    cfg = AgentConfig(embedding_query_similarity_threshold=0.9)

    def ddg_results(q: str):
        return [{"title": "T", "link": "http://x", "snippet": f"S:{q}"}]

    class Planning:
        def invoke(self, inputs):
            return "candidate"

    class QF:
        def invoke(self, inputs):
            return "YES"

    # embedding returns a vector, but similarity function returns low value
    embedding = DummyEmbedding([0.1, 0.2])

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
