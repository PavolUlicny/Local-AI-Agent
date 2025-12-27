from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.search_orchestrator import SearchOrchestrator
from src.search_context import SearchContext, SearchServices
from src.config import AgentConfig
from src.exceptions import ResponseError, SearchAbort


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


def _make_search_context(**overrides):
    """Create SearchContext for tests."""
    defaults = {
        "current_datetime": "d",
        "current_year": "y",
        "current_month": "m",
        "current_day": "dd",
        "user_query": "q",
        "conversation_text": "c",
        "prior_responses_text": "p",
        "question_embedding": None,
        "topic_embedding_current": None,
    }
    return SearchContext(**{**defaults, **overrides})


def _make_query_inputs(**overrides):
    """Create query_inputs dict for tests using new API."""
    defaults = {
        "current_datetime": "d",
        "current_year": "y",
        "current_month": "m",
        "current_day": "dd",
        "conversation_history": "c",
        "user_question": "q",
        "known_answers": "p",
    }
    return {**defaults, **overrides}


def _make_search_services(
    cfg=None, chains=None, ddg_results=None, embedding_client=None, context_similarity=None, **overrides
):
    """Create SearchServices for tests."""
    defaults = {
        "cfg": cfg or AgentConfig(),
        "chains": chains or {},
        "embedding_client": embedding_client or _StubEmbeddingClient(),
        "ddg_results": ddg_results or (lambda q: []),
        "inputs_builder": lambda *a, **k: {},
        "reduce_context_and_rebuild": lambda k, label: None,
        "mark_error": lambda m: m,
        "context_similarity": context_similarity or (lambda a, b, c: 0.0),
        "char_budget": lambda n: n,
        "rebuild_counts": {"relevance": 0, "planning": 0, "query_filter": 0},
    }
    return SearchServices(**{**defaults, **overrides})


def test_search_orchestrator_raises_on_result_filter_model_missing() -> None:
    """Integration test: SearchOrchestrator raises SearchAbort when result filter model is missing."""
    cfg = AgentConfig(
        embedding_result_similarity_threshold=0.9,  # High threshold to force LLM check
        embedding_query_similarity_threshold=0.1,  # Low threshold to pass query validation
    )

    def ddg_results(q: str):
        return [{"title": "about dogs", "link": "http://x", "snippet": "canines"}]

    class BadChain:
        def invoke(self, inputs):
            raise ResponseError("Model Not Found: Robot model not found")

    chains = {
        "result_filter": BadChain(),
        "planning": SimpleNamespace(invoke=lambda inputs: "test query"),
        "query_filter": SimpleNamespace(invoke=lambda inputs: "YES"),
    }

    # Use user_query with keywords that don't match result to force LLM check
    query_inputs = _make_query_inputs(user_question="cats birds fish")
    services = _make_search_services(
        cfg=cfg,
        chains=chains,
        ddg_results=ddg_results,
        embedding_client=DummyEmbedding([0.0]),
        context_similarity=lambda a, b, c: 0.5,  # 0.5 > 0.1 (query passes), 0.5 < 0.9 (result goes to LLM)
    )
    orch = SearchOrchestrator(services)

    with pytest.raises(SearchAbort):
        orch.run(query_inputs=query_inputs, user_query="cats birds fish")


def test_result_included_when_similarity_exceeds_threshold() -> None:
    """Integration test: Results are included when embedding similarity exceeds threshold."""
    cfg = AgentConfig(embedding_result_similarity_threshold=0.1)

    def ddg_results(q: str):
        return [{"title": "T", "link": "http://x", "snippet": "S"}]

    chains = {
        "result_filter": SimpleNamespace(invoke=lambda inputs: "NO"),
        "planning": SimpleNamespace(invoke=lambda inputs: "test query"),
        "query_filter": SimpleNamespace(invoke=lambda inputs: "YES"),
    }

    query_inputs = _make_query_inputs()
    services = _make_search_services(
        cfg=cfg,
        chains=chains,
        ddg_results=ddg_results,
        embedding_client=DummyEmbedding([1.0]),
        context_similarity=lambda a, b, c: 1.0,  # High similarity to pass validation
    )
    orch = SearchOrchestrator(services)

    aggregated = orch.run(query_inputs=query_inputs, user_query="q")

    assert aggregated


def test_result_accepted_by_result_filter() -> None:
    """Integration test: Results are accepted when result filter returns YES."""
    cfg = AgentConfig(
        embedding_result_similarity_threshold=0.9,  # High threshold to force LLM check
        embedding_query_similarity_threshold=0.1,  # Low threshold to pass query validation
    )

    def ddg_results(q: str):
        return [{"title": "about dogs", "link": "http://x", "snippet": "canines"}]

    class RF:
        def __init__(self):
            self.called = 0

        def invoke(self, inputs):
            self.called += 1
            return "YES"

    rf = RF()
    chains = {
        "result_filter": rf,
        "planning": SimpleNamespace(invoke=lambda inputs: "test query"),
        "query_filter": SimpleNamespace(invoke=lambda inputs: "YES"),
    }

    # Use user_query with keywords that don't match result to force LLM check
    query_inputs = _make_query_inputs(user_question="cats birds fish")
    services = _make_search_services(
        cfg=cfg,
        chains=chains,
        ddg_results=ddg_results,
        embedding_client=DummyEmbedding([0.0]),
        context_similarity=lambda a, b, c: 0.5,  # 0.5 > 0.1 (query passes), 0.5 < 0.9 (result goes to LLM)
    )
    orch = SearchOrchestrator(services)

    aggregated = orch.run(query_inputs=query_inputs, user_query="cats birds fish")

    assert aggregated
    assert rf.called >= 1


def test_relevance_retry_on_context_length_then_accepts(monkeypatch) -> None:
    """Integration test: SearchOrchestrator retries on context length error and succeeds."""
    cfg = AgentConfig(
        embedding_result_similarity_threshold=0.9,  # High threshold to force LLM check
        embedding_query_similarity_threshold=0.1,  # Low threshold to pass query validation
    )

    def ddg_results(q: str):
        return [{"title": "about dogs", "link": "http://x", "snippet": "canines"}]

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
        "planning": SimpleNamespace(invoke=lambda inputs: "test query"),
        "query_filter": SimpleNamespace(invoke=lambda inputs: "YES"),
    }

    # Use user_query with keywords that don't match result to force LLM check
    query_inputs = _make_query_inputs(user_question="cats birds fish")
    services = _make_search_services(
        cfg=cfg,
        chains=chains,
        ddg_results=ddg_results,
        embedding_client=DummyEmbedding([0.0]),
        context_similarity=lambda a, b, c: 0.5,  # 0.5 > 0.1 (query passes), 0.5 < 0.9 (result goes to LLM)
        reduce_context_and_rebuild=reduce,
    )
    orch = SearchOrchestrator(services)

    aggregated = orch.run(query_inputs=query_inputs, user_query="cats birds fish")

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

    query_inputs = _make_query_inputs()
    services = _make_search_services(
        cfg=cfg,
        chains=chains,
        ddg_results=ddg_results,
        embedding_client=_StubEmbeddingNone(),
        context_similarity=lambda a, b, c: 1.0,
        reduce_context_and_rebuild=reduce,
    )
    orch = SearchOrchestrator(services)

    orch.run(query_inputs=query_inputs, user_query="q")

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

    query_inputs = _make_query_inputs()
    services = _make_search_services(
        cfg=cfg,
        chains=chains,
        ddg_results=ddg_results,
        embedding_client=_StubEmbeddingNone(),
        context_similarity=lambda a, b, c: 1.0,
        reduce_context_and_rebuild=reduce,
    )
    orch = SearchOrchestrator(services)

    orch.run(query_inputs=query_inputs, user_query="q")

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

    query_inputs = _make_query_inputs()
    services = _make_search_services(
        cfg=cfg,
        chains={
            "planning": Planning(),
            "query_filter": QF(),
            "result_filter": SimpleNamespace(invoke=lambda inputs: "NO"),
        },
        ddg_results=ddg_results,
        embedding_client=embedding,
        context_similarity=lambda a, b, c: 0.1,
    )
    orch = SearchOrchestrator(services)

    aggregated = orch.run(query_inputs=query_inputs, user_query="q")

    # candidate should be skipped due to low similarity; no results for candidate
    assert not any("S:candidate" in r for r in aggregated)


def test_orchestrator_returns_empty_when_planning_returns_none_initially() -> None:
    """Integration test: Orchestrator returns empty when planning returns NONE for initial queries."""
    cfg = AgentConfig()

    def ddg_results(q: str):
        return [{"title": "T", "link": "http://x", "snippet": "S"}]

    chains = {
        "planning": SimpleNamespace(invoke=lambda inputs: "NONE"),  # Returns NONE for initial call
        "query_filter": SimpleNamespace(invoke=lambda inputs: "YES"),
        "result_filter": SimpleNamespace(invoke=lambda inputs: "YES"),
    }

    query_inputs = _make_query_inputs()
    services = _make_search_services(
        cfg=cfg,
        chains=chains,
        ddg_results=ddg_results,
        embedding_client=_StubEmbeddingClient(),
        context_similarity=lambda a, b, c: 1.0,
    )
    orch = SearchOrchestrator(services)

    aggregated = orch.run(query_inputs=query_inputs, user_query="q")

    # Planning returned NONE, so no queries generated, search returns empty
    assert aggregated == []


def test_orchestrator_returns_empty_when_all_initial_queries_filtered() -> None:
    """Integration test: Orchestrator returns empty when all initial queries fail validation."""
    cfg = AgentConfig(embedding_query_similarity_threshold=0.9)

    def ddg_results(q: str):
        return [{"title": "T", "link": "http://x", "snippet": "S"}]

    chains = {
        "planning": SimpleNamespace(invoke=lambda inputs: "some query"),  # Returns a query
        "query_filter": SimpleNamespace(invoke=lambda inputs: "NO"),  # But LLM rejects it
        "result_filter": SimpleNamespace(invoke=lambda inputs: "YES"),
    }

    query_inputs = _make_query_inputs()
    services = _make_search_services(
        cfg=cfg,
        chains=chains,
        ddg_results=ddg_results,
        embedding_client=_StubEmbeddingClient(),
        context_similarity=lambda a, b, c: 0.1,  # Low similarity - fails validation
    )
    orch = SearchOrchestrator(services)

    aggregated = orch.run(query_inputs=query_inputs, user_query="q")

    # All queries filtered out, search returns empty
    assert aggregated == []


def test_enqueued_query_executes_not_skipped_as_duplicate() -> None:
    """Integration test: Query enqueued in pending_queries executes without being skipped."""
    cfg = AgentConfig()

    call_count = {"ddg": 0}

    def ddg_results(q: str):
        call_count["ddg"] += 1
        return [{"title": f"T{call_count['ddg']}", "link": f"http://x/{call_count['ddg']}", "snippet": "S"}]

    chains = {
        "planning": SimpleNamespace(invoke=lambda inputs: "test query"),  # Returns same query
        "query_filter": SimpleNamespace(invoke=lambda inputs: "YES"),
        "result_filter": SimpleNamespace(invoke=lambda inputs: "YES"),
    }

    query_inputs = _make_query_inputs()
    services = _make_search_services(
        cfg=cfg,
        chains=chains,
        ddg_results=ddg_results,
        embedding_client=_StubEmbeddingClient(),
        context_similarity=lambda a, b, c: 1.0,
    )
    orch = SearchOrchestrator(services)

    aggregated = orch.run(query_inputs=query_inputs, user_query="q")

    # Query should have been executed (not skipped), so we should have results
    assert len(aggregated) > 0
    assert call_count["ddg"] == 1  # Search was executed once


def test_executed_query_prevents_duplicate_in_next_planning_round() -> None:
    """Integration test: Query executed in round 1 cannot be added again in round 2."""
    cfg = AgentConfig(max_rounds=3)

    class PlanningChain:
        def __init__(self):
            self.call_count = 0

        def invoke(self, inputs):
            self.call_count += 1
            if self.call_count == 1:
                return "query1"  # First call: initial query
            elif self.call_count == 2:
                return "query1\nquery2"  # Second call: try query1 again (should be filtered)
            else:
                return "NONE"

    planning = PlanningChain()
    search_calls = []

    def ddg_results(q: str):
        search_calls.append(q)
        return [{"title": f"T:{q}", "link": f"http://x/{q}", "snippet": "S"}]

    chains = {
        "planning": planning,
        "query_filter": SimpleNamespace(invoke=lambda inputs: "YES"),
        "result_filter": SimpleNamespace(invoke=lambda inputs: "YES"),
    }

    query_inputs = _make_query_inputs()
    services = _make_search_services(
        cfg=cfg,
        chains=chains,
        ddg_results=ddg_results,
        embedding_client=_StubEmbeddingClient(),
        context_similarity=lambda a, b, c: 1.0,
    )
    orch = SearchOrchestrator(services)

    aggregated = orch.run(query_inputs=query_inputs, user_query="q")

    # Should have executed query1 and query2, but query1 only once (not twice)
    assert len(search_calls) == 2
    assert search_calls[0] == "query1"
    assert search_calls[1] == "query2"
    # Results should contain both queries
    assert any("query1" in r for r in aggregated)
    assert any("query2" in r for r in aggregated)
