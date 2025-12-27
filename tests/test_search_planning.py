from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config import AgentConfig
from src.exceptions import ResponseError, SearchAbort
from src.search_context import SearchContext, SearchState, SearchServices
from src.search_planning import (
    generate_query_suggestions,
    enqueue_validated_queries,
    parse_suggestions,
)


class _StubEmbeddingClient:
    def __init__(self, vec=None):
        self.vec = vec or [0.1, 0.2]

    def embed(self, text: str):
        return list(self.vec) if self.vec else None


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


def _make_search_state(**overrides):
    """Create SearchState for tests."""
    defaults = {
        "seen_urls": set(),
        "seen_result_hashes": set(),
        "seen_query_norms": set(),
        "topic_keywords": set(),
    }
    return SearchState(**{**defaults, **overrides})


def _make_search_services(cfg=None, chains=None, embedding_client=None, context_similarity=None, **overrides):
    """Create SearchServices for tests."""
    defaults = {
        "cfg": cfg or AgentConfig(),
        "chains": chains or {},
        "embedding_client": embedding_client or _StubEmbeddingClient(),
        "ddg_results": lambda q: [],
        "inputs_builder": lambda *a, **k: {},
        "reduce_context_and_rebuild": lambda k, label: None,
        "mark_error": lambda m: m,
        "context_similarity": context_similarity or (lambda a, b, c: 0.0),
        "char_budget": lambda n: n,
        "rebuild_counts": {"planning": 0, "query_filter": 0},
    }
    return SearchServices(**{**defaults, **overrides})


def test_generate_query_suggestions_raise_on_error():
    """Test generate_query_suggestions raises SearchAbort when raise_on_error=True."""

    class BadChain:
        def invoke(self, inputs):
            raise ResponseError("Some server error")

    chains = {"planning": BadChain()}
    context = _make_search_context()
    services = _make_search_services(chains=chains)

    # raise_on_error True should raise SearchAbort
    with pytest.raises(SearchAbort):
        generate_query_suggestions(
            aggregated_results=[],
            suggestion_limit=2,
            context=context,
            services=services,
            raise_on_error=True,
        )

    # raise_on_error False should return empty list (fallback)
    res = generate_query_suggestions(
        aggregated_results=[],
        suggestion_limit=2,
        context=context,
        services=services,
        raise_on_error=False,
    )

    assert res == []


def test_generate_query_suggestions_returns_parsed_suggestions():
    """Test generate_query_suggestions returns parsed list of suggestions."""
    chains = {"planning": SimpleNamespace(invoke=lambda inputs: "Query 1\nQuery 2\nQuery 3")}
    context = _make_search_context()
    services = _make_search_services(chains=chains)

    res = generate_query_suggestions(
        aggregated_results=["result1", "result2"],
        suggestion_limit=2,
        context=context,
        services=services,
    )

    # Should parse and limit to 2
    assert len(res) == 2
    assert res[0] == "Query 1"
    assert res[1] == "Query 2"


def test_generate_query_suggestions_truncates_results():
    """Test generate_query_suggestions truncates aggregated results by char budget."""
    chains = {"planning": SimpleNamespace(invoke=lambda inputs: "Q1")}

    # Use char_budget that limits to 10 chars
    context = _make_search_context()
    services = _make_search_services(chains=chains, char_budget=lambda x: 10)

    # Aggregated results exceed budget
    long_results = ["x" * 100, "y" * 100]

    res = generate_query_suggestions(
        aggregated_results=long_results,
        suggestion_limit=1,
        context=context,
        services=services,
    )

    # Should still work (truncation happens internally)
    assert res == ["Q1"]


def test_enqueue_validated_queries_appends_valid_queries():
    """Test enqueue_validated_queries appends validated queries to pending list."""
    chains = {"query_filter": SimpleNamespace(invoke=lambda inputs: "YES")}
    pending = []

    context = _make_search_context()
    state = _make_search_state()
    services = _make_search_services(chains=chains, context_similarity=lambda a, b, c: 0.9)

    enqueue_validated_queries(
        candidate_queries=["query1", "query2"],
        pending_queries=pending,
        max_rounds=10,
        context=context,
        state=state,
        services=services,
    )

    assert len(pending) == 2
    assert "query1" in pending
    assert "query2" in pending
    # Note: seen_query_norms is NOT updated by enqueue (it's updated during query execution)
    assert len(state.seen_query_norms) == 0


def test_enqueue_validated_queries_skips_duplicates():
    """Test enqueue_validated_queries skips duplicate normalized queries."""
    chains = {"query_filter": SimpleNamespace(invoke=lambda inputs: "YES")}
    pending = []

    context = _make_search_context()
    state = _make_search_state(seen_query_norms={"duplicate query"})
    services = _make_search_services(chains=chains, context_similarity=lambda a, b, c: 0.9)

    enqueue_validated_queries(
        candidate_queries=["Duplicate Query", "new query"],  # First is duplicate
        pending_queries=pending,
        max_rounds=10,
        context=context,
        state=state,
        services=services,
    )

    # Should only add "new query"
    assert len(pending) == 1
    assert "new query" in pending


def test_enqueue_validated_queries_respects_max_rounds():
    """Test enqueue_validated_queries stops when max_rounds reached."""
    chains = {"query_filter": SimpleNamespace(invoke=lambda inputs: "YES")}
    pending = ["existing1", "existing2"]  # Already 2 queries

    context = _make_search_context()
    state = _make_search_state()
    services = _make_search_services(chains=chains, context_similarity=lambda a, b, c: 0.9)

    enqueue_validated_queries(
        candidate_queries=["query1", "query2", "query3"],
        pending_queries=pending,
        max_rounds=3,  # Only room for 1 more
        context=context,
        state=state,
        services=services,
    )

    # Should only add 1 query (to reach max of 3)
    assert len(pending) == 3


def test_parse_suggestions():
    """Test parse_suggestions parses LLM output correctly."""
    raw = "- Query 1\n* Query 2\nQuery 3\n  \n"
    res = parse_suggestions(raw, limit=10)

    assert len(res) == 3
    assert res[0] == "Query 1"
    assert res[1] == "Query 2"
    assert res[2] == "Query 3"


def test_parse_suggestions_returns_empty_on_none():
    """Test parse_suggestions returns empty list when NONE found."""
    raw = "Query 1\nNONE\nQuery 2"
    res = parse_suggestions(raw, limit=10)

    # Should return empty list immediately when "none" encountered
    assert res == []


def test_parse_suggestions_limits_results():
    """Test parse_suggestions respects limit parameter."""
    raw = "Q1\nQ2\nQ3\nQ4\nQ5"
    res = parse_suggestions(raw, limit=3)

    assert len(res) == 3
    assert res == ["Q1", "Q2", "Q3"]
