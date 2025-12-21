"""Tests for src/search_context.py - Search context dataclasses."""

from __future__ import annotations

import pytest

from src.config import AgentConfig
from src.search_context import SearchContext, SearchServices, SearchState


def test_search_context_is_frozen():
    """Test SearchContext is immutable (frozen dataclass)."""
    context = SearchContext(
        current_datetime="2024-01-01 12:00:00",
        current_year="2024",
        current_month="01",
        current_day="01",
        user_query="test query",
        conversation_text="conversation",
        prior_responses_text="prior responses",
        question_embedding=[1.0, 2.0],
        topic_embedding_current=None,
    )

    with pytest.raises((AttributeError, TypeError)):  # FrozenInstanceError
        context.user_query = "modified"  # type: ignore[misc]


def test_search_state_is_mutable():
    """Test SearchState is mutable and has proper defaults."""
    state = SearchState()

    # Check defaults
    assert state.seen_urls == set()
    assert state.seen_result_hashes == set()
    assert state.seen_query_norms == set()
    assert state.topic_keywords == set()

    # Verify mutability
    state.seen_urls.add("http://example.com")
    assert "http://example.com" in state.seen_urls

    state.topic_keywords.add("python")
    assert "python" in state.topic_keywords


def test_search_state_with_initial_values():
    """Test SearchState can be created with initial values."""
    state = SearchState(
        seen_urls={"http://test.com"},
        seen_result_hashes={"hash1", "hash2"},
        seen_query_norms={"norm1"},
        topic_keywords={"keyword1", "keyword2"},
    )

    assert len(state.seen_urls) == 1
    assert len(state.seen_result_hashes) == 2
    assert len(state.seen_query_norms) == 1
    assert len(state.topic_keywords) == 2


def test_search_services_is_frozen():
    """Test SearchServices is immutable (frozen dataclass)."""
    cfg = AgentConfig()
    chains = {"test": lambda x: x}

    services = SearchServices(
        cfg=cfg,
        chains=chains,
        embedding_client=None,
        ddg_results=lambda q: [],
        inputs_builder=lambda *a, **k: {},
        reduce_context_and_rebuild=lambda k, label: None,
        mark_error=lambda m: m,
        context_similarity=lambda a, b, c: 0.0,
        char_budget=lambda n: n,
        rebuild_counts={"test": 0},
    )

    with pytest.raises((AttributeError, TypeError)):  # FrozenInstanceError
        services.cfg = AgentConfig()  # type: ignore[misc]


def test_search_services_stores_all_dependencies():
    """Test SearchServices correctly stores all dependency injection components."""
    cfg = AgentConfig(max_rounds=5)
    chains = {"planning": "chain1", "result_filter": "chain2"}
    embedding_client = "mock_embedding"

    def ddg_results(q):
        return [{"result": q}]

    def inputs_builder(*a, **k):
        return {"key": "value"}

    def reduce_fn(k, label):
        return None

    def mark_error(m):
        return f"Error: {m}"

    def sim_fn(a, b, c):
        return 0.75

    def budget_fn(n):
        return n * 2

    rebuild_counts = {"relevance": 1, "planning": 2}

    services = SearchServices(
        cfg=cfg,
        chains=chains,
        embedding_client=embedding_client,
        ddg_results=ddg_results,
        inputs_builder=inputs_builder,
        reduce_context_and_rebuild=reduce_fn,
        mark_error=mark_error,
        context_similarity=sim_fn,
        char_budget=budget_fn,
        rebuild_counts=rebuild_counts,
    )

    assert services.cfg.max_rounds == 5
    assert services.chains == chains
    assert services.embedding_client == embedding_client
    assert services.ddg_results("test") == [{"result": "test"}]
    assert services.inputs_builder() == {"key": "value"}
    assert services.mark_error("fail") == "Error: fail"
    assert services.context_similarity(None, None, None) == 0.75
    assert services.char_budget(100) == 200
    assert services.rebuild_counts["relevance"] == 1


def test_search_context_with_none_embeddings():
    """Test SearchContext handles None embeddings correctly."""
    context = SearchContext(
        current_datetime="2024-01-01",
        current_year="2024",
        current_month="01",
        current_day="01",
        user_query="query",
        conversation_text="conv",
        prior_responses_text="prior",
        question_embedding=None,
        topic_embedding_current=None,
    )

    assert context.question_embedding is None
    assert context.topic_embedding_current is None


def test_search_context_preserves_embeddings():
    """Test SearchContext preserves embedding vectors correctly."""
    question_emb = [0.1, 0.2, 0.3, 0.4]
    topic_emb = [0.5, 0.6, 0.7, 0.8]

    context = SearchContext(
        current_datetime="2024-01-01",
        current_year="2024",
        current_month="01",
        current_day="01",
        user_query="query",
        conversation_text="conv",
        prior_responses_text="prior",
        question_embedding=question_emb,
        topic_embedding_current=topic_emb,
    )

    assert context.question_embedding == question_emb
    assert context.topic_embedding_current == topic_emb
    assert len(context.question_embedding) == 4  # type: ignore[arg-type]
    assert len(context.topic_embedding_current) == 4  # type: ignore[arg-type]


def test_search_state_independent_sets():
    """Test SearchState maintains independent set instances."""
    state1 = SearchState()
    state2 = SearchState()

    state1.seen_urls.add("url1")
    state2.seen_urls.add("url2")

    assert "url1" in state1.seen_urls
    assert "url1" not in state2.seen_urls
    assert "url2" in state2.seen_urls
    assert "url2" not in state1.seen_urls


def test_search_context_datetime_fields():
    """Test SearchContext datetime fields are properly structured."""
    context = SearchContext(
        current_datetime="2024-12-21 15:30:45",
        current_year="2024",
        current_month="12",
        current_day="21",
        user_query="when is the next release?",
        conversation_text="Previous discussion about releases",
        prior_responses_text="Last release was in November",
        question_embedding=None,
        topic_embedding_current=None,
    )

    assert context.current_year == "2024"
    assert context.current_month == "12"
    assert context.current_day == "21"
    assert "15:30:45" in context.current_datetime
