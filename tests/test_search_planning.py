from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config import AgentConfig
from src.exceptions import ResponseError
from src.search_planning import (
    generate_query_suggestions,
    enqueue_validated_queries,
    parse_suggestions,
)
from src.search_chain_utils import SearchAbort


class _StubEmbeddingClient:
    def __init__(self, vec=None):
        self.vec = vec or [0.1, 0.2]

    def embed(self, text: str):
        return list(self.vec) if self.vec else None


def _make_suggestion_kwargs(cfg=None):
    """Create common kwargs for generate_query_suggestions."""
    return {
        "cfg": cfg or AgentConfig(),
        "inputs_builder": lambda *a, **k: {},
        "char_budget": lambda x: x,
        "reduce_context_and_rebuild": lambda k, label: None,
        "rebuild_counts": {"planning": 0},
        "mark_error": lambda m: m,
    }


def _make_enqueue_kwargs(cfg=None, embedding_client=None, context_similarity=None):
    """Create common kwargs for enqueue_validated_queries."""
    return {
        "cfg": cfg or AgentConfig(),
        "embedding_client": embedding_client or _StubEmbeddingClient(),
        "context_similarity": context_similarity or (lambda a, b, c: 0.0),
        "inputs_builder": lambda *a, **k: {},
        "reduce_context_and_rebuild": lambda k, label: None,
        "rebuild_counts": {"query_filter": 0},
        "mark_error": lambda m: m,
    }


def test_generate_query_suggestions_raise_on_error():
    """Test generate_query_suggestions raises SearchAbort when raise_on_error=True."""

    class BadChain:
        def invoke(self, inputs):
            raise ResponseError("Some server error")

    chains = {"planning": BadChain()}
    common = _make_suggestion_kwargs()

    # raise_on_error True should raise SearchAbort
    with pytest.raises(SearchAbort):
        generate_query_suggestions(
            chains=chains,
            aggregated_results=[],
            suggestion_limit=2,
            user_query="q",
            conversation_text="c",
            prior_responses_text="p",
            current_datetime="d",
            current_year="y",
            current_month="m",
            current_day="dd",
            raise_on_error=True,
            **common,
        )

    # raise_on_error False should return empty list (fallback)
    res = generate_query_suggestions(
        chains=chains,
        aggregated_results=[],
        suggestion_limit=2,
        user_query="q",
        conversation_text="c",
        prior_responses_text="p",
        current_datetime="d",
        current_year="y",
        current_month="m",
        current_day="dd",
        raise_on_error=False,
        **common,
    )
    assert res == []


def test_generate_query_suggestions_returns_empty_on_NONE():
    """Test generate_query_suggestions returns empty list on NONE response."""
    chains = {"planning": SimpleNamespace(invoke=lambda inputs: "NONE")}
    common = _make_suggestion_kwargs()

    queries = generate_query_suggestions(
        chains=chains,
        aggregated_results=["r1"],
        suggestion_limit=3,
        user_query="q",
        conversation_text="c",
        prior_responses_text="p",
        current_datetime="d",
        current_year="y",
        current_month="m",
        current_day="dd",
        raise_on_error=False,
        **common,
    )

    assert queries == []


def test_enqueue_validated_queries_respects_duplicates_and_max_rounds():
    """Test enqueue_validated_queries respects duplicates and max_rounds limits."""
    chains = {"query_filter": SimpleNamespace(invoke=lambda inputs: "YES")}
    common = _make_enqueue_kwargs()

    # Duplicates should be skipped; unique candidates enqueued
    pending = []
    seen = set()
    enqueue_validated_queries(
        candidate_queries=["A", "A", "B"],
        pending_queries=pending,
        seen_query_norms=seen,
        max_rounds=10,
        chains=chains,
        question_embedding=None,
        topic_embedding_current=None,
        user_query="q",
        conversation_text="c",
        current_datetime="d",
        current_year="y",
        current_month="m",
        current_day="dd",
        **common,
    )
    assert pending == ["A", "B"]
    assert {x.lower() for x in seen} == {"a", "b"}

    # Respect max_rounds: when pending length >= max_rounds no new items are added
    pending2 = ["x"]
    seen2 = {"x"}
    enqueue_validated_queries(
        candidate_queries=["C"],
        pending_queries=pending2,
        seen_query_norms=seen2,
        max_rounds=1,
        chains=chains,
        question_embedding=None,
        topic_embedding_current=None,
        user_query="q",
        conversation_text="c",
        current_datetime="d",
        current_year="y",
        current_month="m",
        current_day="dd",
        **common,
    )
    assert pending2 == ["x"]


def test_enqueue_validated_queries_handles_rejected_queries():
    """Test enqueue_validated_queries handles LLM rejecting queries."""
    chains = {"query_filter": SimpleNamespace(invoke=lambda inputs: "NO")}
    common = _make_enqueue_kwargs()

    pending = []
    seen = set()
    enqueue_validated_queries(
        candidate_queries=["A", "B"],
        pending_queries=pending,
        seen_query_norms=seen,
        max_rounds=10,
        chains=chains,
        question_embedding=None,
        topic_embedding_current=None,
        user_query="q",
        conversation_text="c",
        current_datetime="d",
        current_year="y",
        current_month="m",
        current_day="dd",
        **common,
    )

    # All rejected, nothing enqueued
    assert pending == []


def test_parse_suggestions_strips_and_limits():
    """Test parse_suggestions strips formatting and respects limit."""
    raw = "- first\n* second\n 'third'\nfourth"
    out = parse_suggestions(raw, limit=2)
    assert out == ["first", "second"]


def test_parse_suggestions_returns_empty_on_none_marker():
    """Test parse_suggestions returns empty list on NONE marker."""
    raw = "- first\nNone\nsecond"
    out = parse_suggestions(raw, limit=5)
    assert out == []
