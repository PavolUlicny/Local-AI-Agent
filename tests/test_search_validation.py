from __future__ import annotations

from types import SimpleNamespace

from src.config import AgentConfig
from src.search_context import SearchContext, SearchServices
from src.search_validation import check_result_relevance, validate_candidate_query


class _StubEmbeddingClient:
    def __init__(self, vec=None):
        self.vec = vec or [0.1, 0.2]

    def embed(self, text: str):
        return list(self.vec) if self.vec else None


def _make_search_context(user_query="q", question_embedding=None, topic_embedding_current=None, **overrides):
    """Create SearchContext for tests."""
    defaults = {
        "current_datetime": "d",
        "current_year": "y",
        "current_month": "m",
        "current_day": "dd",
        "user_query": user_query,
        "conversation_text": "c",
        "prior_responses_text": "p",
        "question_embedding": question_embedding,
        "topic_embedding_current": topic_embedding_current,
    }
    return SearchContext(**{**defaults, **overrides})


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
        "rebuild_counts": {"relevance": 0, "query_filter": 0},
    }
    return SearchServices(**{**defaults, **overrides})


def test_check_result_relevance_embedding_threshold():
    """Test check_result_relevance passes when embedding similarity exceeds threshold."""
    cfg = AgentConfig(embedding_result_similarity_threshold=0.1)
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "NO")}

    context = _make_search_context(question_embedding=[1.0])
    services = _make_search_services(
        cfg=cfg,
        chains=chains,
        context_similarity=lambda a, b, c: 0.5,  # High similarity
    )

    is_rel, checks = check_result_relevance(
        result_text="some text",
        keywords_source="some",
        current_query="q",
        topic_keywords=set(),
        relevance_llm_checks=0,
        context=context,
        services=services,
    )

    assert is_rel is True
    assert checks == 0


def test_check_result_relevance_llm_increments_check_count():
    """Test check_result_relevance increments LLM check counter when LLM is used."""

    class RF:
        def invoke(self, inputs):
            return "YES"

    chains = {"result_filter": RF()}
    context = _make_search_context()
    services = _make_search_services(chains=chains)

    is_rel, checks = check_result_relevance(
        result_text="txt",
        keywords_source="k",
        current_query="q",
        topic_keywords={"kw"},
        relevance_llm_checks=0,
        context=context,
        services=services,
    )

    assert is_rel is True
    assert checks == 1


def test_check_result_relevance_respects_llm_check_limit():
    """Test check_result_relevance skips LLM when limit reached."""
    cfg = AgentConfig(max_relevance_llm_checks=2, embedding_result_similarity_threshold=0.9)
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "YES")}

    context = _make_search_context(question_embedding=[0.5])
    services = _make_search_services(cfg=cfg, chains=chains)

    is_rel, checks = check_result_relevance(
        result_text="this text about dogs",
        keywords_source="dogs",
        current_query="q",
        topic_keywords={"cats", "birds"},  # Keywords don't match "dogs"
        relevance_llm_checks=2,  # Already at limit
        context=context,
        services=services,
    )

    # Should be False: tier 1 fails (keyword mismatch), tier 2 fails (low similarity 0.0 < 0.9), tier 3 skipped (limit reached)
    assert is_rel is False
    assert checks == 2  # Unchanged


def test_validate_candidate_query_embedding_and_filter():
    """Test validate_candidate_query rejects low similarity candidates."""
    chains = {"query_filter": SimpleNamespace(invoke=lambda inputs: "YES")}
    context = _make_search_context(question_embedding=[1.0])
    services = _make_search_services(chains=chains)

    ok = validate_candidate_query(
        candidate="cand",
        context=context,
        services=services,
    )

    # similarity 0 < default threshold (0.5) => should be False
    assert ok is False


def test_validate_candidate_query_no_embeddings_still_filters():
    """Test validate_candidate_query works when embeddings are None."""

    class NoneEmbedding:
        def embed(self, text):
            return None

    chains = {"query_filter": SimpleNamespace(invoke=lambda inputs: "YES")}
    context = _make_search_context()
    services = _make_search_services(chains=chains, embedding_client=NoneEmbedding())

    ok = validate_candidate_query(
        candidate="cand",
        context=context,
        services=services,
    )

    # Should skip embedding check and go straight to LLM filter
    assert ok is True


def test_check_result_relevance_uses_topic_embedding_when_question_embedding_none():
    """Test check_result_relevance uses topic_embedding when question_embedding is None."""
    cfg = AgentConfig(embedding_result_similarity_threshold=0.1)
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "NO")}

    context = _make_search_context(
        question_embedding=None,  # No question embedding
        topic_embedding_current=[0.5, 0.5],  # But topic embedding present
    )
    services = _make_search_services(
        cfg=cfg,
        chains=chains,
        context_similarity=lambda a, b, c: 0.8,  # High similarity
    )

    is_rel, checks = check_result_relevance(
        result_text="some text",
        keywords_source="some",
        current_query="q",
        topic_keywords=set(),
        relevance_llm_checks=0,
        context=context,
        services=services,
    )

    # Should use topic_embedding and pass due to high similarity
    assert is_rel is True
    assert checks == 0


def test_validate_candidate_query_passes_when_similarity_high_and_llm_yes():
    """Test validate_candidate_query accepts candidates with high similarity and LLM YES."""
    cfg = AgentConfig(embedding_query_similarity_threshold=0.5)
    chains = {"query_filter": SimpleNamespace(invoke=lambda inputs: "YES")}  # LLM accepts

    context = _make_search_context(question_embedding=[1.0])
    services = _make_search_services(
        cfg=cfg,
        chains=chains,
        context_similarity=lambda a, b, c: 0.9,  # High similarity
    )

    ok = validate_candidate_query(
        candidate="cand",
        context=context,
        services=services,
    )

    # High similarity (0.9 > 0.5) passes embedding check, then LLM says YES
    assert ok is True


def test_validate_candidate_query_rejects_when_llm_says_no():
    """Test validate_candidate_query rejects when LLM says NO even with high similarity."""
    cfg = AgentConfig(embedding_query_similarity_threshold=0.5)
    chains = {"query_filter": SimpleNamespace(invoke=lambda inputs: "NO")}  # LLM rejects

    context = _make_search_context(question_embedding=[1.0])
    services = _make_search_services(
        cfg=cfg,
        chains=chains,
        context_similarity=lambda a, b, c: 0.9,  # High similarity
    )

    ok = validate_candidate_query(
        candidate="cand",
        context=context,
        services=services,
    )

    # Embedding check passes, but LLM says NO, so should be False
    assert ok is False
