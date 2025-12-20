from __future__ import annotations

from types import SimpleNamespace

from src.config import AgentConfig
from src.search_validation import check_result_relevance, validate_candidate_query


class _StubEmbeddingClient:
    def __init__(self, vec=None):
        self.vec = vec or [0.1, 0.2]

    def embed(self, text: str):
        return list(self.vec) if self.vec else None


def _make_common_kwargs(cfg=None, embedding_client=None, context_similarity=None):
    """Create common kwargs for validation functions."""
    return {
        "cfg": cfg or AgentConfig(),
        "embedding_client": embedding_client or _StubEmbeddingClient(),
        "context_similarity": context_similarity or (lambda a, b, c: 0.0),
        "inputs_builder": lambda *a, **k: {},
        "reduce_context_and_rebuild": lambda k, label: None,
        "rebuild_counts": {"relevance": 0, "query_filter": 0},
        "mark_error": lambda m: m,
    }


def test_check_result_relevance_embedding_threshold():
    """Test check_result_relevance passes when embedding similarity exceeds threshold."""
    cfg = AgentConfig(embedding_result_similarity_threshold=0.1)
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "NO")}

    common = _make_common_kwargs(
        cfg=cfg,
        context_similarity=lambda a, b, c: 0.5,  # High similarity
    )

    is_rel, checks = check_result_relevance(
        result_text="some text",
        keywords_source="some",
        topic_keywords=set(),
        question_embedding=[1.0],
        topic_embedding_current=None,
        current_query="q",
        chains=chains,
        conversation_text="c",
        user_query="q",
        prior_responses_text="p",
        current_datetime="d",
        current_year="y",
        current_month="m",
        current_day="dd",
        relevance_llm_checks=0,
        **common,
    )

    assert is_rel is True
    assert checks == 0


def test_check_result_relevance_llm_increments_check_count():
    """Test check_result_relevance increments LLM check counter when LLM is used."""

    class RF:
        def invoke(self, inputs):
            return "YES"

    chains = {"result_filter": RF()}
    common = _make_common_kwargs()

    is_rel, checks = check_result_relevance(
        result_text="txt",
        keywords_source="k",
        topic_keywords={"kw"},
        question_embedding=None,
        topic_embedding_current=None,
        current_query="q",
        chains=chains,
        conversation_text="c",
        user_query="q",
        prior_responses_text="p",
        current_datetime="d",
        current_year="y",
        current_month="m",
        current_day="dd",
        relevance_llm_checks=0,
        **common,
    )

    assert is_rel is True
    assert checks == 1


def test_check_result_relevance_respects_llm_check_limit():
    """Test check_result_relevance skips LLM when limit reached."""
    cfg = AgentConfig(max_relevance_llm_checks=2, embedding_result_similarity_threshold=0.9)
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "YES")}

    common = _make_common_kwargs(cfg=cfg)

    is_rel, checks = check_result_relevance(
        result_text="this text about dogs",
        keywords_source="dogs",
        topic_keywords={"cats", "birds"},  # Keywords don't match "dogs"
        question_embedding=[0.5],  # Provide embedding so tier 2 runs
        topic_embedding_current=None,
        current_query="q",
        chains=chains,
        conversation_text="c",
        user_query="q",
        prior_responses_text="p",
        current_datetime="d",
        current_year="y",
        current_month="m",
        current_day="dd",
        relevance_llm_checks=2,  # Already at limit
        **common,
    )

    # Should be False: tier 1 fails (keyword mismatch), tier 2 fails (low similarity 0.0 < 0.9), tier 3 skipped (limit reached)
    assert is_rel is False
    assert checks == 2  # Unchanged


def test_validate_candidate_query_embedding_and_filter():
    """Test validate_candidate_query rejects low similarity candidates."""
    chains = {"query_filter": SimpleNamespace(invoke=lambda inputs: "YES")}
    common = _make_common_kwargs()

    ok = validate_candidate_query(
        candidate="cand",
        chains=chains,
        question_embedding=[1.0],
        topic_embedding_current=None,
        user_query="q",
        conversation_text="c",
        current_datetime="d",
        current_year="y",
        current_month="m",
        current_day="dd",
        **common,
    )

    # similarity 0 < default threshold (0.5) => should be False
    assert ok is False


def test_validate_candidate_query_no_embeddings_still_filters():
    """Test validate_candidate_query works when embeddings are None."""

    class NoneEmbedding:
        def embed(self, text):
            return None

    chains = {"query_filter": SimpleNamespace(invoke=lambda inputs: "YES")}
    common = _make_common_kwargs(embedding_client=NoneEmbedding())

    ok = validate_candidate_query(
        candidate="cand",
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

    # Should skip embedding check and go straight to LLM filter
    assert ok is True
