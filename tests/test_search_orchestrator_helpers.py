from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.search_orchestrator import SearchOrchestrator, SearchAbort
from src.config import AgentConfig
from src.exceptions import ResponseError


class _StubEmbeddingClient:
    def __init__(self, vec=None):
        self.vec = vec or [0.1, 0.2]

    def embed(self, text: str):  # noqa: ANN001 - test stub
        return list(self.vec)


def _inputs_builder(*args, **kwargs):
    return {}


def make_orch(cfg=None, **overrides):
    cfg = cfg or AgentConfig()
    defaults = {
        "ddg_results": lambda q: [],
        "embedding_client": _StubEmbeddingClient(),
        "context_similarity": lambda a, b, c: 0.0,
        "inputs_builder": _inputs_builder,
        "reduce_context_and_rebuild": lambda k, label: None,
        "rebuild_counts": {"relevance": 0, "planning": 0, "query_filter": 0},
        "char_budget": lambda x: x,
        "mark_error": lambda m: m,
    }
    defaults.update(overrides)
    return SearchOrchestrator(cfg, **defaults)


def test_invoke_chain_with_retry_model_missing_raises():
    orch = make_orch()

    class BadChain:
        def invoke(self, inputs):
            raise ResponseError("Model Not Found: Robot model not found")

    with pytest.raises(SearchAbort):
        orch._invoke_chain_with_retry(
            chain=BadChain(),
            inputs={},
            rebuild_key="planning",
            rebuild_label="planning",
            raise_on_non_context_error=True,
        )


def test_invoke_chain_with_retry_context_retry_then_success():
    calls = {"reduced": 0}

    def reduce(key, label):
        calls["reduced"] += 1

    cfg = AgentConfig()
    orch = make_orch(cfg=cfg, reduce_context_and_rebuild=reduce)

    class Chain:
        def __init__(self):
            self.behavior = [ResponseError("Context length exceeded"), "OK"]

        def invoke(self, inputs):
            val = self.behavior.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

    out, success = orch._invoke_chain_with_retry(
        chain=Chain(), inputs={}, rebuild_key="planning", rebuild_label="planning"
    )

    assert success is True
    assert out == "OK"
    assert calls["reduced"] == 1


def test_process_search_result_accepts_and_updates_sets():
    orch = make_orch()
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "YES")}
    seen_hashes = set()
    seen_urls = set()
    kws = set()

    res = {"title": "T", "link": "http://x", "snippet": "S"}

    result_text, checks = orch._process_search_result(
        result=res,
        seen_result_hashes=seen_hashes,
        seen_urls=seen_urls,
        topic_keywords=kws,
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
    )

    assert result_text is not None
    assert seen_hashes
    assert seen_urls
    # topic keywords may or may not be updated depending on keyword extraction; ensure it's a set
    assert isinstance(kws, set)
    assert isinstance(checks, int)


def test_check_result_relevance_embedding_threshold():
    cfg = AgentConfig(embedding_result_similarity_threshold=0.1)
    # context_similarity returns high similarity
    orch = make_orch(cfg=cfg, context_similarity=lambda a, b, c: 0.5)
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "NO")}

    is_rel, checks = orch._check_result_relevance(
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
    )

    assert is_rel is True
    assert checks == 0


def test_generate_query_suggestions_raise_on_error():
    orch = make_orch()

    class BadChain:
        def invoke(self, inputs):
            raise ResponseError("Some server error")

    chains = {"planning": BadChain()}

    # raise_on_error True should raise SearchAbort
    with pytest.raises(SearchAbort):
        orch._generate_query_suggestions(
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
        )

    # raise_on_error False should return empty list (fallback)
    res = orch._generate_query_suggestions(
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
    )
    assert res == []


def test_validate_candidate_query_embedding_and_filter():
    # candidate_embedding low similarity -> skip
    orch = make_orch(context_similarity=lambda a, b, c: 0.0)
    chains = {"query_filter": SimpleNamespace(invoke=lambda inputs: "YES")}
    ok = orch._validate_candidate_query(
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
    )
    # similarity 0 < default threshold (0.5) => should be False
    assert ok is False


def test_invoke_chain_with_retry_non_context_error_behavior():
    orch = make_orch()

    class BadChain:
        def invoke(self, inputs):
            raise ResponseError("Some other error")

    # When raise_on_non_context_error=True should raise SearchAbort
    with pytest.raises(SearchAbort):
        orch._invoke_chain_with_retry(
            chain=BadChain(),
            inputs={},
            rebuild_key="planning",
            rebuild_label="planning",
            raise_on_non_context_error=True,
        )

    # When raise_on_non_context_error=False should return fallback and False
    out, success = orch._invoke_chain_with_retry(
        chain=BadChain(),
        inputs={},
        rebuild_key="planning",
        rebuild_label="planning",
        raise_on_non_context_error=False,
    )
    assert success is False
    assert out == "NO"


def test_check_result_relevance_llm_increments_check_count():
    # LLM returns YES and should increment the llm check counter
    orch = make_orch(context_similarity=lambda a, b, c: 0.0)

    class RF:
        def invoke(self, inputs):
            return "YES"

    chains = {"result_filter": RF()}
    is_rel, checks = orch._check_result_relevance(
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
    )
    assert is_rel is True
    assert checks == 1


def test_enqueue_validated_queries_respects_duplicates_and_max_rounds():
    orch = make_orch()
    chains = {"query_filter": SimpleNamespace(invoke=lambda inputs: "YES")}

    # Duplicates should be skipped; unique candidates enqueued
    pending = []
    seen = set()
    orch._enqueue_validated_queries(
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
    )
    assert pending == ["A", "B"]
    assert {x.lower() for x in seen} == {"a", "b"}

    # Respect max_rounds: when pending length >= max_rounds no new items are added
    pending2 = ["x"]
    seen2 = {"x"}
    orch._enqueue_validated_queries(
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
    )
    assert pending2 == ["x"]
