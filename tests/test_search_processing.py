from __future__ import annotations

import hashlib
from types import SimpleNamespace

from src.config import AgentConfig
from src.search_processing import process_search_result, process_search_round


class _StubEmbeddingClient:
    def __init__(self, vec=None):
        self.vec = vec or [0.1, 0.2]

    def embed(self, text: str):
        return list(self.vec) if self.vec else None


def _make_result_kwargs(cfg=None, embedding_client=None, context_similarity=None):
    """Create common kwargs for process_search_result."""
    return {
        "cfg": cfg or AgentConfig(),
        "embedding_client": embedding_client or _StubEmbeddingClient(),
        "context_similarity": context_similarity or (lambda a, b, c: 0.0),
        "inputs_builder": lambda *a, **k: {},
        "reduce_context_and_rebuild": lambda k, label: None,
        "rebuild_counts": {"relevance": 0},
        "mark_error": lambda m: m,
    }


def _make_round_kwargs(cfg=None, ddg_results=None, embedding_client=None, context_similarity=None):
    """Create common kwargs for process_search_round."""
    return {
        "cfg": cfg or AgentConfig(),
        "ddg_results": ddg_results or (lambda q: []),
        "embedding_client": embedding_client or _StubEmbeddingClient(),
        "context_similarity": context_similarity or (lambda a, b, c: 0.0),
        "inputs_builder": lambda *a, **k: {},
        "reduce_context_and_rebuild": lambda k, label: None,
        "rebuild_counts": {"relevance": 0},
        "mark_error": lambda m: m,
    }


def test_process_search_result_accepts_and_updates_sets():
    """Test process_search_result accepts valid results and updates tracking sets."""
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "YES")}
    seen_hashes = set()
    seen_urls = set()
    kws = set()

    res = {"title": "T", "link": "http://x", "snippet": "S"}

    common = _make_result_kwargs()

    result_text, checks = process_search_result(
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
        **common,
    )

    assert result_text is not None
    assert seen_hashes
    assert seen_urls
    assert isinstance(kws, set)
    assert isinstance(checks, int)


def test_process_search_result_skips_duplicate_by_url():
    """Test process_search_result skips duplicate URLs."""
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "YES")}
    seen_urls = {"http://x"}

    res = {"title": "T", "link": "http://x", "snippet": "S"}

    common = _make_result_kwargs()

    result_text, checks = process_search_result(
        result=res,
        seen_result_hashes=set(),
        seen_urls=seen_urls,
        topic_keywords=set(),
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

    assert result_text is None


def test_process_search_result_skips_duplicate_by_hash():
    """Test process_search_result skips duplicate content hashes."""
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "YES")}

    # Pre-calculate hash
    assembled = "Title: T\nURL: http://x\nSnippet: S"
    h = hashlib.sha256(assembled.encode("utf-8", errors="ignore")).hexdigest()
    seen_hashes = {h}

    res = {"title": "T", "link": "http://x", "snippet": "S"}

    common = _make_result_kwargs()

    result_text, checks = process_search_result(
        result=res,
        seen_result_hashes=seen_hashes,
        seen_urls=set(),
        topic_keywords=set(),
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

    assert result_text is None


def test_process_search_result_skips_empty_results():
    """Test process_search_result skips results with no content."""
    chains = {}

    res = {"title": "", "link": "", "snippet": ""}

    common = _make_result_kwargs()

    result_text, checks = process_search_result(
        result=res,
        seen_result_hashes=set(),
        seen_urls=set(),
        topic_keywords=set(),
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

    assert result_text is None


def test_process_search_round_returns_empty_when_no_ddg_results():
    """Test process_search_round with no results from DDG."""
    chains = {}

    common = _make_round_kwargs(ddg_results=lambda q: None)

    results = process_search_round(
        current_query="test",
        chains=chains,
        seen_result_hashes=set(),
        seen_urls=set(),
        topic_keywords=set(),
        question_embedding=None,
        topic_embedding_current=None,
        user_query="q",
        conversation_text="c",
        prior_responses_text="p",
        current_datetime="d",
        current_year="y",
        current_month="m",
        current_day="dd",
        **common,
    )

    assert results == []


def test_process_search_round_filters_irrelevant_results():
    """Test process_search_round filters out irrelevant results."""

    def ddg(q):
        return [
            {"title": "T1", "link": "http://x1", "snippet": "S1"},
            {"title": "T2", "link": "http://x2", "snippet": "S2"},
        ]

    # All results get filtered out by the filter returning NO
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "NO")}

    common = _make_round_kwargs(ddg_results=ddg)

    results = process_search_round(
        current_query="test",
        chains=chains,
        seen_result_hashes=set(),
        seen_urls=set(),
        topic_keywords={"kw"},  # Not empty so LLM is consulted
        question_embedding=None,
        topic_embedding_current=None,
        user_query="q",
        conversation_text="c",
        prior_responses_text="p",
        current_datetime="d",
        current_year="y",
        current_month="m",
        current_day="dd",
        **common,
    )

    assert results == []


def test_process_search_result_extracts_and_adds_keywords():
    """Test process_search_result extracts keywords from accepted results."""
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "YES")}
    topic_keywords = set()

    res = {"title": "Machine Learning Tutorial", "link": "http://x", "snippet": "Learn about neural networks"}

    common = _make_result_kwargs()

    result_text, checks = process_search_result(
        result=res,
        seen_result_hashes=set(),
        seen_urls=set(),
        topic_keywords=topic_keywords,
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

    assert result_text is not None
    # Keywords should have been extracted and added
    assert len(topic_keywords) > 0


def test_process_search_round_handles_empty_results_list():
    """Test process_search_round handles empty list from DDG gracefully."""
    chains = {}

    common = _make_round_kwargs(ddg_results=lambda q: [])  # Empty list

    results = process_search_round(
        current_query="test",
        chains=chains,
        seen_result_hashes=set(),
        seen_urls=set(),
        topic_keywords=set(),
        question_embedding=None,
        topic_embedding_current=None,
        user_query="q",
        conversation_text="c",
        prior_responses_text="p",
        current_datetime="d",
        current_year="y",
        current_month="m",
        current_day="dd",
        **common,
    )

    assert results == []
