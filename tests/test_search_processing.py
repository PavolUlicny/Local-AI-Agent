from __future__ import annotations

import hashlib
from types import SimpleNamespace

from src.config import AgentConfig
from src.search_context import SearchContext, SearchState, SearchServices
from src.search_processing import process_search_result, process_search_round


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
        "rebuild_counts": {"relevance": 0},
    }
    return SearchServices(**{**defaults, **overrides})


def test_process_search_result_accepts_and_updates_sets():
    """Test process_search_result accepts valid results and updates tracking sets."""
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "YES")}

    res = {"title": "T", "link": "http://x", "snippet": "S"}

    context = _make_search_context()
    state = _make_search_state()
    services = _make_search_services(chains=chains)

    result_text, checks = process_search_result(
        result=res,
        current_query="q",
        relevance_llm_checks=0,
        context=context,
        state=state,
        services=services,
    )

    assert result_text is not None
    assert state.seen_result_hashes
    assert state.seen_urls
    assert isinstance(state.topic_keywords, set)
    assert isinstance(checks, int)


def test_process_search_result_skips_duplicate_by_url():
    """Test process_search_result skips duplicate URLs."""
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "YES")}

    res = {"title": "T", "link": "http://x", "snippet": "S"}

    context = _make_search_context()
    state = _make_search_state(seen_urls={"http://x"})
    services = _make_search_services(chains=chains)

    result_text, checks = process_search_result(
        result=res,
        current_query="q",
        relevance_llm_checks=0,
        context=context,
        state=state,
        services=services,
    )

    assert result_text is None
    assert checks == 0


def test_process_search_result_skips_duplicate_by_hash():
    """Test process_search_result skips duplicate content hashes."""
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "YES")}

    res = {"title": "T", "link": "http://x", "snippet": "S"}
    assembled = "Title: T\nURL: http://x\nSnippet: S"
    result_hash = hashlib.sha256(assembled.encode("utf-8", errors="ignore")).hexdigest()

    context = _make_search_context()
    state = _make_search_state(seen_result_hashes={result_hash})
    services = _make_search_services(chains=chains)

    result_text, checks = process_search_result(
        result=res,
        current_query="q",
        relevance_llm_checks=0,
        context=context,
        state=state,
        services=services,
    )

    assert result_text is None
    assert checks == 0


def test_process_search_result_skips_empty_results():
    """Test process_search_result skips empty results."""
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "YES")}

    res = {"title": "", "link": "", "snippet": ""}

    context = _make_search_context()
    state = _make_search_state()
    services = _make_search_services(chains=chains)

    result_text, checks = process_search_result(
        result=res,
        current_query="q",
        relevance_llm_checks=0,
        context=context,
        state=state,
        services=services,
    )

    assert result_text is None
    assert checks == 0


def test_process_search_result_extracts_keywords():
    """Test process_search_result extracts keywords from accepted results."""
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "YES")}

    res = {"title": "dogs cats", "link": "http://x", "snippet": "birds"}

    context = _make_search_context()
    state = _make_search_state()
    services = _make_search_services(chains=chains)

    result_text, checks = process_search_result(
        result=res,
        current_query="q",
        relevance_llm_checks=0,
        context=context,
        state=state,
        services=services,
    )

    assert result_text is not None
    # Keywords should be extracted from title + snippet
    assert "dogs" in state.topic_keywords or "cats" in state.topic_keywords or "birds" in state.topic_keywords


def test_process_search_result_skips_irrelevant():
    """Test process_search_result skips results deemed irrelevant."""
    cfg = AgentConfig(embedding_result_similarity_threshold=0.9)  # High threshold to force LLM check
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "NO")}

    res = {"title": "about dogs", "link": "http://x", "snippet": "canines"}

    context = _make_search_context()
    state = _make_search_state(topic_keywords={"cats", "birds"})  # Different keywords
    services = _make_search_services(cfg=cfg, chains=chains)

    result_text, checks = process_search_result(
        result=res,
        current_query="q",
        relevance_llm_checks=0,
        context=context,
        state=state,
        services=services,
    )

    assert result_text is None
    # LLM check should have been performed
    assert checks == 1  # One LLM check performed


def test_process_search_result_respects_max_llm_checks():
    """Test process_search_result respects max LLM relevance check limit."""
    cfg = AgentConfig(max_relevance_llm_checks=0, embedding_result_similarity_threshold=0.9)
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "YES")}

    res = {"title": "about dogs", "link": "http://x", "snippet": "canines"}

    context = _make_search_context()
    state = _make_search_state(topic_keywords={"cats", "birds"})  # Different keywords
    services = _make_search_services(cfg=cfg, chains=chains)

    result_text, checks = process_search_result(
        result=res,
        current_query="q",
        relevance_llm_checks=0,
        context=context,
        state=state,
        services=services,
    )

    # Should skip: keyword mismatch, low similarity, LLM limit reached
    assert result_text is None
    assert checks == 0  # No LLM check performed


def test_process_search_round_returns_accepted_results():
    """Test process_search_round returns list of accepted results."""
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "YES")}

    def ddg_results(q):
        return [
            {"title": "T1", "link": "http://x1", "snippet": "S1"},
            {"title": "T2", "link": "http://x2", "snippet": "S2"},
        ]

    context = _make_search_context()
    state = _make_search_state()
    services = _make_search_services(chains=chains, ddg_results=ddg_results)

    accepted = process_search_round(
        current_query="test",
        context=context,
        state=state,
        services=services,
    )

    assert len(accepted) == 2
    assert all(isinstance(r, str) for r in accepted)


def test_process_search_round_deduplicates_across_results():
    """Test process_search_round deduplicates results within a round."""
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "YES")}

    def ddg_results(q):
        return [
            {"title": "T", "link": "http://x", "snippet": "S"},
            {"title": "T", "link": "http://x", "snippet": "S"},  # Exact duplicate
        ]

    context = _make_search_context()
    state = _make_search_state()
    services = _make_search_services(chains=chains, ddg_results=ddg_results)

    accepted = process_search_round(
        current_query="test",
        context=context,
        state=state,
        services=services,
    )

    # Should only accept first result, deduplicate second
    assert len(accepted) == 1


def test_process_search_round_handles_empty_results():
    """Test process_search_round handles empty search results gracefully."""
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "YES")}

    def ddg_results(q):
        return None  # Returns None instead of list

    context = _make_search_context()
    state = _make_search_state()
    services = _make_search_services(chains=chains, ddg_results=ddg_results)

    accepted = process_search_round(
        current_query="test",
        context=context,
        state=state,
        services=services,
    )

    assert accepted == []


def test_process_search_round_filters_irrelevant():
    """Test process_search_round filters out irrelevant results."""
    cfg = AgentConfig(embedding_result_similarity_threshold=0.9)  # High threshold to force LLM check
    chains = {"result_filter": SimpleNamespace(invoke=lambda inputs: "NO")}

    def ddg_results(q):
        return [
            {"title": "dogs canines", "link": "http://x1", "snippet": "puppies"},
            {"title": "cats felines", "link": "http://x2", "snippet": "kittens"},
        ]

    context = _make_search_context()
    state = _make_search_state(topic_keywords={"birds", "fish"})  # Different keywords
    services = _make_search_services(cfg=cfg, chains=chains, ddg_results=ddg_results)

    accepted = process_search_round(
        current_query="test",
        context=context,
        state=state,
        services=services,
    )

    # All results should be filtered out
    assert accepted == []
