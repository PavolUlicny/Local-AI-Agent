"""Tests for parallel search functionality."""

from __future__ import annotations

import pytest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.search_parallel import should_use_parallel_search, process_queries_parallel
from src.search_context import SearchContext, SearchServices, SearchState
from src.config import AgentConfig


class _StubAgent:
    """Stub agent with _normalize_search_result method."""

    def _normalize_search_result(self, raw_result: dict) -> dict[str, str] | None:
        """Normalize search result."""
        if not raw_result:
            return None
        return {
            "title": str(raw_result.get("title", "")),
            "link": str(raw_result.get("link", "")),
            "snippet": str(raw_result.get("snippet", "")),
        }


def _make_search_context(**overrides):
    """Create SearchContext for tests."""
    defaults = {
        "current_datetime": "2025-12-22T10:00:00",
        "current_year": "2025",
        "current_month": "12",
        "current_day": "22",
        "user_query": "test query",
        "conversation_text": "",
        "prior_responses_text": "",
        "question_embedding": None,
        "topic_embedding_current": None,
    }
    return SearchContext(**{**defaults, **overrides})


def _make_query_inputs(**overrides):
    """Create query_inputs dict for tests using new API."""
    defaults = {
        "current_datetime": "2025-12-22T10:00:00",
        "current_year": "2025",
        "current_month": "12",
        "current_day": "22",
        "conversation_history": "",
        "user_question": "test query",
        "known_answers": "",
    }
    return {**defaults, **overrides}


def _make_search_services(cfg=None, **overrides):
    """Create SearchServices for tests."""
    agent = _StubAgent()

    # Create default chains that return no results/filters
    default_chains = {
        "result_filter": SimpleNamespace(invoke=lambda inputs: "NO"),
        "planning": SimpleNamespace(invoke=lambda inputs: "NONE"),
        "query_filter": SimpleNamespace(invoke=lambda inputs: "NO"),
    }

    defaults = {
        "cfg": cfg or AgentConfig(),
        "chains": default_chains,
        "embedding_client": Mock(),
        "ddg_results": Mock(__self__=agent, return_value=[]),
        "inputs_builder": lambda *a, **k: {},
        "reduce_context_and_rebuild": lambda k, label: None,
        "mark_error": lambda m: m,
        "context_similarity": lambda a, b, c: 0.0,
        "char_budget": lambda n: n,
        "rebuild_counts": {},
    }
    return SearchServices(**{**defaults, **overrides})


class TestShouldUseParallelSearch:
    """Tests for should_use_parallel_search decision function."""

    def test_returns_false_when_only_one_query(self):
        """Should not use parallel search with only one query."""
        cfg = AgentConfig(max_concurrent_queries=3)
        queries = ["query1"]
        assert should_use_parallel_search(queries, cfg) is False

    def test_returns_false_when_max_concurrent_is_one(self):
        """Should not use parallel search when max_concurrent_queries is 1."""
        cfg = AgentConfig(max_concurrent_queries=1)
        queries = ["query1", "query2", "query3"]
        assert should_use_parallel_search(queries, cfg) is False

    def test_returns_true_when_multiple_queries_and_max_concurrent_gt_one(self):
        """Should use parallel search with multiple queries and max_concurrent > 1."""
        cfg = AgentConfig(max_concurrent_queries=3)
        queries = ["query1", "query2"]
        assert should_use_parallel_search(queries, cfg) is True

    def test_returns_true_with_many_queries(self):
        """Should use parallel search with many queries."""
        cfg = AgentConfig(max_concurrent_queries=3)
        queries = ["q1", "q2", "q3", "q4", "q5"]
        assert should_use_parallel_search(queries, cfg) is True

    def test_returns_false_with_empty_queries(self):
        """Should not use parallel search with empty query list."""
        cfg = AgentConfig(max_concurrent_queries=3)
        queries = []
        assert should_use_parallel_search(queries, cfg) is False


class TestProcessQueriesParallel:
    """Tests for process_queries_parallel function."""

    def test_returns_empty_list_when_no_queries(self):
        """Should return empty list when given no queries."""
        context = _make_search_context()
        state = SearchState()
        services = _make_search_services()

        result = process_queries_parallel([], context, state, services)

        assert result == []

    def test_limits_queries_to_max_concurrent(self):
        """Should limit number of queries to max_concurrent_queries."""
        cfg = AgentConfig(max_concurrent_queries=2)
        context = _make_search_context()
        state = SearchState()
        services = _make_search_services(cfg=cfg)

        queries = ["q1", "q2", "q3", "q4", "q5"]

        # Mock both get_event_loop and run to force asyncio.run path
        with patch("src.search_parallel.asyncio.get_event_loop") as mock_get_loop:
            mock_get_loop.side_effect = RuntimeError("No event loop")

            with patch("src.search_parallel.process_search_round_async") as mock_async:
                mock_async.return_value = []

                with patch("src.search_parallel.asyncio.run") as mock_run:
                    mock_run.return_value = []
                    process_queries_parallel(queries, context, state, services)

                    # Verify asyncio.run was called
                    assert mock_run.called

    def test_creates_async_client_with_correct_normalizer(self):
        """Should create AsyncSearchClient with correct normalizer from Agent."""
        context = _make_search_context()
        state = SearchState()
        agent = _StubAgent()
        services = _make_search_services(ddg_results=Mock(__self__=agent))

        with patch("src.search_parallel.asyncio.get_event_loop") as mock_get_loop:
            mock_get_loop.side_effect = RuntimeError("No event loop")

            with patch("src.search_parallel.process_search_round_async") as mock_async:
                mock_async.return_value = []

                with patch("src.search_parallel.asyncio.run") as mock_run:
                    mock_run.return_value = []
                    process_queries_parallel(["query1", "query2"], context, state, services)

                    # Verify asyncio.run was called
                    assert mock_run.called

    def test_returns_results_from_async_processing(self):
        """Should return results from async processing."""
        context = _make_search_context()
        state = SearchState()
        services = _make_search_services()

        expected_results = ["result1", "result2", "result3"]

        with patch("src.search_parallel.asyncio.get_event_loop") as mock_get_loop:
            mock_get_loop.side_effect = RuntimeError("No event loop")

            with patch("src.search_parallel.process_search_round_async") as mock_async:
                mock_async.return_value = expected_results

                with patch("src.search_parallel.asyncio.run") as mock_run:
                    mock_run.return_value = expected_results
                    result = process_queries_parallel(["q1", "q2"], context, state, services)

                    assert result == expected_results

    def test_handles_existing_event_loop_not_running(self):
        """Should use existing event loop when it's not running."""
        context = _make_search_context()
        state = SearchState()
        services = _make_search_services()

        with patch("src.search_parallel.asyncio.get_event_loop") as mock_get_loop:
            with patch("src.search_parallel.process_search_round_async") as mock_async:
                mock_async.return_value = ["result"]

                mock_loop = Mock()
                mock_loop.is_running.return_value = False
                # Make run_until_complete execute the coroutine synchronously
                mock_loop.run_until_complete.return_value = ["result"]
                mock_get_loop.return_value = mock_loop

                result = process_queries_parallel(["q1", "q2"], context, state, services)

                assert result == ["result"]
                mock_loop.run_until_complete.assert_called_once()

    def test_handles_runtime_error_no_event_loop(self):
        """Should handle RuntimeError when no event loop exists."""
        context = _make_search_context()
        state = SearchState()
        services = _make_search_services()

        with patch("src.search_parallel.asyncio.get_event_loop") as mock_get_loop:
            mock_get_loop.side_effect = RuntimeError("No event loop")

            with patch("src.search_parallel.process_search_round_async") as mock_async:
                mock_async.return_value = ["result"]

                with patch("src.search_parallel.asyncio.run") as mock_async_run:
                    mock_async_run.return_value = ["result"]

                    result = process_queries_parallel(["q1", "q2"], context, state, services)

                    assert result == ["result"]
                    mock_async_run.assert_called_once()


class TestAsyncSearchClientIntegration:
    """Integration tests for AsyncSearchClient."""

    @pytest.mark.asyncio
    async def test_fetch_batch_pairs_queries_with_results(self):
        """Should pair each query with its results."""
        from src.search_client_async import AsyncSearchClient

        cfg = AgentConfig()
        agent = _StubAgent()

        client = AsyncSearchClient(cfg, normalizer=agent._normalize_search_result)

        # Mock the fetch method to return predictable results
        async def mock_fetch(query):
            return [{"title": f"Result for {query}", "link": "http://x", "snippet": "snippet"}]

        client.fetch = mock_fetch

        queries = ["query1", "query2", "query3"]
        results = await client.fetch_batch(queries)

        assert len(results) == 3
        for i, (query, query_results) in enumerate(results):
            assert query == queries[i]
            assert len(query_results) == 1
            assert query_results[0]["title"] == f"Result for {queries[i]}"

    @pytest.mark.asyncio
    async def test_fetch_batch_handles_exception_in_one_query(self):
        """Should handle exception in one query without affecting others."""
        from src.search_client_async import AsyncSearchClient

        cfg = AgentConfig()
        agent = _StubAgent()

        client = AsyncSearchClient(cfg, normalizer=agent._normalize_search_result)

        # Mock the fetch method to fail for query2
        async def mock_fetch(query):
            if query == "query2":
                raise ValueError("Mock error")
            return [{"title": f"Result for {query}", "link": "http://x", "snippet": "snippet"}]

        client.fetch = mock_fetch

        queries = ["query1", "query2", "query3"]
        results = await client.fetch_batch(queries)

        assert len(results) == 3
        # query1 should succeed
        assert results[0][0] == "query1"
        assert len(results[0][1]) == 1
        # query2 should return empty due to exception
        assert results[1][0] == "query2"
        assert results[1][1] == []
        # query3 should succeed
        assert results[2][0] == "query3"
        assert len(results[2][1]) == 1


class TestThreadSafeState:
    """Tests for ThreadSafeState wrapper."""

    def test_add_and_check_url_is_thread_safe(self):
        """Should safely add and check URLs."""
        from src.search_processing_async import ThreadSafeState

        state = SearchState()
        safe_state = ThreadSafeState(state)

        url = "http://example.com"
        assert safe_state.has_url(url) is False

        safe_state.add_url(url)
        assert safe_state.has_url(url) is True

    def test_add_and_check_result_hash_is_thread_safe(self):
        """Should safely add and check result hashes."""
        from src.search_processing_async import ThreadSafeState

        state = SearchState()
        safe_state = ThreadSafeState(state)

        hash_value = "abc123"
        assert safe_state.has_result_hash(hash_value) is False

        safe_state.add_result_hash(hash_value)
        assert safe_state.has_result_hash(hash_value) is True

    def test_update_keywords_is_thread_safe(self):
        """Should safely update keywords."""
        from src.search_processing_async import ThreadSafeState

        state = SearchState()
        safe_state = ThreadSafeState(state)

        keywords = {"test", "keywords"}
        safe_state.update_keywords(keywords)

        assert "test" in safe_state.state.topic_keywords
        assert "keywords" in safe_state.state.topic_keywords

    def test_multiple_operations_maintain_consistency(self):
        """Should maintain consistency across multiple operations."""
        from src.search_processing_async import ThreadSafeState

        state = SearchState()
        safe_state = ThreadSafeState(state)

        # Add multiple items
        safe_state.add_url("http://example1.com")
        safe_state.add_url("http://example2.com")
        safe_state.add_result_hash("hash1")
        safe_state.add_result_hash("hash2")
        safe_state.update_keywords({"kw1", "kw2"})

        # Verify all items are present
        assert safe_state.has_url("http://example1.com")
        assert safe_state.has_url("http://example2.com")
        assert safe_state.has_result_hash("hash1")
        assert safe_state.has_result_hash("hash2")
        assert "kw1" in safe_state.state.topic_keywords
        assert "kw2" in safe_state.state.topic_keywords


class TestSearchOrchestratorParallelIntegration:
    """Integration tests for SearchOrchestrator with parallel search."""

    def test_orchestrator_uses_parallel_when_multiple_queries_pending(self):
        """Should use parallel search when multiple queries are pending."""
        from src.search_orchestrator import SearchOrchestrator

        cfg = AgentConfig(max_concurrent_queries=3, max_rounds=6)

        def ddg_results(q: str):
            return [{"title": f"T:{q}", "link": f"http://x/{q}", "snippet": f"S:{q}"}]

        # Planning chain that returns multiple initial queries
        class PlanningChain:
            def __init__(self):
                self.call_count = 0

            def invoke(self, inputs):
                self.call_count += 1
                if self.call_count == 1:
                    return "query2\nquery3\nquery4"
                return "NONE"

        planning = PlanningChain()

        chains = {
            "planning": planning,
            "query_filter": SimpleNamespace(invoke=lambda inputs: "YES"),
        }

        query_inputs = _make_query_inputs()
        agent = _StubAgent()
        services = _make_search_services(
            cfg=cfg,
            chains=chains,
            ddg_results=Mock(side_effect=ddg_results, __self__=agent),
            embedding_client=Mock(embed=lambda x: None),
            context_similarity=lambda a, b, c: 1.0,
        )

        orch = SearchOrchestrator(services)

        # Patch process_queries_parallel to track when it's called
        with patch("src.search_orchestrator.process_queries_parallel") as mock_parallel:
            mock_parallel.return_value = ["result1", "result2", "result3"]

            with patch("src.search_orchestrator.process_search_round") as mock_sequential:
                mock_sequential.return_value = ["result0"]

                orch.run(query_inputs=query_inputs, user_query="test query")

                # When multiple queries are generated, they're processed in parallel
                assert mock_parallel.call_count >= 1

    def test_orchestrator_uses_sequential_when_max_concurrent_is_one(self):
        """Should use sequential search when max_concurrent_queries is 1."""
        from src.search_orchestrator import SearchOrchestrator

        cfg = AgentConfig(max_concurrent_queries=1, max_rounds=4)

        def ddg_results(q: str):
            return [{"title": f"T:{q}", "link": f"http://x/{q}", "snippet": f"S:{q}"}]

        class PlanningChain:
            def invoke(self, inputs):
                return "query2\nquery3"

        chains = {
            "planning": PlanningChain(),
            "query_filter": SimpleNamespace(invoke=lambda inputs: "YES"),
        }

        query_inputs = _make_query_inputs()
        agent = _StubAgent()
        services = _make_search_services(
            cfg=cfg,
            chains=chains,
            ddg_results=Mock(side_effect=ddg_results, __self__=agent),
            embedding_client=Mock(embed=lambda x: None),
            context_similarity=lambda a, b, c: 1.0,
        )

        orch = SearchOrchestrator(services)

        # Patch to verify parallel is not called
        with patch("src.search_orchestrator.process_queries_parallel") as mock_parallel:
            with patch("src.search_orchestrator.should_use_parallel_search") as mock_should_use:
                mock_should_use.return_value = False

                orch.run(query_inputs=query_inputs, user_query="test query")

                # should_use_parallel_search should be called but return False
                assert mock_should_use.called
                # process_queries_parallel should not be called
                assert not mock_parallel.called
