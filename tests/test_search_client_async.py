"""Tests for AsyncSearchClient."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.search_client_async import AsyncSearchClient
from src.config import AgentConfig
from ddgs.exceptions import DDGSException, TimeoutException


def _stub_normalizer(raw_result: dict) -> dict[str, str] | None:
    """Stub normalizer that returns normalized result."""
    if not raw_result:
        return None
    return {
        "title": str(raw_result.get("title", "")),
        "link": str(raw_result.get("link", "")),
        "snippet": str(raw_result.get("snippet", "")),
    }


class TestAsyncSearchClientInit:
    """Tests for AsyncSearchClient initialization."""

    def test_init_stores_config(self):
        """Should store config on initialization."""
        cfg = AgentConfig()
        client = AsyncSearchClient(cfg, normalizer=_stub_normalizer)

        assert client.cfg is cfg

    def test_init_stores_normalizer(self):
        """Should store normalizer function."""
        cfg = AgentConfig()
        normalizer = Mock()
        client = AsyncSearchClient(cfg, normalizer=normalizer)

        assert client._normalize_result is normalizer

    def test_init_stores_notify_retry_callback(self):
        """Should store notify_retry callback."""
        cfg = AgentConfig()
        notify_retry = Mock()
        client = AsyncSearchClient(cfg, normalizer=_stub_normalizer, notify_retry=notify_retry)

        assert client._notify_retry is notify_retry

    def test_init_allows_none_notify_retry(self):
        """Should allow None for notify_retry."""
        cfg = AgentConfig()
        client = AsyncSearchClient(cfg, normalizer=_stub_normalizer, notify_retry=None)

        assert client._notify_retry is None


class TestAsyncSearchClientFetchSync:
    """Tests for _fetch_sync method."""

    @patch("src.search_client_async.DDGS")
    def test_fetch_sync_returns_normalized_results(self, mock_ddgs_class):
        """Should return normalized results from DDGS."""
        cfg = AgentConfig()
        client = AsyncSearchClient(cfg, normalizer=_stub_normalizer)

        # Mock DDGS instance
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = [
            {"title": "Title1", "link": "http://link1", "snippet": "Snippet1"},
            {"title": "Title2", "link": "http://link2", "snippet": "Snippet2"},
        ]
        mock_ddgs_class.return_value = mock_ddgs

        results = client._fetch_sync("test query")

        assert len(results) == 2
        assert results[0]["title"] == "Title1"
        assert results[1]["title"] == "Title2"

    @patch("src.search_client_async.DDGS")
    def test_fetch_sync_uses_correct_ddgs_params(self, mock_ddgs_class):
        """Should pass correct parameters to DDGS."""
        cfg = AgentConfig(
            ddg_region="us-en",
            ddg_safesearch="moderate",
            ddg_backend="duckduckgo",
            search_max_results=10,
            search_timeout=15.0,
        )
        client = AsyncSearchClient(cfg, normalizer=_stub_normalizer)

        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = []
        mock_ddgs_class.return_value = mock_ddgs

        client._fetch_sync("test query")

        # Verify DDGS was instantiated with correct timeout
        mock_ddgs_class.assert_called_once_with(timeout=15)

        # Verify text() was called with correct parameters
        mock_ddgs.text.assert_called_once_with(
            "test query",
            region="us-en",
            safesearch="moderate",
            backend="duckduckgo",
            max_results=10,
        )

    @patch("src.search_client_async.DDGS")
    def test_fetch_sync_filters_out_none_normalized_results(self, mock_ddgs_class):
        """Should filter out results that normalizer returns None for."""
        cfg = AgentConfig()

        def selective_normalizer(raw):
            # Only normalize results with "good" in title
            if "good" in raw.get("title", ""):
                return _stub_normalizer(raw)
            return None

        client = AsyncSearchClient(cfg, normalizer=selective_normalizer)

        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = [
            {"title": "good result", "link": "http://1", "snippet": "s1"},
            {"title": "bad result", "link": "http://2", "snippet": "s2"},
            {"title": "another good result", "link": "http://3", "snippet": "s3"},
        ]
        mock_ddgs_class.return_value = mock_ddgs

        results = client._fetch_sync("test query")

        assert len(results) == 2
        assert "good result" in results[0]["title"]
        assert "another good result" in results[1]["title"]

    @patch("src.search_client_async.DDGS")
    @patch("time.sleep")
    def test_fetch_sync_retries_on_timeout(self, mock_sleep, mock_ddgs_class):
        """Should retry on TimeoutException."""
        cfg = AgentConfig(search_retries=3)
        client = AsyncSearchClient(cfg, normalizer=_stub_normalizer)

        mock_ddgs = MagicMock()
        # Fail first two times, succeed third time
        mock_ddgs.text.side_effect = [
            TimeoutException("Timeout 1"),
            TimeoutException("Timeout 2"),
            [{"title": "Success", "link": "http://x", "snippet": "s"}],
        ]
        mock_ddgs_class.return_value = mock_ddgs

        results = client._fetch_sync("test query")

        assert len(results) == 1
        assert results[0]["title"] == "Success"
        assert mock_ddgs.text.call_count == 3
        assert mock_sleep.call_count == 2  # Sleep between retries

    @patch("src.search_client_async.DDGS")
    @patch("time.sleep")
    def test_fetch_sync_retries_on_ddgs_exception(self, mock_sleep, mock_ddgs_class):
        """Should retry on DDGSException."""
        cfg = AgentConfig(search_retries=2)
        client = AsyncSearchClient(cfg, normalizer=_stub_normalizer)

        mock_ddgs = MagicMock()
        mock_ddgs.text.side_effect = [
            DDGSException("Rate limited"),
            [{"title": "Success", "link": "http://x", "snippet": "s"}],
        ]
        mock_ddgs_class.return_value = mock_ddgs

        results = client._fetch_sync("test query")

        assert len(results) == 1
        assert mock_ddgs.text.call_count == 2

    @patch("src.search_client_async.DDGS")
    @patch("time.sleep")
    def test_fetch_sync_returns_empty_after_max_retries(self, mock_sleep, mock_ddgs_class):
        """Should return empty list after exhausting retries."""
        cfg = AgentConfig(search_retries=2)
        client = AsyncSearchClient(cfg, normalizer=_stub_normalizer)

        mock_ddgs = MagicMock()
        mock_ddgs.text.side_effect = DDGSException("Always fails")
        mock_ddgs_class.return_value = mock_ddgs

        results = client._fetch_sync("test query")

        assert results == []
        assert mock_ddgs.text.call_count == 2

    @patch("src.search_client_async.DDGS")
    def test_fetch_sync_calls_notify_retry_on_retry(self, mock_ddgs_class):
        """Should call notify_retry callback when retrying."""
        cfg = AgentConfig(search_retries=2)
        notify_retry = Mock()
        client = AsyncSearchClient(cfg, normalizer=_stub_normalizer, notify_retry=notify_retry)

        mock_ddgs = MagicMock()
        mock_ddgs.text.side_effect = [
            TimeoutException("Timeout"),
            [{"title": "Success", "link": "http://x", "snippet": "s"}],
        ]
        mock_ddgs_class.return_value = mock_ddgs

        with patch("time.sleep"):
            client._fetch_sync("test query")

        # notify_retry should be called once (for the first retry)
        assert notify_retry.call_count == 1
        call_args = notify_retry.call_args[0]
        assert call_args[0] == 1  # attempt number
        assert call_args[1] == 2  # max retries
        assert isinstance(call_args[2], float)  # delay
        assert isinstance(call_args[3], TimeoutException)  # exception

    @patch("src.search_client_async.DDGS")
    @patch("time.sleep")
    def test_fetch_sync_uses_exponential_backoff(self, mock_sleep, mock_ddgs_class):
        """Should use exponential backoff for retries."""
        cfg = AgentConfig(search_retries=4)
        client = AsyncSearchClient(cfg, normalizer=_stub_normalizer)

        mock_ddgs = MagicMock()
        mock_ddgs.text.side_effect = [
            DDGSException("Fail"),
            DDGSException("Fail"),
            DDGSException("Fail"),
            [{"title": "Success", "link": "http://x", "snippet": "s"}],
        ]
        mock_ddgs_class.return_value = mock_ddgs

        client._fetch_sync("test query")

        # Check that sleep was called with increasing delays
        assert mock_sleep.call_count == 3
        sleep_delays = [call[0][0] for call in mock_sleep.call_args_list]
        # Delays should increase (accounting for jitter, check base delay increases)
        assert sleep_delays[1] > sleep_delays[0]
        assert sleep_delays[2] > sleep_delays[1]

    @patch("src.search_client_async.DDGS")
    def test_fetch_sync_closes_client_on_success(self, mock_ddgs_class):
        """Should close DDGS client after successful fetch."""
        cfg = AgentConfig()
        client = AsyncSearchClient(cfg, normalizer=_stub_normalizer)

        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = [{"title": "T", "link": "http://x", "snippet": "s"}]
        mock_ddgs.close = Mock()
        mock_ddgs_class.return_value = mock_ddgs

        client._fetch_sync("test query")

        mock_ddgs.close.assert_called_once()

    @patch("src.search_client_async.DDGS")
    @patch("time.sleep")
    def test_fetch_sync_closes_client_on_exception(self, mock_sleep, mock_ddgs_class):
        """Should close DDGS client even when exception occurs."""
        cfg = AgentConfig(search_retries=1)
        client = AsyncSearchClient(cfg, normalizer=_stub_normalizer)

        mock_ddgs = MagicMock()
        mock_ddgs.text.side_effect = DDGSException("Fail")
        mock_ddgs.close = Mock()
        mock_ddgs_class.return_value = mock_ddgs

        client._fetch_sync("test query")

        # close should be called once per attempt
        assert mock_ddgs.close.call_count == 1


class TestAsyncSearchClientFetch:
    """Tests for fetch async method."""

    @pytest.mark.asyncio
    async def test_fetch_runs_fetch_sync_in_executor(self):
        """Should run _fetch_sync in thread pool executor."""
        cfg = AgentConfig()
        client = AsyncSearchClient(cfg, normalizer=_stub_normalizer)

        # Mock _fetch_sync to return test data
        client._fetch_sync = Mock(return_value=[{"title": "T", "link": "http://x", "snippet": "s"}])

        results = await client.fetch("test query")

        assert len(results) == 1
        assert results[0]["title"] == "T"
        client._fetch_sync.assert_called_once_with("test query")


class TestAsyncSearchClientFetchBatch:
    """Tests for fetch_batch method."""

    @pytest.mark.asyncio
    async def test_fetch_batch_executes_all_queries(self):
        """Should execute all queries in the batch."""
        cfg = AgentConfig()
        client = AsyncSearchClient(cfg, normalizer=_stub_normalizer)

        # Mock fetch to return different results for each query
        async def mock_fetch(query):
            return [{"title": f"Result for {query}", "link": "http://x", "snippet": "s"}]

        client.fetch = mock_fetch

        queries = ["query1", "query2", "query3"]
        results = await client.fetch_batch(queries)

        assert len(results) == 3
        for i, (query, query_results) in enumerate(results):
            assert query == queries[i]
            assert query_results[0]["title"] == f"Result for {queries[i]}"

    @pytest.mark.asyncio
    async def test_fetch_batch_maintains_query_order(self):
        """Should maintain the order of queries in results."""
        cfg = AgentConfig()
        client = AsyncSearchClient(cfg, normalizer=_stub_normalizer)

        # Mock fetch with varying delays (simulated by different results)
        async def mock_fetch(query):
            return [{"title": query, "link": "http://x", "snippet": "s"}]

        client.fetch = mock_fetch

        queries = ["first", "second", "third"]
        results = await client.fetch_batch(queries)

        assert results[0][0] == "first"
        assert results[1][0] == "second"
        assert results[2][0] == "third"

    @pytest.mark.asyncio
    async def test_fetch_batch_handles_exception_gracefully(self):
        """Should handle exception in one query without failing entire batch."""
        cfg = AgentConfig()
        client = AsyncSearchClient(cfg, normalizer=_stub_normalizer)

        # Mock fetch to fail for specific query
        async def mock_fetch(query):
            if query == "fail":
                raise ValueError("Mock error")
            return [{"title": query, "link": "http://x", "snippet": "s"}]

        client.fetch = mock_fetch

        queries = ["good1", "fail", "good2"]
        results = await client.fetch_batch(queries)

        assert len(results) == 3
        # good1 should succeed
        assert results[0][0] == "good1"
        assert len(results[0][1]) == 1
        # fail should return empty list
        assert results[1][0] == "fail"
        assert results[1][1] == []
        # good2 should succeed
        assert results[2][0] == "good2"
        assert len(results[2][1]) == 1

    @pytest.mark.asyncio
    async def test_fetch_batch_with_empty_queries_list(self):
        """Should handle empty queries list."""
        cfg = AgentConfig()
        client = AsyncSearchClient(cfg, normalizer=_stub_normalizer)

        results = await client.fetch_batch([])

        assert results == []


class TestAsyncSearchClientSafeClose:
    """Tests for _safe_close static method."""

    def test_safe_close_calls_close_method(self):
        """Should call close method if it exists."""
        mock_client = Mock()
        mock_client.close = Mock()

        AsyncSearchClient._safe_close(mock_client)

        mock_client.close.assert_called_once()

    def test_safe_close_handles_none_client(self):
        """Should handle None client gracefully."""
        AsyncSearchClient._safe_close(None)
        # Should not raise exception

    def test_safe_close_handles_client_without_close_method(self):
        """Should handle client without close method."""
        mock_client = Mock(spec=[])  # No close method

        AsyncSearchClient._safe_close(mock_client)
        # Should not raise exception

    def test_safe_close_handles_close_exception(self):
        """Should handle exception during close."""
        mock_client = Mock()
        mock_client.close.side_effect = Exception("Close failed")

        AsyncSearchClient._safe_close(mock_client)
        # Should not raise exception
