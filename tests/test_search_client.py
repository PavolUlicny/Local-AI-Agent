from __future__ import annotations

from src.search_client import SearchClient
from src.config import AgentConfig
from ddgs.exceptions import TimeoutException


def test_fetch_success_calls_normalizer(monkeypatch):
    cfg = AgentConfig(search_retries=1)

    # fake DDGS that returns a raw result
    class FakeDDGS:
        def __init__(self, timeout=None):
            pass

        def text(self, query, **kwargs):
            return [{"title": "T", "snippet": "S", "href": "http://x"}]

    monkeypatch.setattr("src.search_client.DDGS", FakeDDGS)

    def normalizer(entry):
        return {"title": entry.get("title"), "link": entry.get("href"), "snippet": entry.get("snippet")}

    client = SearchClient(cfg, normalizer=normalizer)
    res = client.fetch("q")
    assert res and res[0]["title"] == "T"


def test_fetch_retries_on_timeout_and_notifies(monkeypatch):
    cfg = AgentConfig(search_retries=3, search_timeout=1)
    calls = {"count": 0}

    class FakeDDGS:
        def __init__(self, timeout=None):
            pass

        def text(self, query, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise TimeoutException("timeout")
            return [{"title": "T", "snippet": "S", "href": "http://x"}]

    monkeypatch.setattr("src.search_client.DDGS", FakeDDGS)

    notified = []

    def notify(attempt, total, delay, reason):
        notified.append((attempt, total, delay, reason))

    def normalizer(e):
        return {"title": e.get("title"), "link": e.get("href"), "snippet": e.get("snippet")}

    client = SearchClient(cfg, normalizer=normalizer, notify_retry=notify)
    res = client.fetch("q")
    assert res and notified, "Expected notify_retry to be called on timeout"
    assert calls["count"] >= 2


def test_fetch_all_fail_returns_empty_and_notifies(monkeypatch):
    from ddgs.exceptions import DDGSException

    cfg = AgentConfig(search_retries=2, search_timeout=1)

    class FakeDDGS:
        def __init__(self, timeout=None):
            pass

        def text(self, query, **kwargs):
            # Use DDGSException which will be retried (not generic Exception which breaks immediately)
            raise DDGSException("boom")

    monkeypatch.setattr("src.search_client.DDGS", FakeDDGS)

    notified = []

    def notify(attempt, total, delay, reason):
        notified.append((attempt, total, delay, reason))

    def normalizer(e):
        return None

    client = SearchClient(cfg, normalizer=normalizer, notify_retry=notify)
    res = client.fetch("q")
    assert res == []
    # notify called at least once for DDGSException (which is retried)
    assert notified


class TestResourceLeakFix:
    """Tests for resource leak prevention in SearchClient."""

    def test_client_cleanup_on_success(self, monkeypatch):
        """Test that client is properly closed on successful fetch."""
        cfg = AgentConfig(search_retries=1)
        closed_clients = []

        class FakeDDGS:
            def __init__(self, timeout=None):
                self.closed = False

            def text(self, query, **kwargs):
                return [{"title": "T", "snippet": "S", "href": "http://x"}]

            def close(self):
                self.closed = True
                closed_clients.append(self)

        monkeypatch.setattr("src.search_client.DDGS", FakeDDGS)

        def normalizer(e):
            return {"title": e.get("title"), "link": e.get("href"), "snippet": e.get("snippet")}

        client = SearchClient(cfg, normalizer=normalizer)
        res = client.fetch("q")

        # Should have results
        assert len(res) == 1

        # Client should have been closed
        assert len(closed_clients) == 1
        assert closed_clients[0].closed is True

    def test_client_cleanup_on_exception(self, monkeypatch):
        """Test that client is properly closed even when exception occurs."""
        from ddgs.exceptions import DDGSException

        cfg = AgentConfig(search_retries=1)
        closed_clients = []

        class FakeDDGS:
            def __init__(self, timeout=None):
                self.closed = False

            def text(self, query, **kwargs):
                raise DDGSException("boom")

            def close(self):
                self.closed = True
                closed_clients.append(self)

        monkeypatch.setattr("src.search_client.DDGS", FakeDDGS)

        client = SearchClient(cfg, normalizer=lambda e: None)
        res = client.fetch("q")

        # Should return empty on failure
        assert res == []

        # Client should have been closed even though exception occurred
        assert len(closed_clients) == 1
        assert closed_clients[0].closed is True

    def test_no_leak_on_repeated_failures(self, monkeypatch):
        """Test that all clients are cleaned up across multiple retry attempts."""
        from ddgs.exceptions import TimeoutException

        cfg = AgentConfig(search_retries=3)
        closed_clients = []

        class FakeDDGS:
            def __init__(self, timeout=None):
                self.closed = False

            def text(self, query, **kwargs):
                raise TimeoutException("timeout")

            def close(self):
                self.closed = True
                closed_clients.append(self)

        monkeypatch.setattr("src.search_client.DDGS", FakeDDGS)

        client = SearchClient(cfg, normalizer=lambda e: None)
        res = client.fetch("q")

        # Should return empty after all retries fail
        assert res == []

        # All 3 clients (one per retry) should have been closed
        assert len(closed_clients) == 3
        assert all(c.closed for c in closed_clients)

    def test_cleanup_handles_client_without_close_method(self, monkeypatch):
        """Test that cleanup works even if client doesn't have close method."""
        cfg = AgentConfig(search_retries=1)

        class FakeDDGSNoClose:
            """DDGS without close method."""

            def __init__(self, timeout=None):
                pass

            def text(self, query, **kwargs):
                return [{"title": "T", "snippet": "S", "href": "http://x"}]

            # No close() method

        monkeypatch.setattr("src.search_client.DDGS", FakeDDGSNoClose)

        def normalizer(e):
            return {"title": e.get("title"), "link": e.get("href"), "snippet": e.get("snippet")}

        client = SearchClient(cfg, normalizer=normalizer)

        # Should not crash even though client has no close()
        res = client.fetch("q")
        assert len(res) == 1

    def test_cleanup_handles_close_exception(self, monkeypatch, caplog):
        """Test that cleanup handles exceptions during client.close()."""
        import logging

        caplog.set_level(logging.DEBUG)
        cfg = AgentConfig(search_retries=1)

        class FakeDDGSBrokenClose:
            """DDGS with close() that raises exception."""

            def __init__(self, timeout=None):
                pass

            def text(self, query, **kwargs):
                return [{"title": "T", "snippet": "S", "href": "http://x"}]

            def close(self):
                raise RuntimeError("close failed!")

        monkeypatch.setattr("src.search_client.DDGS", FakeDDGSBrokenClose)

        def normalizer(e):
            return {"title": e.get("title"), "link": e.get("href"), "snippet": e.get("snippet")}

        client = SearchClient(cfg, normalizer=normalizer)

        # Should not crash even though close() raises exception
        res = client.fetch("q")
        assert len(res) == 1

        # Should have logged the close failure
        assert "Client close failed" in caplog.text

    def test_unexpected_exception_cleans_up_and_breaks(self, monkeypatch):
        """Test that unexpected exceptions (non-retryable) still clean up resources."""
        cfg = AgentConfig(search_retries=3)
        closed_clients = []

        class FakeDDGS:
            def __init__(self, timeout=None):
                self.closed = False

            def text(self, query, **kwargs):
                # Unexpected exception (not DDGSException or TimeoutException)
                raise ValueError("unexpected error")

            def close(self):
                self.closed = True
                closed_clients.append(self)

        monkeypatch.setattr("src.search_client.DDGS", FakeDDGS)

        client = SearchClient(cfg, normalizer=lambda e: None)
        res = client.fetch("q")

        # Should return empty
        assert res == []

        # Client should have been closed (only 1 attempt, no retry for unexpected errors)
        assert len(closed_clients) == 1
        assert closed_clients[0].closed is True
