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
    cfg = AgentConfig(search_retries=2, search_timeout=1)

    class FakeDDGS:
        def __init__(self, timeout=None):
            pass

        def text(self, query, **kwargs):
            raise Exception("boom")

    monkeypatch.setattr("src.search_client.DDGS", FakeDDGS)

    notified = []

    def notify(attempt, total, delay, reason):
        notified.append((attempt, total, delay, reason))

    def normalizer(e):
        return None

    client = SearchClient(cfg, normalizer=normalizer, notify_retry=notify)
    res = client.fetch("q")
    assert res == []
    # notify called at least once
    assert notified
