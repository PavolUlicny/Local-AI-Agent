from __future__ import annotations


def test_fetch_retries_on_timeout_and_notifies(monkeypatch):
    from src import search_client as SC

    # Dummy DDGS that always raises TimeoutException
    class DummyDDGS:
        def __init__(self, timeout=None):
            pass

        def text(self, *args, **kwargs):
            raise SC.TimeoutException("timeout")

        def close(self):
            pass

    monkeypatch.setattr(SC, "DDGS", DummyDDGS)

    # Patch sleep to avoid delays and record sleeps
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("time.sleep", fake_sleep)

    notified = []

    def notify(attempt, total, delay, reason):
        notified.append((attempt, total, delay, reason))

    cfg = type(
        "C",
        (),
        {
            "search_retries": 3,
            "search_timeout": 1,
            "ddg_region": None,
            "ddg_safesearch": None,
            "ddg_backend": None,
            "search_max_results": None,
        },
    )
    client = SC.SearchClient(cfg, normalizer=lambda e: e, notify_retry=notify)

    res = client.fetch("query")
    assert res == []
    # notify called for attempts 1..(retries-1)
    assert len(notified) == cfg.search_retries - 1
    # sleep should have been called same number of times
    assert len(sleeps) == cfg.search_retries - 1


def test_fetch_handles_ddgs_and_generic_errors(monkeypatch):
    from src import search_client as SC

    class DummyDDGS:
        def __init__(self, timeout=None):
            pass

        def text(self, *args, **kwargs):
            raise SC.DDGSException("ddgs error")

        def close(self):
            pass

    monkeypatch.setattr(SC, "DDGS", DummyDDGS)
    monkeypatch.setattr("time.sleep", lambda s: None)

    cfg = type(
        "C",
        (),
        {
            "search_retries": 2,
            "search_timeout": 1,
            "ddg_region": None,
            "ddg_safesearch": None,
            "ddg_backend": None,
            "search_max_results": None,
        },
    )
    client = SC.SearchClient(cfg, normalizer=lambda e: e, notify_retry=None)
    res = client.fetch("q")
    assert res == []
