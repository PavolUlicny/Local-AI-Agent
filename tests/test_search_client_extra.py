from __future__ import annotations


def test_fetch_handles_ddgs_and_generic_errors(monkeypatch):
    """Ensure DDGSException and generic exceptions produce empty results after retries."""
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
