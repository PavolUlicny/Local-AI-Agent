from __future__ import annotations


def test_build_client_not_found_calls_handle_missing_model(monkeypatch):
    from src import embedding_client as EC

    class FailCtor:
        def __init__(self, *a, **k):
            raise Exception("Model not found: foo")

    monkeypatch.setattr(EC, "OllamaEmbeddings", FailCtor)

    called = {}

    def fake_handle_missing_model(*args, **kwargs):
        called["ok"] = True

    monkeypatch.setattr(EC._model_utils_mod, "handle_missing_model", fake_handle_missing_model)

    c = EC.EmbeddingClient("foo")
    assert c.embed("text") is None
    assert called.get("ok") is True


def test_embed_returns_none_and_logs_on_exception(monkeypatch, caplog):
    from src import embedding_client as EC

    class Client:
        def __init__(self, *a, **k):
            pass

        def embed_query(self, _):
            raise Exception("boom")

    monkeypatch.setattr(EC, "OllamaEmbeddings", Client)
    c = EC.EmbeddingClient("foo")
    caplog.clear()
    caplog.set_level("WARNING")
    assert c.embed("hello") is None
    # should have logged a warning about embedding generation failed
    assert any(
        "Embedding generation failed" in rec.message or "failed" in rec.message.lower() for rec in caplog.records
    )
