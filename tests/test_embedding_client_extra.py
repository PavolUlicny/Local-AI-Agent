from __future__ import annotations


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
