from types import SimpleNamespace

from src.embedding_client import EmbeddingClient


def test_embed_returns_none_for_empty_text():
    c = EmbeddingClient("model")
    assert c.embed("") is None


def test_embed_uses_client_and_returns_list(monkeypatch):
    c = EmbeddingClient("mymodel")

    class FakeClient:
        def embed_query(self, text):
            return [0.1, 0.2]

    monkeypatch.setattr("src.embedding_client.OllamaEmbeddings", lambda model: FakeClient())
    out = c.embed("some text")
    assert out == [0.1, 0.2]


def test_build_client_handles_not_found(monkeypatch):
    c = EmbeddingClient("mymodel")

    # make OllamaEmbeddings raise not found
    def raise_exc(model):
        raise Exception("Model not found: mymodel")

    monkeypatch.setattr("src.embedding_client.OllamaEmbeddings", raise_exc)
    # monkeypatch model_utils handler to record call
    called = {}
    monkeypatch.setattr(
        "src.embedding_client._model_utils_mod",
        SimpleNamespace(handle_missing_model=lambda cb, kind, name: called.setdefault("missing", name)),
    )
    cli = c._build_client()
    assert cli is None
    assert called.get("missing") == "mymodel"


def test_log_embed_failure_calls_handler_on_not_found(monkeypatch):
    c = EmbeddingClient("mymodel")
    called = {}
    monkeypatch.setattr(
        "src.embedding_client._model_utils_mod",
        SimpleNamespace(handle_missing_model=lambda cb, kind, name: called.setdefault("missing", name)),
    )
    c._log_embed_failure(Exception("Not found"))
    assert called.get("missing") == "mymodel"
