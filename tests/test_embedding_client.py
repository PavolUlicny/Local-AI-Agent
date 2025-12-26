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


def test_log_embed_failure_logs_warning_for_other_errors(caplog):
    """Test that non-'not found' errors are logged as warnings."""
    c = EmbeddingClient("mymodel")
    caplog.clear()
    caplog.set_level("WARNING")
    c._log_embed_failure(Exception("Connection timeout"))
    assert "Embedding generation failed" in caplog.text


def test_context_manager_usage(monkeypatch):
    """Test that EmbeddingClient works as a context manager."""

    class FakeClient:
        def embed_query(self, text):
            return [0.1, 0.2]

    monkeypatch.setattr("src.embedding_client.OllamaEmbeddings", lambda model: FakeClient())

    with EmbeddingClient("mymodel") as client:
        result = client.embed("test")
        assert result == [0.1, 0.2]

    # Client should be cleaned up after exit
    assert client._client is None


def test_empty_model_name_returns_none():
    """Test that empty or None model name returns None without creating client."""
    c = EmbeddingClient("")
    assert c.embed("text") is None

    c2 = EmbeddingClient(None)
    assert c2.embed("text") is None


def test_ensure_client_returns_none_for_empty_model():
    """Test _ensure_client returns None when model_name is empty."""
    c = EmbeddingClient("")
    assert c._ensure_client() is None

    c2 = EmbeddingClient(None)
    assert c2._ensure_client() is None


def test_build_client_returns_none_for_empty_model():
    """Test _build_client returns None when model_name is empty."""
    c = EmbeddingClient("")
    assert c._build_client() is None
