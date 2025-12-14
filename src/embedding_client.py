"""Wrapper around Ollama embeddings with friendly logging."""

from __future__ import annotations

import importlib
import logging
from typing import List

from langchain_ollama import OllamaEmbeddings

try:
    _model_utils_mod = importlib.import_module("src.model_utils")
except ModuleNotFoundError:
    _model_utils_mod = importlib.import_module("model_utils")


class EmbeddingClient:
    """Lazily construct and reuse Ollama embedding clients."""

    def __init__(self, model_name: str | None) -> None:
        self.model_name = (model_name or "").strip()
        self._client: OllamaEmbeddings | None = None
        self._warning_logged = False

    def embed(self, text: str) -> List[float] | None:
        normalized = (text or "").strip()
        if not normalized:
            return None
        client = self._ensure_client()
        if client is None:
            return None
        try:
            return list(client.embed_query(normalized))
        except Exception as exc:  # pragma: no cover - network/model specific
            self._log_embed_failure(exc)
            return None

    def close(self) -> None:
        self._client = None

    def _ensure_client(self) -> OllamaEmbeddings | None:
        if not self.model_name:
            return None
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> OllamaEmbeddings | None:
        if not self.model_name:
            return None
        try:
            return OllamaEmbeddings(model=self.model_name)
        except Exception as exc:  # pragma: no cover - runtime specific
            # Prefer the centralized missing-model handler when the model is simply not found.
            msg = str(exc).lower()
            if "not found" in msg:
                _model_utils_mod.handle_missing_model(None, "Embedding", self.model_name)
            else:
                if not self._warning_logged:
                    logging.warning(
                        "Embedding model '%s' unavailable; semantic similarity checks are disabled (%s). "
                        "If you expect this model locally run 'ollama pull %s' or check your Ollama service.",
                        self.model_name,
                        exc,
                        self.model_name,
                    )
                    self._warning_logged = True
            return None

    def _log_embed_failure(self, exc: Exception) -> None:
        message = str(exc).lower()
        if "not found" in message:
            _model_utils_mod.handle_missing_model(None, "Embedding", self.model_name)
        else:
            logging.warning("Embedding generation failed (%s): %s", self.model_name, exc)


__all__ = ["EmbeddingClient"]
