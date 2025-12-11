"""Wrapper around Ollama embeddings with friendly logging."""

from __future__ import annotations

import logging
from typing import List

from langchain_ollama import OllamaEmbeddings


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
            if not self._warning_logged:
                logging.warning(
                    "Embedding model '%s' unavailable; semantic similarity checks are disabled (%s)",
                    self.model_name,
                    exc,
                )
                self._warning_logged = True
            return None

    def _log_embed_failure(self, exc: Exception) -> None:
        message = str(exc).lower()
        if "not found" in message:
            logging.warning(
                "Embedding model '%s' not found. Run 'ollama pull %s' to enable semantic topic tracking.",
                self.model_name,
                self.model_name,
            )
        else:
            logging.warning("Embedding generation failed (%s): %s", self.model_name, exc)


__all__ = ["EmbeddingClient"]
