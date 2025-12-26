"""Wrapper around Ollama embeddings with friendly logging."""

from __future__ import annotations

import logging
import time
from typing import Any, List

from langchain_ollama import OllamaEmbeddings

try:
    from ollama._types import ResponseError
except ImportError:  # pragma: no cover
    ResponseError = None

from . import model_utils as _model_utils_mod


class EmbeddingClient:
    """Lazily construct and reuse Ollama embedding clients.

    Can be used as a context manager to ensure proper cleanup:
        with EmbeddingClient(model_name) as client:
            embeddings = client.embed("text")
    """

    def __init__(self, model_name: str | None) -> None:
        self.model_name = (model_name or "").strip()
        self._client: OllamaEmbeddings | None = None
        self._warning_logged = False
        self._load_error_logged = False

    def __enter__(self) -> "EmbeddingClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - cleanup resources."""
        self.close()

    def embed(self, text: str) -> List[float] | None:
        normalized = (text or "").strip()
        if not normalized:
            return None
        client = self._ensure_client()
        if client is None:
            return None

        # Retry logic for transient Ollama errors
        max_retries = 3
        delay = 0.5

        for attempt in range(1, max_retries + 1):
            try:
                return list(client.embed_query(normalized))
            except (ConnectionError, TimeoutError, OSError) as exc:  # pragma: no cover - network
                # Network/connection errors
                self._log_embed_failure(exc)
                return None
            except Exception as exc:  # pragma: no cover - unexpected errors
                # Check if this is an Ollama ResponseError with status 500 (model load failure)
                if ResponseError is not None and isinstance(exc, ResponseError):
                    error_msg = str(exc)
                    if exc.status_code == 500 and "do load request" in error_msg:
                        # Model load failure - log once and suggest fix
                        if not self._load_error_logged:
                            logging.warning(
                                "Embedding model '%s' failed to load (attempt %d/%d). "
                                "This may indicate a corrupted model or resource issue. "
                                "Try: ollama rm %s && ollama pull %s",
                                self.model_name,
                                attempt,
                                max_retries,
                                self.model_name,
                                self.model_name,
                            )
                            self._load_error_logged = True

                        # Retry with exponential backoff for transient issues
                        if attempt < max_retries:
                            time.sleep(delay)
                            delay *= 2
                            continue
                        return None

                # Other unexpected errors (log at error level only on first occurrence)
                if not self._warning_logged:
                    logging.error("Unexpected embedding error (%s): %s", self.model_name, exc)
                    self._warning_logged = True
                return None

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
