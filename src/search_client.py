"""Utilities for performing DDGS searches with retries and normalization."""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, List, TYPE_CHECKING, cast

from ddgs import DDGS
from ddgs.exceptions import DDGSException, TimeoutException

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.config import AgentConfig

# Retry behavior constants
RETRY_JITTER_MAX = 0.2  # Maximum random jitter to add to retry delay (seconds)
RETRY_BACKOFF_MULTIPLIER = 1.75  # Exponential backoff multiplier for retry delays
RETRY_MAX_DELAY = 3.0  # Maximum delay between retries (seconds)


class SearchClient:
    """Wrap DDGS calls with retry/backoff and result normalization."""

    def __init__(
        self,
        cfg: "AgentConfig",
        *,
        normalizer: Callable[[dict[str, Any]], dict[str, str] | None],
        notify_retry: Callable[[int, int, float, Exception], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self._normalize_result = normalizer
        self._notify_retry = notify_retry
        self._client: Any = self._new_closed_ddgs(cfg.search_timeout)

    def close(self) -> None:
        self._safe_close(self._client)
        self._client = None

    def fetch(self, query: str) -> List[dict[str, str]]:
        delay = 0.5
        for attempt in range(1, self.cfg.search_retries + 1):
            reason: Exception | None = None
            self._client = DDGS(timeout=cast(int | None, self.cfg.search_timeout))
            try:
                raw_results = self._client.text(
                    query,
                    region=self.cfg.ddg_region,
                    safesearch=self.cfg.ddg_safesearch,
                    backend=self.cfg.ddg_backend,
                    max_results=self.cfg.search_max_results,
                )
                results: List[dict[str, str]] = []
                for entry in raw_results or []:
                    normalized = self._normalize_result(entry)
                    if normalized:
                        results.append(normalized)
                return results
            except TimeoutException as exc:  # pragma: no cover - network
                logging.warning(
                    "Search timeout for '%s' (attempt %s/%s); retrying after %.1fs",
                    query,
                    attempt,
                    self.cfg.search_retries,
                    delay,
                )
                logging.debug("Timeout details: %s", exc)
                reason = exc
            except DDGSException as exc:  # pragma: no cover - network
                logging.warning(
                    "DDGS search error for '%s' (attempt %s/%s): %s; retrying after %.1fs",
                    query,
                    attempt,
                    self.cfg.search_retries,
                    exc,
                    delay,
                )
                reason = exc
            except Exception as exc:  # pragma: no cover - network
                logging.warning(
                    "Unexpected search error for '%s' (attempt %s/%s): %s; retrying after %.1fs",
                    query,
                    attempt,
                    self.cfg.search_retries,
                    exc,
                    delay,
                )
                reason = exc
            finally:
                self._safe_close(self._client)
                self._client = self._new_closed_ddgs(self.cfg.search_timeout)
            if attempt < self.cfg.search_retries:
                if reason is not None and self._notify_retry is not None:
                    self._notify_retry(attempt, self.cfg.search_retries, delay, reason)
                jitter = random.random() * RETRY_JITTER_MAX
                time.sleep(delay + jitter)
                delay = min(delay * RETRY_BACKOFF_MULTIPLIER, RETRY_MAX_DELAY)
        logging.warning("Search failed after %s attempts for '%s'.", self.cfg.search_retries, query)
        return []

    @staticmethod
    def _safe_close(client: Any) -> None:
        if client is None:
            return
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:
                logging.debug("Client close failed: %s", exc)

    @classmethod
    def _new_closed_ddgs(cls, timeout: float | None = None) -> Any:
        try:
            client = DDGS(timeout=cast(int | None, timeout))
            cls._safe_close(client)
            return client
        except Exception:
            return None


__all__ = ["SearchClient"]
