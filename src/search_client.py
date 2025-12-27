"""Utilities for performing DDGS searches with retries and normalization."""

from __future__ import annotations

import random
import time
from typing import Any, Callable, List, TYPE_CHECKING, cast

from ddgs import DDGS

from .constants import RETRY_JITTER_MAX, RETRY_BACKOFF_MULTIPLIER, RETRY_MAX_DELAY
from .search_retry_utils import (
    handle_search_exception,
    log_final_failure,
    safe_close_client,
    should_notify_retry,
)

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.config import AgentConfig


class SearchClient:
    """Wrap DDGS calls with retry/backoff and result normalization.

    Can be used as a context manager to ensure proper cleanup:
        with SearchClient(cfg, normalizer=...) as client:
            results = client.fetch("query")
    """

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
        self._client: Any = None  # Not used by fetch() - only for context manager cleanup

    def __enter__(self) -> "SearchClient":
        """Context manager entry."""
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Context manager exit - cleanup resources."""
        self.close()

    def close(self) -> None:
        safe_close_client(self._client)
        self._client = None

    def fetch(self, query: str) -> List[dict[str, str]]:
        """Fetch search results with retry logic.

        Creates a fresh DDGS client for each attempt to avoid connection reuse issues.
        Properly cleans up resources even if errors occur during retry.

        Args:
            query: Search query string

        Returns:
            List of normalized search results, or empty list on failure
        """
        delay = 0.5
        for attempt in range(1, self.cfg.search_retries + 1):
            reason: Exception | None = None
            client = None  # Initialize to None for safe cleanup

            try:
                # Create fresh client for this attempt
                client = DDGS(timeout=cast(int | None, self.cfg.search_timeout))

                raw_results = client.text(
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

                # Success - return results (cleanup happens in finally)
                return results

            except Exception as exc:  # pragma: no cover - network/unexpected errors
                # Handle all exceptions and determine if should retry
                should_retry = handle_search_exception(exc, query, attempt, self.cfg.search_retries, delay)
                reason = exc
                if not should_retry:
                    # Don't retry on unexpected errors - break and let finally clean up
                    break
            finally:
                # Always clean up the client for this attempt
                safe_close_client(client)

            # Retry logic (only reached if exception was caught)
            if attempt < self.cfg.search_retries:
                if reason is not None:
                    should_notify_retry(attempt, self.cfg.search_retries, self._notify_retry, delay, reason)
                jitter = random.random() * RETRY_JITTER_MAX
                time.sleep(delay + jitter)
                delay = min(delay * RETRY_BACKOFF_MULTIPLIER, RETRY_MAX_DELAY)

        log_final_failure(query, self.cfg.search_retries)
        return []


__all__ = ["SearchClient"]
