"""Async utilities for performing parallel DDGS searches with retries and normalization."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Callable, List, TYPE_CHECKING

from ddgs import DDGS
from ddgs.exceptions import DDGSException, TimeoutException

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.config import AgentConfig

# Retry behavior constants
RETRY_JITTER_MAX = 0.2  # Maximum random jitter to add to retry delay (seconds)
RETRY_BACKOFF_MULTIPLIER = 1.75  # Exponential backoff multiplier for retry delays
RETRY_MAX_DELAY = 3.0  # Maximum delay between retries (seconds)


class AsyncSearchClient:
    """Async wrapper for DDGS calls with retry/backoff and result normalization."""

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

    async def fetch(self, query: str) -> List[dict[str, str]]:
        """Fetch search results asynchronously.

        Since DDGS is synchronous, we run it in a thread pool to avoid blocking.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_sync, query)

    def _fetch_sync(self, query: str) -> List[dict[str, str]]:
        """Synchronous fetch implementation (runs in thread pool)."""
        delay = 0.5
        for attempt in range(1, self.cfg.search_retries + 1):
            reason: Exception | None = None
            client = None
            try:
                client = DDGS(timeout=int(self.cfg.search_timeout) if self.cfg.search_timeout else None)
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
                if client is not None:
                    self._safe_close(client)

            if attempt < self.cfg.search_retries:
                if reason is not None and self._notify_retry is not None:
                    self._notify_retry(attempt, self.cfg.search_retries, delay, reason)
                jitter = random.random() * RETRY_JITTER_MAX
                import time
                time.sleep(delay + jitter)
                delay = min(delay * RETRY_BACKOFF_MULTIPLIER, RETRY_MAX_DELAY)

        logging.warning("Search failed after %s attempts for '%s'.", self.cfg.search_retries, query)
        return []

    @staticmethod
    def _safe_close(client: Any) -> None:
        """Safely close DDGS client."""
        if client is None:
            return
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:
                logging.debug("Client close failed: %s", exc)

    async def fetch_batch(self, queries: List[str]) -> List[tuple[str, List[dict[str, str]]]]:
        """Fetch multiple queries in parallel.

        Args:
            queries: List of search queries to execute

        Returns:
            List of (query, results) tuples in same order as input queries
        """
        # Create tasks for all queries
        tasks = [self.fetch(query) for query in queries]

        # Execute all tasks concurrently
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Pair queries with their results
        output = []
        for query, results in zip(queries, results_list):
            if isinstance(results, Exception):
                logging.error(f"Query '{query}' failed with exception: {results}")
                output.append((query, []))
            else:
                output.append((query, results))

        return output


__all__ = ["AsyncSearchClient"]
