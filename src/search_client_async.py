"""Async utilities for performing parallel DDGS searches with retries and normalization."""

from __future__ import annotations

import asyncio
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, TYPE_CHECKING

from ddgs import DDGS
from ddgs.exceptions import DDGSException, TimeoutException

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.config import AgentConfig

# Retry behavior constants
RETRY_JITTER_MAX = 0.2  # Maximum random jitter to add to retry delay (seconds)
RETRY_BACKOFF_MULTIPLIER = 1.75  # Exponential backoff multiplier for retry delays
RETRY_MAX_DELAY = 3.0  # Maximum delay between retries (seconds)

# Concurrency control constants
# Thread pool size: slightly higher than max concurrent queries to handle retries
THREAD_POOL_HEADROOM = 2  # Extra threads beyond max_concurrent_queries


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

        # Bounded thread pool for run_in_executor calls
        # Size adapts to config: max_concurrent_queries + headroom for retries
        max_workers = cfg.max_concurrent_queries + THREAD_POOL_HEADROOM
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ddgs")

        # Semaphore to limit concurrent async operations
        # Respects config setting to provide appropriate backpressure
        self._semaphore = asyncio.Semaphore(cfg.max_concurrent_queries)

    async def fetch(self, query: str) -> List[dict[str, str]]:
        """Fetch search results asynchronously with bounded concurrency.

        Since DDGS is synchronous, we run it in a bounded thread pool.
        The semaphore ensures we don't overwhelm the thread pool.
        """
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._executor, self._fetch_sync, query)

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
        """Fetch multiple queries in parallel with bounded concurrency.

        The semaphore in fetch() ensures we don't exceed MAX_CONCURRENT_FETCHES
        concurrent operations, even if many queries are batched together.

        Args:
            queries: List of search queries to execute

        Returns:
            List of (query, results) tuples in same order as input queries
        """
        # Create tasks for all queries
        # The semaphore in fetch() provides backpressure
        tasks = [self.fetch(query) for query in queries]

        # Execute all tasks concurrently (bounded by semaphore)
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Pair queries with their results
        output: list[tuple[str, list[dict[str, str]]]] = []
        for query, results in zip(queries, results_list):
            if isinstance(results, Exception):
                logging.error(f"Query '{query}' failed with exception: {results}")
                output.append((query, []))
            elif isinstance(results, list):
                output.append((query, results))

        return output

    def shutdown(self) -> None:
        """Shutdown the thread pool executor.

        Should be called when the client is no longer needed to ensure
        threads are properly cleaned up.
        """
        if hasattr(self, "_executor") and self._executor is not None:
            self._executor.shutdown(wait=True)

    def __del__(self) -> None:
        """Cleanup executor on deletion."""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore errors during cleanup


__all__ = ["AsyncSearchClient"]
