"""Async search result processing and parallel round execution utilities."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from src.search_processing_common import process_result_core

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.search_context import SearchContext, SearchServices, SearchState
    from src.search_client_async import AsyncSearchClient


class ThreadSafeState:
    """Thread-safe wrapper for SearchState mutations.

    Provides atomic check-and-add operations to prevent race conditions
    in concurrent search result processing.
    """

    def __init__(self, state: "SearchState"):
        self.state = state
        self.lock = threading.Lock()

    # Atomic check-and-add operations (race-condition safe)
    def check_and_add_url(self, url: str) -> bool:
        """Atomically check if URL exists and add if not.

        Args:
            url: URL to check and add

        Returns:
            True if URL was added (new), False if it already existed

        Thread-safe: Entire check-and-add operation is atomic.
        """
        with self.lock:
            if url in self.state.seen_urls:
                return False
            self.state.seen_urls.add(url)
            return True

    def check_and_add_result_hash(self, hash_value: str) -> bool:
        """Atomically check if result hash exists and add if not.

        Args:
            hash_value: Hash value to check and add

        Returns:
            True if hash was added (new), False if it already existed

        Thread-safe: Entire check-and-add operation is atomic.
        """
        with self.lock:
            if hash_value in self.state.seen_result_hashes:
                return False
            self.state.seen_result_hashes.add(hash_value)
            return True

    # Individual operations
    def add_result_hash(self, hash_value: str) -> None:
        """Add a result hash to the seen set.

        Note: Prefer check_and_add_result_hash() for atomic check-and-add.
        """
        with self.lock:
            self.state.seen_result_hashes.add(hash_value)

    def add_url(self, url: str) -> None:
        """Add a URL to the seen set.

        Note: Prefer check_and_add_url() for atomic check-and-add.
        """
        with self.lock:
            self.state.seen_urls.add(url)

    def update_keywords(self, keywords: set[str]) -> None:
        """Update topic keywords with new keywords."""
        with self.lock:
            self.state.topic_keywords.update(keywords)

    def has_result_hash(self, hash_value: str) -> bool:
        """Check if result hash exists.

        Note: If you need to add after checking, use check_and_add_result_hash()
        to avoid race conditions.
        """
        with self.lock:
            return hash_value in self.state.seen_result_hashes

    def has_url(self, url: str) -> bool:
        """Check if URL exists.

        Note: If you need to add after checking, use check_and_add_url()
        to avoid race conditions.
        """
        with self.lock:
            return url in self.state.seen_urls

    def get_topic_keywords(self) -> set[str]:
        """Get a copy of current topic keywords."""
        with self.lock:
            return self.state.topic_keywords.copy()


def process_search_result(
    result: dict[str, str],
    current_query: str,
    relevance_llm_checks: int,
    context: "SearchContext",
    safe_state: ThreadSafeState,
    services: "SearchServices",
) -> tuple[str | None, int]:
    """Process a single search result with deduplication and relevance filtering.

    This is the same as the original, but uses ThreadSafeState for concurrent access.

    Args:
        result: Raw search result dict with 'title', 'link', 'snippet'
        current_query: The search query that produced this result
        relevance_llm_checks: Current count of LLM relevance checks
        context: Immutable search context (query, conversation, embeddings)
        safe_state: Thread-safe state wrapper
        services: Services bundle (config, clients, callbacks)

    Returns:
        Tuple of (result_text: str | None, updated_llm_checks: int)
        result_text is None if result should be skipped

    Side Effects:
        - Updates safe_state with seen hashes, URLs, and keywords
    """
    return process_result_core(
        result=result,
        current_query=current_query,
        relevance_llm_checks=relevance_llm_checks,
        context=context,
        state_accessor=safe_state,
        services=services,
    )


async def process_search_round_async(
    queries: list[str],
    context: "SearchContext",
    state: "SearchState",
    services: "SearchServices",
    async_client: "AsyncSearchClient",
) -> list[str]:
    """Execute one search round with parallel query execution.

    Args:
        queries: List of search queries to execute in parallel
        context: Immutable search context (query, conversation, embeddings)
        state: Mutable search state (seen_urls, seen_result_hashes, topic_keywords)
        services: Services bundle (config, clients, callbacks)
        async_client: AsyncSearchClient instance for parallel fetching

    Returns:
        List of accepted result texts from this search round

    Side Effects:
        - Updates state.seen_result_hashes, state.seen_urls, state.topic_keywords
    """
    # Fetch all queries in parallel
    query_results = await async_client.fetch_batch(queries)

    # Process results in parallel (each query's results processed sequentially,
    # but different queries processed concurrently)
    safe_state = ThreadSafeState(state)
    accepted_results: list[str] = []
    relevance_llm_checks = 0

    # Process results from each query
    for query, results_list in query_results:
        for res in results_list or []:
            result_text, relevance_llm_checks = process_search_result(
                result=res,
                current_query=query,
                relevance_llm_checks=relevance_llm_checks,
                context=context,
                safe_state=safe_state,
                services=services,
            )
            if result_text is not None:
                accepted_results.append(result_text)

    return accepted_results


__all__ = ["process_search_result", "process_search_round_async", "ThreadSafeState"]
