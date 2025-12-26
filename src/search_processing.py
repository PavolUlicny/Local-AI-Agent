"""Search result processing and round execution utilities."""

from __future__ import annotations

from typing import List, TYPE_CHECKING

from src.search_processing_common import process_result_core

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.search_context import SearchContext, SearchServices, SearchState


class DirectStateAccessor:
    """Direct accessor for SearchState (non-thread-safe, for sequential processing).

    Implements the same interface as ThreadSafeState but without locking overhead
    since it's used in single-threaded contexts.
    """

    def __init__(self, state: "SearchState"):
        self.state = state

    # Atomic check-and-add operations (no locking needed for single-threaded)
    def check_and_add_url(self, url: str) -> bool:
        """Check if URL exists and add if not.

        Returns:
            True if URL was added (new), False if it already existed
        """
        if url in self.state.seen_urls:
            return False
        self.state.seen_urls.add(url)
        return True

    def check_and_add_result_hash(self, hash_value: str) -> bool:
        """Check if result hash exists and add if not.

        Returns:
            True if hash was added (new), False if it already existed
        """
        if hash_value in self.state.seen_result_hashes:
            return False
        self.state.seen_result_hashes.add(hash_value)
        return True

    # Individual query operations
    def has_url(self, url: str) -> bool:
        """Check if URL has been seen."""
        return url in self.state.seen_urls

    def has_result_hash(self, hash_value: str) -> bool:
        """Check if result hash has been seen."""
        return hash_value in self.state.seen_result_hashes

    def add_result_hash(self, hash_value: str) -> None:
        """Add a result hash to the seen set."""
        self.state.seen_result_hashes.add(hash_value)

    def add_url(self, url: str) -> None:
        """Add a URL to the seen set."""
        self.state.seen_urls.add(url)

    def update_keywords(self, keywords: set[str]) -> None:
        """Update topic keywords with new keywords."""
        self.state.topic_keywords.update(keywords)

    def get_topic_keywords(self) -> set[str]:
        """Get current topic keywords."""
        return self.state.topic_keywords


def process_search_result(
    result: dict[str, str],
    current_query: str,
    relevance_llm_checks: int,
    context: "SearchContext",
    state: "SearchState",
    services: "SearchServices",
) -> tuple[str | None, int]:
    """Process a single search result with deduplication and relevance filtering.

    Args:
        result: Raw search result dict with 'title', 'link', 'snippet'
        current_query: The search query that produced this result
        relevance_llm_checks: Current count of LLM relevance checks
        context: Immutable search context (query, conversation, embeddings)
        state: Mutable search state (seen_urls, seen_result_hashes, topic_keywords)
        services: Services bundle (config, clients, callbacks)

    Returns:
        Tuple of (result_text: str | None, updated_llm_checks: int)
        result_text is None if result should be skipped

    Side Effects:
        - Updates state.seen_result_hashes if result is accepted
        - Updates state.seen_urls if result is accepted
        - Updates state.topic_keywords if result is accepted
    """
    state_accessor = DirectStateAccessor(state)
    return process_result_core(
        result=result,
        current_query=current_query,
        relevance_llm_checks=relevance_llm_checks,
        context=context,
        state_accessor=state_accessor,
        services=services,
    )


def process_search_round(
    current_query: str,
    context: "SearchContext",
    state: "SearchState",
    services: "SearchServices",
) -> List[str]:
    """Execute one search round: fetch results and filter for relevance.

    Args:
        current_query: The search query to execute
        context: Immutable search context (query, conversation, embeddings)
        state: Mutable search state (seen_urls, seen_result_hashes, topic_keywords)
        services: Services bundle (config, clients, callbacks)

    Returns:
        List of accepted result texts from this search round

    Side Effects:
        - Updates state.seen_result_hashes, state.seen_urls, state.topic_keywords
    """
    results_list = services.ddg_results(current_query)
    accepted_results: List[str] = []
    relevance_llm_checks = 0

    for res in results_list or []:
        result_text, relevance_llm_checks = process_search_result(
            result=res,
            current_query=current_query,
            relevance_llm_checks=relevance_llm_checks,
            context=context,
            state=state,
            services=services,
        )
        if result_text is not None:
            accepted_results.append(result_text)

    return accepted_results


__all__ = ["process_search_result", "process_search_round"]
