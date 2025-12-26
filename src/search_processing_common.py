"""Shared result processing logic for both sync and async search operations."""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol, TYPE_CHECKING

from src.keywords import extract_keywords
from src.text_utils import truncate_result
from src.url_utils import canonicalize_url
from src.search_validation import check_result_relevance

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.search_context import SearchContext, SearchServices


class StateAccessor(Protocol):
    """Protocol for accessing search state (both sync and thread-safe variants).

    Provides both individual operations and atomic check-and-add operations
    to prevent race conditions in concurrent scenarios.
    """

    # Atomic operations (preferred for concurrent access)
    def check_and_add_url(self, url: str) -> bool:
        """Atomically check if URL exists and add if not.

        Returns:
            True if URL was added (new), False if it already existed
        """
        ...

    def check_and_add_result_hash(self, hash_value: str) -> bool:
        """Atomically check if result hash exists and add if not.

        Returns:
            True if hash was added (new), False if it already existed
        """
        ...

    # Individual query operations
    def has_url(self, url: str) -> bool:
        """Check if URL has been seen."""
        ...

    def has_result_hash(self, hash_value: str) -> bool:
        """Check if result hash has been seen."""
        ...

    def add_result_hash(self, hash_value: str) -> None:
        """Add a result hash to seen set."""
        ...

    def add_url(self, url: str) -> None:
        """Add a URL to seen set."""
        ...

    def update_keywords(self, keywords: set[str]) -> None:
        """Update topic keywords."""
        ...

    def get_topic_keywords(self) -> set[str]:
        """Get current topic keywords."""
        ...


def process_result_core(
    result: dict[str, str],
    current_query: str,
    relevance_llm_checks: int,
    context: "SearchContext",
    state_accessor: StateAccessor,
    services: "SearchServices",
) -> tuple[str | None, int]:
    """Core result processing logic shared by sync and async implementations.

    Args:
        result: Raw search result dict with 'title', 'link', 'snippet'
        current_query: The search query that produced this result
        relevance_llm_checks: Current count of LLM relevance checks
        context: Immutable search context (query, conversation, embeddings)
        state_accessor: State accessor (either direct SearchState or ThreadSafeState)
        services: Services bundle (config, clients, callbacks)

    Returns:
        Tuple of (result_text: str | None, updated_llm_checks: int)
        result_text is None if result should be skipped

    Side Effects:
        - Updates state via state_accessor if result is accepted
    """
    # Extract and clean fields (handle None explicitly to avoid "None" strings)
    title_raw = result.get("title")
    link_raw = result.get("link")
    snippet_raw = result.get("snippet")

    title = str(title_raw).strip() if title_raw is not None else ""
    link = str(link_raw).strip() if link_raw is not None else ""
    snippet = str(snippet_raw).strip() if snippet_raw is not None else ""

    # Skip empty results
    if not any([title, snippet, link]):
        return None, relevance_llm_checks

    # Deduplicate by URL (atomic check-and-add to prevent race conditions)
    norm_link = canonicalize_url(link) if link else ""
    if norm_link:
        # check_and_add_url returns False if URL already existed (duplicate)
        if not state_accessor.check_and_add_url(norm_link):
            logging.info(f"Skipping duplicate URL: {norm_link}")
            return None, relevance_llm_checks
        # URL was added successfully, continue processing

    # Assemble result text
    assembled = "\n".join(
        part
        for part in [
            (f"Title: {title}" if title else ""),
            (f"URL: {link}" if link else ""),
            (f"Snippet: {snippet}" if snippet else ""),
        ]
        if part
    )

    # Deduplicate by content hash (atomic check-and-add to prevent race conditions)
    result_hash = hashlib.sha256(assembled.encode("utf-8", errors="ignore")).hexdigest()
    if not state_accessor.check_and_add_result_hash(result_hash):
        # Hash already existed, this is a duplicate
        # Note: URL was already added above, but that's okay - URL deduplication will catch future instances
        logging.info(f"Skipping duplicate content (hash: {result_hash[:8]}...)")
        return None, relevance_llm_checks
    # Hash was added successfully, continue processing

    # Prepare for relevance check
    result_text = truncate_result(assembled)
    keywords_source = " ".join([part for part in [title, snippet] if part])

    # Check relevance using three-tier approach
    is_relevant_result, updated_llm_checks = check_result_relevance(
        result_text=result_text,
        keywords_source=keywords_source,
        current_query=current_query,
        topic_keywords=state_accessor.get_topic_keywords(),
        relevance_llm_checks=relevance_llm_checks,
        context=context,
        services=services,
    )

    if not is_relevant_result:
        return None, updated_llm_checks

    # Accept result: update keywords
    # Note: URL and hash were already added atomically during deduplication checks above
    state_accessor.update_keywords(extract_keywords(keywords_source))

    return result_text, updated_llm_checks


__all__ = ["StateAccessor", "process_result_core"]
