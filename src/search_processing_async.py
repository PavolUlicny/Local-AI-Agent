"""Async search result processing and parallel round execution utilities."""

from __future__ import annotations

import hashlib
import threading
from typing import List, TYPE_CHECKING

from src.keywords import extract_keywords
from src.text_utils import truncate_result
from src.url_utils import canonicalize_url
from src.search_validation import check_result_relevance

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.search_context import SearchContext, SearchServices, SearchState


class ThreadSafeState:
    """Thread-safe wrapper for SearchState mutations."""

    def __init__(self, state: "SearchState"):
        self.state = state
        self.lock = threading.Lock()

    def add_result_hash(self, hash_value: str) -> None:
        with self.lock:
            self.state.seen_result_hashes.add(hash_value)

    def add_url(self, url: str) -> None:
        with self.lock:
            self.state.seen_urls.add(url)

    def update_keywords(self, keywords: set[str]) -> None:
        with self.lock:
            self.state.topic_keywords.update(keywords)

    def has_result_hash(self, hash_value: str) -> bool:
        with self.lock:
            return hash_value in self.state.seen_result_hashes

    def has_url(self, url: str) -> bool:
        with self.lock:
            return url in self.state.seen_urls


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
    # Extract and clean fields
    title = str(result.get("title", "")).strip()
    link = str(result.get("link", "")).strip()
    snippet = str(result.get("snippet", "")).strip()

    # Skip empty results
    if not any([title, snippet, link]):
        return None, relevance_llm_checks

    # Deduplicate by URL
    norm_link = canonicalize_url(link) if link else ""
    if norm_link and safe_state.has_url(norm_link):
        return None, relevance_llm_checks

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

    # Deduplicate by content hash
    result_hash = hashlib.sha256(assembled.encode("utf-8", errors="ignore")).hexdigest()
    if safe_state.has_result_hash(result_hash):
        return None, relevance_llm_checks

    # Prepare for relevance check
    result_text = truncate_result(assembled)
    keywords_source = " ".join([part for part in [title, snippet] if part])

    # Check relevance using three-tier approach
    # Note: This is still synchronous because LLM calls are synchronous
    is_relevant_result, updated_llm_checks = check_result_relevance(
        result_text=result_text,
        keywords_source=keywords_source,
        current_query=current_query,
        topic_keywords=safe_state.state.topic_keywords,  # Read-only access is safe
        relevance_llm_checks=relevance_llm_checks,
        context=context,
        services=services,
    )

    if not is_relevant_result:
        return None, updated_llm_checks

    # Accept result: update tracking sets (thread-safe)
    safe_state.add_result_hash(result_hash)
    if norm_link:
        safe_state.add_url(norm_link)
    safe_state.update_keywords(extract_keywords(keywords_source))

    return result_text, updated_llm_checks


async def process_search_round_async(
    queries: List[str],
    context: "SearchContext",
    state: "SearchState",
    services: "SearchServices",
    async_client: any,  # AsyncSearchClient
) -> List[str]:
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
    accepted_results: List[str] = []
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
