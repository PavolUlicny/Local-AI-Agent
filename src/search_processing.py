"""Search result processing and round execution utilities."""

from __future__ import annotations

import hashlib
from typing import List, TYPE_CHECKING

from src.keywords import extract_keywords
from src.text_utils import truncate_result
from src.url_utils import canonicalize_url
from src.search_validation import check_result_relevance

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.search_context import SearchContext, SearchServices, SearchState


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
    # Extract and clean fields
    title = str(result.get("title", "")).strip()
    link = str(result.get("link", "")).strip()
    snippet = str(result.get("snippet", "")).strip()

    # Skip empty results
    if not any([title, snippet, link]):
        return None, relevance_llm_checks

    # Deduplicate by URL
    norm_link = canonicalize_url(link) if link else ""
    if norm_link and norm_link in state.seen_urls:
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
    if result_hash in state.seen_result_hashes:
        return None, relevance_llm_checks

    # Prepare for relevance check
    result_text = truncate_result(assembled)
    keywords_source = " ".join([part for part in [title, snippet] if part])

    # Check relevance using three-tier approach
    is_relevant_result, updated_llm_checks = check_result_relevance(
        result_text=result_text,
        keywords_source=keywords_source,
        current_query=current_query,
        topic_keywords=state.topic_keywords,
        relevance_llm_checks=relevance_llm_checks,
        context=context,
        services=services,
    )

    if not is_relevant_result:
        return None, updated_llm_checks

    # Accept result: update tracking sets
    state.seen_result_hashes.add(result_hash)
    if norm_link:
        state.seen_urls.add(norm_link)
    state.topic_keywords.update(extract_keywords(keywords_source))

    return result_text, updated_llm_checks


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
