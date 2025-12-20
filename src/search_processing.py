"""Search result processing and round execution utilities."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, List, Set, TYPE_CHECKING

from src.keywords import extract_keywords
from src.text_utils import truncate_result
from src.url_utils import canonicalize_url
from src.search_validation import check_result_relevance

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.config import AgentConfig


def process_search_result(
    *,
    result: dict[str, str],
    seen_result_hashes: Set[str],
    seen_urls: Set[str],
    topic_keywords: Set[str],
    question_embedding: List[float] | None,
    topic_embedding_current: List[float] | None,
    current_query: str,
    chains: dict[str, Any],
    conversation_text: str,
    user_query: str,
    prior_responses_text: str,
    current_datetime: str,
    current_year: str,
    current_month: str,
    current_day: str,
    relevance_llm_checks: int,
    cfg: "AgentConfig",
    embedding_client: Any,
    context_similarity: Callable[[List[float] | None, List[float] | None, List[float] | None], float],
    inputs_builder: Callable[..., dict[str, Any]],
    reduce_context_and_rebuild: Callable[[str, str], None],
    rebuild_counts: dict[str, int],
    mark_error: Callable[[str], str],
) -> tuple[str | None, int]:
    """Process a single search result with deduplication and relevance filtering.

    Args:
        result: Raw search result dict with 'title', 'link', 'snippet'
        seen_result_hashes: Set of content hashes for deduplication
        seen_urls: Set of canonicalized URLs for deduplication
        topic_keywords: Current topic keywords (mutated if result accepted)
        question_embedding: User question embedding
        topic_embedding_current: Current topic embedding
        current_query: The search query that produced this result
        chains: Dictionary of LLM chains
        conversation_text: Conversation context
        user_query: Original user query
        prior_responses_text: Prior responses
        current_datetime: Current datetime string
        current_year: Current year string
        current_month: Current month string
        current_day: Current day string
        relevance_llm_checks: Current count of LLM relevance checks
        cfg: Agent configuration
        embedding_client: Client for generating embeddings
        context_similarity: Function to compute similarity
        inputs_builder: Function to build chain inputs
        reduce_context_and_rebuild: Callback to reduce context
        rebuild_counts: Dictionary tracking rebuild attempts
        mark_error: Callback to mark errors

    Returns:
        Tuple of (result_text: str | None, updated_llm_checks: int)
        result_text is None if result should be skipped

    Side Effects:
        - Updates seen_result_hashes if result is accepted
        - Updates seen_urls if result is accepted
        - Updates topic_keywords if result is accepted
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
    if norm_link and norm_link in seen_urls:
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
    if result_hash in seen_result_hashes:
        return None, relevance_llm_checks

    # Prepare for relevance check
    result_text = truncate_result(assembled)
    keywords_source = " ".join([part for part in [title, snippet] if part])

    # Check relevance using three-tier approach
    is_relevant_result, updated_llm_checks = check_result_relevance(
        result_text=result_text,
        keywords_source=keywords_source,
        topic_keywords=topic_keywords,
        question_embedding=question_embedding,
        topic_embedding_current=topic_embedding_current,
        current_query=current_query,
        chains=chains,
        conversation_text=conversation_text,
        user_query=user_query,
        prior_responses_text=prior_responses_text,
        current_datetime=current_datetime,
        current_year=current_year,
        current_month=current_month,
        current_day=current_day,
        relevance_llm_checks=relevance_llm_checks,
        cfg=cfg,
        embedding_client=embedding_client,
        context_similarity=context_similarity,
        inputs_builder=inputs_builder,
        reduce_context_and_rebuild=reduce_context_and_rebuild,
        rebuild_counts=rebuild_counts,
        mark_error=mark_error,
    )

    if not is_relevant_result:
        return None, updated_llm_checks

    # Accept result: update tracking sets
    seen_result_hashes.add(result_hash)
    if norm_link:
        seen_urls.add(norm_link)
    topic_keywords.update(extract_keywords(keywords_source))

    return result_text, updated_llm_checks


def process_search_round(
    *,
    current_query: str,
    chains: dict[str, Any],
    seen_result_hashes: Set[str],
    seen_urls: Set[str],
    topic_keywords: Set[str],
    question_embedding: List[float] | None,
    topic_embedding_current: List[float] | None,
    user_query: str,
    conversation_text: str,
    prior_responses_text: str,
    current_datetime: str,
    current_year: str,
    current_month: str,
    current_day: str,
    ddg_results: Callable[[str], List[dict[str, str]] | None],
    cfg: "AgentConfig",
    embedding_client: Any,
    context_similarity: Callable[[List[float] | None, List[float] | None, List[float] | None], float],
    inputs_builder: Callable[..., dict[str, Any]],
    reduce_context_and_rebuild: Callable[[str, str], None],
    rebuild_counts: dict[str, int],
    mark_error: Callable[[str], str],
) -> List[str]:
    """Execute one search round: fetch results and filter for relevance.

    Args:
        current_query: The search query to execute
        chains: Dictionary of LLM chains
        seen_result_hashes: Set of content hashes for deduplication
        seen_urls: Set of canonicalized URLs for deduplication
        topic_keywords: Current topic keywords (mutated)
        question_embedding: User question embedding
        topic_embedding_current: Topic embedding
        user_query: Original user query
        conversation_text: Conversation context
        prior_responses_text: Prior responses
        current_datetime: Current datetime string
        current_year: Current year string
        current_month: Current month string
        current_day: Current day string
        ddg_results: Callable to fetch DuckDuckGo results
        cfg: Agent configuration
        embedding_client: Client for generating embeddings
        context_similarity: Function to compute similarity
        inputs_builder: Function to build chain inputs
        reduce_context_and_rebuild: Callback to reduce context
        rebuild_counts: Dictionary tracking rebuild attempts
        mark_error: Callback to mark errors

    Returns:
        List of accepted result texts from this search round

    Side Effects:
        - Updates seen_result_hashes, seen_urls, topic_keywords
    """
    results_list = ddg_results(current_query)
    accepted_results: List[str] = []
    relevance_llm_checks = 0

    for res in results_list or []:
        result_text, relevance_llm_checks = process_search_result(
            result=res,
            seen_result_hashes=seen_result_hashes,
            seen_urls=seen_urls,
            topic_keywords=topic_keywords,
            question_embedding=question_embedding,
            topic_embedding_current=topic_embedding_current,
            current_query=current_query,
            chains=chains,
            conversation_text=conversation_text,
            user_query=user_query,
            prior_responses_text=prior_responses_text,
            current_datetime=current_datetime,
            current_year=current_year,
            current_month=current_month,
            current_day=current_day,
            relevance_llm_checks=relevance_llm_checks,
            cfg=cfg,
            embedding_client=embedding_client,
            context_similarity=context_similarity,
            inputs_builder=inputs_builder,
            reduce_context_and_rebuild=reduce_context_and_rebuild,
            rebuild_counts=rebuild_counts,
            mark_error=mark_error,
        )
        if result_text is not None:
            accepted_results.append(result_text)

    return accepted_results


__all__ = ["process_search_result", "process_search_round"]
