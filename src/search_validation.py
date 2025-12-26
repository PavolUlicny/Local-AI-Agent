"""Validation utilities for search results and queries."""

from __future__ import annotations

import logging
from typing import Set, TYPE_CHECKING

from src.constants import ChainName, RebuildKey
from src.keywords import is_relevant
from src.text_utils import PATTERN_YES_NO, regex_validate, truncate_text
from src.search_chain_utils import invoke_chain_with_retry

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.search_context import SearchContext, SearchServices


def check_result_relevance(
    result_text: str,
    keywords_source: str,
    current_query: str,
    topic_keywords: Set[str],
    relevance_llm_checks: int,
    context: "SearchContext",
    services: "SearchServices",
) -> tuple[bool, int]:
    """Determine if a search result is relevant using three-tier approach.

    Three-tier relevance check:
    1. Keyword overlap (fast)
    2. Embedding similarity (medium)
    3. LLM judgment (slow, limited by max_relevance_llm_checks)

    Args:
        result_text: The formatted search result text
        keywords_source: Text to use for embedding (title + snippet)
        current_query: The search query that produced this result
        topic_keywords: Current topic keywords for keyword matching
        relevance_llm_checks: Current count of LLM checks in this round
        context: Immutable search context (query, conversation, embeddings)
        services: Services bundle (config, clients, callbacks)

    Returns:
        Tuple of (is_relevant: bool, updated_llm_checks: int)
    """
    # Tier 1: Fast keyword matching
    relevant = is_relevant(result_text, topic_keywords)
    if relevant:
        return True, relevance_llm_checks

    # Tier 2: Embedding similarity check
    result_embedding = services.embedding_client.embed(keywords_source)
    similarity = services.context_similarity(
        result_embedding,
        context.question_embedding,
        context.topic_embedding_current,
    )
    if similarity >= services.cfg.embedding_result_similarity_threshold:
        return True, relevance_llm_checks

    # Tier 3: LLM judgment (with limit)
    if relevance_llm_checks >= services.cfg.max_relevance_llm_checks:
        return False, relevance_llm_checks

    kw_list = sorted(topic_keywords) if topic_keywords else []
    if len(kw_list) > 50:
        kw_list = kw_list[:50]
    topic_keywords_text = ", ".join(kw_list) if kw_list else "None"
    topic_keywords_text = truncate_text(topic_keywords_text, 1000)

    relevance_raw, llm_success = invoke_chain_with_retry(
        chain=services.chains[ChainName.RESULT_FILTER],
        inputs=services.inputs_builder(
            context.current_datetime,
            context.current_year,
            context.current_month,
            context.current_day,
            context.conversation_text,
            context.user_query,
            search_query=current_query,
            raw_result=result_text,
            known_answers=context.prior_responses_text,
            topic_keywords=topic_keywords_text,
        ),
        rebuild_key=RebuildKey.RELEVANCE,
        rebuild_label="relevance",
        fallback_value="NO",
        raise_on_non_context_error=False,
        cfg=services.cfg,
        reduce_context_and_rebuild=services.reduce_context_and_rebuild,
        rebuild_counts=services.rebuild_counts,
        mark_error=services.mark_error,
    )
    relevance_decision = regex_validate(relevance_raw, PATTERN_YES_NO, "NO")
    if llm_success:
        relevance_llm_checks += 1

    if relevance_decision == "YES":
        return True, relevance_llm_checks
    else:
        return False, relevance_llm_checks


def validate_candidate_query(
    candidate: str,
    context: "SearchContext",
    services: "SearchServices",
) -> bool:
    """Validate a candidate query using embedding similarity and LLM filter.

    Two-stage validation:
    1. Embedding similarity check (if embeddings available)
    2. LLM query_filter chain (YES/NO decision)

    Args:
        candidate: The candidate query string to validate
        context: Immutable search context (query, conversation, embeddings)
        services: Services bundle (config, clients, callbacks)

    Returns:
        True if candidate query should be accepted, False otherwise
    """
    # Stage 1: Embedding similarity check
    candidate_embedding = services.embedding_client.embed(candidate)
    if candidate_embedding is not None and (context.question_embedding or context.topic_embedding_current):
        similarity = services.context_similarity(
            candidate_embedding,
            context.question_embedding,
            context.topic_embedding_current,
        )
        if similarity < services.cfg.embedding_query_similarity_threshold:
            logging.info(
                "Skipping suggestion with low semantic similarity (%.2f): %s",
                similarity,
                candidate,
            )
            return False

    # Stage 2: LLM filter
    verdict_raw, _llm_success = invoke_chain_with_retry(
        chain=services.chains[ChainName.QUERY_FILTER],
        inputs=services.inputs_builder(
            context.current_datetime,
            context.current_year,
            context.current_month,
            context.current_day,
            context.conversation_text,
            context.user_query,
            candidate_query=candidate,
        ),
        rebuild_key=RebuildKey.QUERY_FILTER,
        rebuild_label="query filter",
        fallback_value="SKIP",
        raise_on_non_context_error=False,
        cfg=services.cfg,
        reduce_context_and_rebuild=services.reduce_context_and_rebuild,
        rebuild_counts=services.rebuild_counts,
        mark_error=services.mark_error,
    )

    if verdict_raw == "SKIP":
        logging.info("Skipping suggestion: %s", candidate)
        return False

    verdict = regex_validate(verdict_raw, PATTERN_YES_NO, "NO")
    return verdict == "YES"


__all__ = ["check_result_relevance", "validate_candidate_query"]
