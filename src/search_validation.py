"""Validation utilities for search results and queries."""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Set, TYPE_CHECKING

from src.keywords import is_relevant
from src.text_utils import PATTERN_YES_NO, regex_validate, truncate_text
from src.search_chain_utils import invoke_chain_with_retry

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.config import AgentConfig


def check_result_relevance(
    *,
    result_text: str,
    keywords_source: str,
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
) -> tuple[bool, int]:
    """Determine if a search result is relevant using three-tier approach.

    Three-tier relevance check:
    1. Keyword overlap (fast)
    2. Embedding similarity (medium)
    3. LLM judgment (slow, limited by max_relevance_llm_checks)

    Args:
        result_text: The formatted search result text
        keywords_source: Text to use for embedding (title + snippet)
        topic_keywords: Current topic keywords for keyword matching
        question_embedding: User question embedding for similarity
        topic_embedding_current: Current topic embedding for similarity
        current_query: The search query that produced this result
        chains: Dictionary of LLM chains (needs 'result_filter')
        conversation_text: Conversation context for LLM
        user_query: Original user query
        prior_responses_text: Prior agent responses
        current_datetime: Current datetime string
        current_year: Current year string
        current_month: Current month string
        current_day: Current day string
        relevance_llm_checks: Current count of LLM checks in this round
        cfg: Agent configuration
        embedding_client: Client for generating embeddings
        context_similarity: Function to compute similarity between embeddings
        inputs_builder: Function to build chain inputs
        reduce_context_and_rebuild: Callback to reduce context and rebuild
        rebuild_counts: Dictionary tracking rebuild attempts
        mark_error: Callback to mark errors

    Returns:
        Tuple of (is_relevant: bool, updated_llm_checks: int)
    """
    # Tier 1: Fast keyword matching
    relevant = is_relevant(result_text, topic_keywords)
    if relevant:
        return True, relevance_llm_checks

    # Tier 2: Embedding similarity check
    result_embedding = embedding_client.embed(keywords_source)
    similarity = context_similarity(
        result_embedding,
        question_embedding,
        topic_embedding_current,
    )
    if similarity >= cfg.embedding_result_similarity_threshold:
        return True, relevance_llm_checks

    # Tier 3: LLM judgment (with limit)
    if relevance_llm_checks >= cfg.max_relevance_llm_checks:
        return False, relevance_llm_checks

    kw_list = sorted(topic_keywords) if topic_keywords else []
    if len(kw_list) > 50:
        kw_list = kw_list[:50]
    topic_keywords_text = ", ".join(kw_list) if kw_list else "None"
    topic_keywords_text = truncate_text(topic_keywords_text, 1000)

    relevance_raw, llm_success = invoke_chain_with_retry(
        chain=chains["result_filter"],
        inputs=inputs_builder(
            current_datetime,
            current_year,
            current_month,
            current_day,
            conversation_text,
            user_query,
            search_query=current_query,
            raw_result=result_text,
            known_answers=prior_responses_text,
            topic_keywords=topic_keywords_text,
        ),
        rebuild_key="relevance",
        rebuild_label="relevance",
        fallback_value="NO",
        raise_on_non_context_error=False,
        cfg=cfg,
        reduce_context_and_rebuild=reduce_context_and_rebuild,
        rebuild_counts=rebuild_counts,
        mark_error=mark_error,
    )
    relevance_decision = regex_validate(relevance_raw, PATTERN_YES_NO, "NO")
    if llm_success:
        relevance_llm_checks += 1

    if relevance_decision == "YES":
        return True, relevance_llm_checks
    else:
        return False, relevance_llm_checks


def validate_candidate_query(
    *,
    candidate: str,
    chains: dict[str, Any],
    question_embedding: List[float] | None,
    topic_embedding_current: List[float] | None,
    user_query: str,
    conversation_text: str,
    current_datetime: str,
    current_year: str,
    current_month: str,
    current_day: str,
    cfg: "AgentConfig",
    embedding_client: Any,
    context_similarity: Callable[[List[float] | None, List[float] | None, List[float] | None], float],
    inputs_builder: Callable[..., dict[str, Any]],
    reduce_context_and_rebuild: Callable[[str, str], None],
    rebuild_counts: dict[str, int],
    mark_error: Callable[[str], str],
) -> bool:
    """Validate a candidate query using embedding similarity and LLM filter.

    Two-stage validation:
    1. Embedding similarity check (if embeddings available)
    2. LLM query_filter chain (YES/NO decision)

    Args:
        candidate: The candidate query string to validate
        chains: Dictionary of LLM chains (needs 'query_filter')
        question_embedding: User question embedding for similarity
        topic_embedding_current: Topic embedding for similarity
        user_query: Original user query
        conversation_text: Conversation context
        current_datetime: Current datetime string
        current_year: Current year string
        current_month: Current month string
        current_day: Current day string
        cfg: Agent configuration
        embedding_client: Client for generating embeddings
        context_similarity: Function to compute similarity
        inputs_builder: Function to build chain inputs
        reduce_context_and_rebuild: Callback to reduce context
        rebuild_counts: Dictionary tracking rebuild attempts
        mark_error: Callback to mark errors

    Returns:
        True if candidate query should be accepted, False otherwise
    """
    # Stage 1: Embedding similarity check
    candidate_embedding = embedding_client.embed(candidate)
    if candidate_embedding is not None and (question_embedding or topic_embedding_current):
        similarity = context_similarity(
            candidate_embedding,
            question_embedding,
            topic_embedding_current,
        )
        if similarity < cfg.embedding_query_similarity_threshold:
            logging.info(
                "Skipping suggestion with low semantic similarity (%.2f): %s",
                similarity,
                candidate,
            )
            return False

    # Stage 2: LLM filter
    verdict_raw, _llm_success = invoke_chain_with_retry(
        chain=chains["query_filter"],
        inputs=inputs_builder(
            current_datetime,
            current_year,
            current_month,
            current_day,
            conversation_text,
            user_query,
            candidate_query=candidate,
        ),
        rebuild_key="query_filter",
        rebuild_label="query filter",
        fallback_value="SKIP",
        raise_on_non_context_error=False,
        cfg=cfg,
        reduce_context_and_rebuild=reduce_context_and_rebuild,
        rebuild_counts=rebuild_counts,
        mark_error=mark_error,
    )

    if verdict_raw == "SKIP":
        logging.info("Skipping suggestion: %s", candidate)
        return False

    verdict = regex_validate(verdict_raw, PATTERN_YES_NO, "NO")
    return verdict == "YES"


__all__ = ["check_result_relevance", "validate_candidate_query"]
