"""Query planning and suggestion management utilities."""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Set, TYPE_CHECKING

from src.text_utils import MAX_SEARCH_RESULTS_CHARS, normalize_query, truncate_text
from src.search_chain_utils import invoke_chain_with_retry
from src.search_validation import validate_candidate_query

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.config import AgentConfig


def generate_query_suggestions(
    *,
    chains: dict[str, Any],
    aggregated_results: List[str],
    suggestion_limit: int,
    user_query: str,
    conversation_text: str,
    prior_responses_text: str,
    current_datetime: str,
    current_year: str,
    current_month: str,
    current_day: str,
    raise_on_error: bool = True,
    cfg: "AgentConfig",
    inputs_builder: Callable[..., dict[str, Any]],
    char_budget: Callable[[int], int],
    reduce_context_and_rebuild: Callable[[str, str], None],
    rebuild_counts: dict[str, int],
    mark_error: Callable[[str], str],
) -> List[str]:
    """Generate follow-up query suggestions using the planning chain.

    Args:
        chains: Dictionary of LLM chains (needs 'planning')
        aggregated_results: Results collected so far
        suggestion_limit: Maximum number of suggestions to request
        user_query: Original user query
        conversation_text: Conversation context
        prior_responses_text: Prior agent responses
        current_datetime: Current datetime string
        current_year: Current year string
        current_month: Current month string
        current_day: Current day string
        raise_on_error: If True, raise SearchAbort on non-context errors; if False, return empty list
        cfg: Agent configuration
        inputs_builder: Function to build chain inputs
        char_budget: Function to compute character budget
        reduce_context_and_rebuild: Callback to reduce context
        rebuild_counts: Dictionary tracking rebuild attempts
        mark_error: Callback to mark errors

    Returns:
        List of suggested query strings (may be empty on failure)
    """
    results_to_date = "\n\n".join(aggregated_results) or "No results yet."
    results_to_date = truncate_text(results_to_date, char_budget(MAX_SEARCH_RESULTS_CHARS))

    suggestions_raw, _llm_success = invoke_chain_with_retry(
        chain=chains["planning"],
        inputs=inputs_builder(
            current_datetime,
            current_year,
            current_month,
            current_day,
            conversation_text,
            user_query,
            results_to_date=results_to_date,
            suggestion_limit=str(suggestion_limit),
            known_answers=prior_responses_text,
        ),
        rebuild_key="planning",
        rebuild_label="query planning" if raise_on_error else "planning",
        fallback_value="NONE",
        raise_on_non_context_error=raise_on_error,
        cfg=cfg,
        reduce_context_and_rebuild=reduce_context_and_rebuild,
        rebuild_counts=rebuild_counts,
        mark_error=mark_error,
    )

    return parse_suggestions(suggestions_raw, suggestion_limit)


def enqueue_validated_queries(
    *,
    candidate_queries: List[str],
    pending_queries: List[str],
    seen_query_norms: Set[str],
    max_rounds: int,
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
) -> None:
    """Validate candidate queries and enqueue accepted ones.

    Args:
        candidate_queries: List of candidate query strings to validate
        pending_queries: List of pending queries (mutated - queries appended)
        seen_query_norms: Set of normalized queries already seen (mutated)
        max_rounds: Maximum number of rounds allowed
        chains: Dictionary of LLM chains
        question_embedding: User question embedding
        topic_embedding_current: Topic embedding
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

    Side Effects:
        - Appends validated queries to pending_queries
        - Updates seen_query_norms with normalized queries
    """
    for candidate in candidate_queries:
        norm_candidate = normalize_query(candidate)
        if norm_candidate in seen_query_norms or len(pending_queries) >= max_rounds:
            continue

        if validate_candidate_query(
            candidate=candidate,
            chains=chains,
            question_embedding=question_embedding,
            topic_embedding_current=topic_embedding_current,
            user_query=user_query,
            conversation_text=conversation_text,
            current_datetime=current_datetime,
            current_year=current_year,
            current_month=current_month,
            current_day=current_day,
            cfg=cfg,
            embedding_client=embedding_client,
            context_similarity=context_similarity,
            inputs_builder=inputs_builder,
            reduce_context_and_rebuild=reduce_context_and_rebuild,
            rebuild_counts=rebuild_counts,
            mark_error=mark_error,
        ):
            pending_queries.append(candidate)
            seen_query_norms.add(norm_candidate)
        else:
            logging.info("Skipping off-topic follow-up suggestion: %s", candidate)


def parse_suggestions(raw: Any, limit: int) -> List[str]:
    """Parse LLM suggestion output into a list of query strings.

    Args:
        raw: Raw LLM output (string with one suggestion per line)
        limit: Maximum number of suggestions to return

    Returns:
        List of parsed suggestion strings (empty if "NONE" found)
    """
    suggestions: List[str] = []
    for line in str(raw).splitlines():
        normalized = line.strip().strip("-*\"'").strip()
        if not normalized:
            continue
        if normalized.lower() == "none":
            return []
        suggestions.append(normalized)
    if suggestions and limit > 0:
        return suggestions[:limit]
    return suggestions


__all__ = [
    "generate_query_suggestions",
    "enqueue_validated_queries",
    "parse_suggestions",
]
