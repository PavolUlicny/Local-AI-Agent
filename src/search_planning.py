"""Query planning and suggestion management utilities."""

from __future__ import annotations

import logging
from typing import Any, List, TYPE_CHECKING

from src.constants import ChainName, RebuildKey, MAX_SEARCH_RESULTS_CHARS
from src.text_utils import normalize_query, truncate_text
from src.search_chain_utils import invoke_chain_with_retry
from src.search_validation import validate_candidate_query

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.search_context import SearchContext, SearchServices, SearchState


def generate_query_suggestions(
    aggregated_results: List[str],
    suggestion_limit: int,
    context: "SearchContext",
    services: "SearchServices",
    raise_on_error: bool = True,
) -> List[str]:
    """Generate follow-up query suggestions using the planning chain.

    Args:
        aggregated_results: Results collected so far
        suggestion_limit: Maximum number of suggestions to request
        context: Immutable search context (query, conversation, datetime)
        services: Services bundle (config, clients, callbacks)
        raise_on_error: If True, raise SearchAbort on non-context errors; if False, return empty list

    Returns:
        List of suggested query strings (may be empty on failure)
    """
    results_to_date = "\n\n".join(aggregated_results) or "No results yet."
    results_to_date = truncate_text(results_to_date, services.char_budget(MAX_SEARCH_RESULTS_CHARS))

    suggestions_raw, _llm_success = invoke_chain_with_retry(
        chain=services.chains[ChainName.PLANNING],
        inputs=services.inputs_builder(
            context.current_datetime,
            context.current_year,
            context.current_month,
            context.current_day,
            context.conversation_text,
            context.user_query,
            results_to_date=results_to_date,
            suggestion_limit=str(suggestion_limit),
            known_answers=context.prior_responses_text,
        ),
        rebuild_key=RebuildKey.PLANNING,
        rebuild_label="query planning" if raise_on_error else "planning",
        fallback_value="NONE",
        raise_on_non_context_error=raise_on_error,
        cfg=services.cfg,
        reduce_context_and_rebuild=services.reduce_context_and_rebuild,
        rebuild_counts=services.rebuild_counts,
        mark_error=services.mark_error,
    )

    return parse_suggestions(suggestions_raw, suggestion_limit)


def enqueue_validated_queries(
    candidate_queries: List[str],
    pending_queries: List[str],
    max_rounds: int,
    context: "SearchContext",
    state: "SearchState",
    services: "SearchServices",
) -> None:
    """Validate candidate queries and enqueue accepted ones.

    Args:
        candidate_queries: List of candidate query strings to validate
        pending_queries: List of pending queries (mutated - queries appended)
        max_rounds: Maximum number of rounds allowed
        context: Immutable search context (query, conversation, embeddings)
        state: Mutable search state (seen_query_norms)
        services: Services bundle (config, clients, callbacks)

    Side Effects:
        - Appends validated queries to pending_queries
        - Checks state.seen_query_norms to prevent duplicates (but doesn't modify it)
    """
    for candidate in candidate_queries:
        norm_candidate = normalize_query(candidate)
        # Check if query already queued or executed, or if queue is full
        if norm_candidate in state.seen_query_norms or len(pending_queries) >= max_rounds:
            continue

        if validate_candidate_query(
            candidate=candidate,
            context=context,
            services=services,
        ):
            pending_queries.append(candidate)
            # Note: seen_query_norms is updated during query execution, not here
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
