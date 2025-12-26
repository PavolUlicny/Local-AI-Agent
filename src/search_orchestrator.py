"""Search orchestration utilities for Local-AI-Agent."""

from __future__ import annotations

import logging
from typing import List, TYPE_CHECKING

from src.constants import ITERATION_GUARD_MULTIPLIER, MIN_ITERATION_GUARD
from src.exceptions import SearchAbort
from src.text_utils import normalize_query
from src.search_context import SearchContext, SearchState
from src.search_processing import process_search_round
from src.search_planning import generate_query_suggestions, enqueue_validated_queries
from src.search_parallel import process_queries_parallel, should_use_parallel_search

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.search_context import SearchServices


class SearchOrchestrator:
    """Coordinates multi-round web searches, relevance checks, and query planning."""

    def __init__(self, services: "SearchServices") -> None:
        """Initialize search orchestrator with services bundle.

        Args:
            services: Services bundle containing config, clients, and callbacks
        """
        self.services = services

    def run(
        self,
        query_inputs: dict,
        user_query: str,
        primary_search_query: str,
    ) -> List[str]:
        """Execute multi-round search orchestration.

        Args:
            query_inputs: Query input dictionary with conversation history
            user_query: User's query text
            primary_search_query: Initial search query

        Returns:
            List of aggregated result strings
        """
        # Build immutable search context from query_inputs
        from datetime import datetime, timezone

        dt_obj = datetime.now(timezone.utc)

        # Embed the user query for search result similarity checking
        question_embedding = self.services.embedding_client.embed(user_query)

        context = SearchContext(
            current_datetime=query_inputs.get("current_datetime", ""),
            current_year=query_inputs.get("current_year", str(dt_obj.year)),
            current_month=query_inputs.get("current_month", f"{dt_obj.month:02d}"),
            current_day=query_inputs.get("current_day", f"{dt_obj.day:02d}"),
            user_query=user_query,
            conversation_text=query_inputs.get("conversation_history", ""),
            prior_responses_text=query_inputs.get("known_answers", ""),
            question_embedding=question_embedding,  # Embed user query for result similarity
            topic_embedding_current=None,  # Topic embeddings removed in refactor
        )

        # Initialize mutable state
        state = SearchState()

        # Initialize search state
        aggregated_results: List[str] = []
        pending_queries: List[str] = [primary_search_query]
        rounds_executed = 0
        max_rounds = self.services.cfg.max_rounds
        max_suggestions = self.services.cfg.max_followup_suggestions
        max_fill_attempts = self.services.cfg.max_fill_attempts

        # Iteration guard to prevent infinite loops
        iteration_guard = max(max_rounds * ITERATION_GUARD_MULTIPLIER, MIN_ITERATION_GUARD)
        iteration_count = 0

        # Main search loop
        while pending_queries and rounds_executed < max_rounds and iteration_count < iteration_guard:
            iteration_count += 1

            # Determine if we should use parallel search
            # Parallel is beneficial when we have multiple pending queries
            if should_use_parallel_search(pending_queries, self.services.cfg):
                # Extract batch of queries to process in parallel
                batch_size = min(
                    self.services.cfg.max_concurrent_queries,
                    len(pending_queries),
                    max_rounds - rounds_executed,  # Don't exceed remaining rounds
                )
                batch_queries: list[str] = []

                # Pop queries for parallel execution
                # Note: queries in pending_queries are already deduplicated by enqueue_validated_queries
                while len(batch_queries) < batch_size and pending_queries:
                    query = pending_queries.pop(0)
                    batch_queries.append(query)

                if not batch_queries:
                    continue

                logging.info(f"Executing {len(batch_queries)} queries in parallel")

                # Execute batch in parallel
                round_results = process_queries_parallel(
                    queries=batch_queries,
                    context=context,
                    state=state,
                    services=self.services,
                )

                # Collect results and count as one round per query
                aggregated_results.extend(round_results)
                rounds_executed += len(batch_queries)

            else:
                # Sequential execution (original behavior)
                current_query = pending_queries.pop(0)

                # Skip duplicate queries
                norm_query = normalize_query(current_query)
                if norm_query in state.seen_query_norms:
                    continue
                state.seen_query_norms.add(norm_query)

                # Execute search round
                round_results = process_search_round(
                    current_query=current_query,
                    context=context,
                    state=state,
                    services=self.services,
                )

                # Collect results
                aggregated_results.extend(round_results)
                rounds_executed += 1

            # Generate follow-up query suggestions if needed
            if rounds_executed < max_rounds and max_suggestions > 0:
                try:
                    candidate_queries = generate_query_suggestions(
                        aggregated_results=aggregated_results,
                        suggestion_limit=max_suggestions,
                        context=context,
                        services=self.services,
                        raise_on_error=True,
                    )

                    # Validate and enqueue suggestions
                    enqueue_validated_queries(
                        candidate_queries=candidate_queries,
                        pending_queries=pending_queries,
                        max_rounds=max_rounds,
                        context=context,
                        state=state,
                        services=self.services,
                    )
                except SearchAbort:
                    logging.info("Query planning aborted; proceeding with pending queries.")

            # Fill pending queue if needed
            fill_attempts = 0
            while (
                len(pending_queries) < max_rounds - rounds_executed
                and fill_attempts < max_fill_attempts
                and max_suggestions > 0
            ):
                fill_attempts += 1
                try:
                    fill_candidates = generate_query_suggestions(
                        aggregated_results=aggregated_results,
                        suggestion_limit=max_suggestions,
                        context=context,
                        services=self.services,
                        raise_on_error=False,
                    )

                    initial_queue_len = len(pending_queries)
                    enqueue_validated_queries(
                        candidate_queries=fill_candidates,
                        pending_queries=pending_queries,
                        max_rounds=max_rounds,
                        context=context,
                        state=state,
                        services=self.services,
                    )

                    # Break if no new queries were added
                    if len(pending_queries) == initial_queue_len:
                        break
                except SearchAbort:
                    logging.info("Fill attempt aborted.")
                    break

        # Log completion
        if iteration_count >= iteration_guard:
            logging.warning(
                "Search orchestration hit iteration guard (%d iterations). " "This may indicate a logic error.",
                iteration_count,
            )

        # Return aggregated results only (topic keywords removed)
        return aggregated_results


__all__ = ["SearchOrchestrator"]
