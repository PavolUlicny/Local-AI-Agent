"""Search orchestration utilities for Local-AI-Agent."""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Set, TYPE_CHECKING

from src.keywords import extract_keywords
from src.text_utils import normalize_query
from src.search_chain_utils import SearchAbort, invoke_chain_with_retry
from src.search_validation import check_result_relevance, validate_candidate_query
from src.search_processing import process_search_result, process_search_round
from src.search_planning import (
    generate_query_suggestions,
    enqueue_validated_queries,
    parse_suggestions,
)

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.config import AgentConfig
    from src.embedding_client import EmbeddingClient


class SearchOrchestrator:
    """Coordinates multi-round web searches, relevance checks, and query planning."""

    def __init__(
        self,
        cfg: "AgentConfig",
        *,
        ddg_results: Callable[[str], List[dict[str, str]] | None],
        embedding_client: "EmbeddingClient",
        context_similarity: Callable[[List[float] | None, List[float] | None, List[float] | None], float],
        inputs_builder: Callable[..., dict[str, Any]],
        reduce_context_and_rebuild: Callable[[str, str], None],
        rebuild_counts: dict[str, int],
        char_budget: Callable[[int], int],
        mark_error: Callable[[str], str],
    ) -> None:
        self.cfg = cfg
        self._ddg_results = ddg_results
        self._embedding_client = embedding_client
        self._context_similarity = context_similarity
        self._inputs = inputs_builder
        self._reduce_context_and_rebuild = reduce_context_and_rebuild
        self._rebuild_counts = rebuild_counts
        self._char_budget = char_budget
        self._mark_error = mark_error

    def _invoke_chain_with_retry(
        self,
        *,
        chain: Any,
        inputs: dict[str, Any],
        rebuild_key: str,
        rebuild_label: str,
        fallback_value: str = "NO",
        raise_on_non_context_error: bool = False,
    ) -> tuple[str, bool]:
        """Delegate to utility function."""
        return invoke_chain_with_retry(
            chain=chain,
            inputs=inputs,
            rebuild_key=rebuild_key,
            rebuild_label=rebuild_label,
            fallback_value=fallback_value,
            raise_on_non_context_error=raise_on_non_context_error,
            cfg=self.cfg,
            reduce_context_and_rebuild=self._reduce_context_and_rebuild,
            rebuild_counts=self._rebuild_counts,
            mark_error=self._mark_error,
        )

    def _check_result_relevance(
        self,
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
    ) -> tuple[bool, int]:
        """Delegate to utility function."""
        return check_result_relevance(
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
            cfg=self.cfg,
            embedding_client=self._embedding_client,
            context_similarity=self._context_similarity,
            inputs_builder=self._inputs,
            reduce_context_and_rebuild=self._reduce_context_and_rebuild,
            rebuild_counts=self._rebuild_counts,
            mark_error=self._mark_error,
        )

    def _process_search_result(
        self,
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
    ) -> tuple[str | None, int]:
        """Delegate to utility function."""
        return process_search_result(
            result=result,
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
            cfg=self.cfg,
            embedding_client=self._embedding_client,
            context_similarity=self._context_similarity,
            inputs_builder=self._inputs,
            reduce_context_and_rebuild=self._reduce_context_and_rebuild,
            rebuild_counts=self._rebuild_counts,
            mark_error=self._mark_error,
        )

    def _generate_query_suggestions(
        self,
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
    ) -> List[str]:
        """Delegate to utility function."""
        return generate_query_suggestions(
            chains=chains,
            aggregated_results=aggregated_results,
            suggestion_limit=suggestion_limit,
            user_query=user_query,
            conversation_text=conversation_text,
            prior_responses_text=prior_responses_text,
            current_datetime=current_datetime,
            current_year=current_year,
            current_month=current_month,
            current_day=current_day,
            raise_on_error=raise_on_error,
            cfg=self.cfg,
            inputs_builder=self._inputs,
            char_budget=self._char_budget,
            reduce_context_and_rebuild=self._reduce_context_and_rebuild,
            rebuild_counts=self._rebuild_counts,
            mark_error=self._mark_error,
        )

    def _validate_candidate_query(
        self,
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
    ) -> bool:
        """Delegate to utility function."""
        return validate_candidate_query(
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
            cfg=self.cfg,
            embedding_client=self._embedding_client,
            context_similarity=self._context_similarity,
            inputs_builder=self._inputs,
            reduce_context_and_rebuild=self._reduce_context_and_rebuild,
            rebuild_counts=self._rebuild_counts,
            mark_error=self._mark_error,
        )

    def _process_search_round(
        self,
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
    ) -> List[str]:
        """Delegate to utility function."""
        return process_search_round(
            current_query=current_query,
            chains=chains,
            seen_result_hashes=seen_result_hashes,
            seen_urls=seen_urls,
            topic_keywords=topic_keywords,
            question_embedding=question_embedding,
            topic_embedding_current=topic_embedding_current,
            user_query=user_query,
            conversation_text=conversation_text,
            prior_responses_text=prior_responses_text,
            current_datetime=current_datetime,
            current_year=current_year,
            current_month=current_month,
            current_day=current_day,
            ddg_results=self._ddg_results,
            cfg=self.cfg,
            embedding_client=self._embedding_client,
            context_similarity=self._context_similarity,
            inputs_builder=self._inputs,
            reduce_context_and_rebuild=self._reduce_context_and_rebuild,
            rebuild_counts=self._rebuild_counts,
            mark_error=self._mark_error,
        )

    def _enqueue_validated_queries(
        self,
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
    ) -> None:
        """Delegate to utility function."""
        return enqueue_validated_queries(
            candidate_queries=candidate_queries,
            pending_queries=pending_queries,
            seen_query_norms=seen_query_norms,
            max_rounds=max_rounds,
            chains=chains,
            question_embedding=question_embedding,
            topic_embedding_current=topic_embedding_current,
            user_query=user_query,
            conversation_text=conversation_text,
            current_datetime=current_datetime,
            current_year=current_year,
            current_month=current_month,
            current_day=current_day,
            cfg=self.cfg,
            embedding_client=self._embedding_client,
            context_similarity=self._context_similarity,
            inputs_builder=self._inputs,
            reduce_context_and_rebuild=self._reduce_context_and_rebuild,
            rebuild_counts=self._rebuild_counts,
            mark_error=self._mark_error,
        )

    def run(
        self,
        *,
        chains: dict[str, Any],
        should_search: bool,
        user_query: str,
        current_datetime: str,
        current_year: str,
        current_month: str,
        current_day: str,
        conversation_text: str,
        prior_responses_text: str,
        question_embedding: List[float] | None,
        topic_embedding_current: List[float] | None,
        topic_keywords: Set[str],
        primary_search_query: str,
    ) -> tuple[List[str], Set[str]]:
        """Orchestrate multi-round web search with relevance filtering and query planning.

        Returns:
            Tuple of (aggregated_results, updated_topic_keywords)
        """
        # Early return if search not needed
        aggregated_results: List[str] = []
        if not should_search:
            return aggregated_results, topic_keywords

        # Initialize search state
        pending_queries: List[str] = [primary_search_query]
        seen_query_norms: Set[str] = {normalize_query(primary_search_query)}
        seen_result_hashes: Set[str] = set()
        seen_urls: Set[str] = set()

        # Bootstrap topic keywords if empty
        if not topic_keywords:
            topic_keywords.update(extract_keywords(user_query))
            topic_keywords.update(extract_keywords(primary_search_query))

        # Search loop with safety guards
        max_rounds = self.cfg.max_rounds
        round_index = 0
        iteration_guard = max(max_rounds * 4, 20)
        iterations = 0

        while round_index < len(pending_queries) and round_index < max_rounds:
            # Safety: prevent infinite loops
            iterations += 1
            if iterations > iteration_guard:
                logging.warning(
                    "Search loop aborted after %d iterations without progress; breaking to avoid a stall.",
                    iteration_guard,
                )
                break

            # Execute search round
            current_query = pending_queries[round_index]
            round_results = self._process_search_round(
                current_query=current_query,
                chains=chains,
                seen_result_hashes=seen_result_hashes,
                seen_urls=seen_urls,
                topic_keywords=topic_keywords,
                question_embedding=question_embedding,
                topic_embedding_current=topic_embedding_current,
                user_query=user_query,
                conversation_text=conversation_text,
                prior_responses_text=prior_responses_text,
                current_datetime=current_datetime,
                current_year=current_year,
                current_month=current_month,
                current_day=current_day,
            )

            # Update state based on results
            if not round_results:
                logging.info("No relevant results for '%s'. Not counting toward limit.", current_query)
                if round_index < len(pending_queries):
                    pending_queries.pop(round_index)
            else:
                aggregated_results.extend(round_results)
                round_index += 1
                if round_index >= max_rounds:
                    break

            # Generate and enqueue follow-up queries if slots remain
            remaining_slots = max_rounds - round_index
            if remaining_slots > 0:
                suggestion_limit = min(self.cfg.max_followup_suggestions, remaining_slots)
                new_queries = self._generate_query_suggestions(
                    chains=chains,
                    aggregated_results=aggregated_results,
                    suggestion_limit=suggestion_limit,
                    user_query=user_query,
                    conversation_text=conversation_text,
                    prior_responses_text=prior_responses_text,
                    current_datetime=current_datetime,
                    current_year=current_year,
                    current_month=current_month,
                    current_day=current_day,
                    raise_on_error=True,
                )
                self._enqueue_validated_queries(
                    candidate_queries=new_queries,
                    pending_queries=pending_queries,
                    seen_query_norms=seen_query_norms,
                    max_rounds=max_rounds,
                    chains=chains,
                    question_embedding=question_embedding,
                    topic_embedding_current=topic_embedding_current,
                    user_query=user_query,
                    conversation_text=conversation_text,
                    current_datetime=current_datetime,
                    current_year=current_year,
                    current_month=current_month,
                    current_day=current_day,
                )
            self._fill_pending_queries(
                chains,
                pending_queries,
                seen_query_norms,
                aggregated_results,
                question_embedding,
                topic_embedding_current,
                conversation_text,
                user_query,
                prior_responses_text,
                current_datetime,
                current_year,
                current_month,
                current_day,
            )
        return aggregated_results, topic_keywords

    def _fill_pending_queries(
        self,
        chains: dict[str, Any],
        pending_queries: List[str],
        seen_query_norms: Set[str],
        aggregated_results: List[str],
        question_embedding: List[float] | None,
        topic_embedding_current: List[float] | None,
        conversation_text: str,
        user_query: str,
        prior_responses_text: str,
        current_datetime: str,
        current_year: str,
        current_month: str,
        current_day: str,
    ) -> None:
        fill_attempts = 0
        while len(pending_queries) < self.cfg.max_rounds and fill_attempts < self.cfg.max_fill_attempts:
            fill_attempts += 1
            remaining_slots = self.cfg.max_rounds - len(pending_queries)
            if remaining_slots <= 0:
                break
            suggestion_limit = min(self.cfg.max_followup_suggestions, remaining_slots)
            fill_queries = self._generate_query_suggestions(
                chains=chains,
                aggregated_results=aggregated_results,
                suggestion_limit=suggestion_limit,
                user_query=user_query,
                conversation_text=conversation_text,
                prior_responses_text=prior_responses_text,
                current_datetime=current_datetime,
                current_year=current_year,
                current_month=current_month,
                current_day=current_day,
                raise_on_error=False,
            )
            # If planning failed (returned empty list), stop trying to fill
            if not fill_queries:
                break
            for candidate in fill_queries:
                norm_candidate = normalize_query(candidate)
                if norm_candidate in seen_query_norms or len(pending_queries) >= self.cfg.max_rounds:
                    continue
                if self._validate_candidate_query(
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
                ):
                    pending_queries.append(candidate)
                    seen_query_norms.add(norm_candidate)

    @staticmethod
    def _parse_suggestions(raw: Any, limit: int) -> List[str]:
        """Delegate to utility function."""
        return parse_suggestions(raw, limit)


__all__ = ["SearchAbort", "SearchOrchestrator"]
