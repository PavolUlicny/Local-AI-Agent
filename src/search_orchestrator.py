"""Search orchestration utilities for Local-AI-Agent."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, List, Set, TYPE_CHECKING

from src.exceptions import ResponseError
from src.keywords import extract_keywords, is_relevant
from src.text_utils import (
    MAX_REBUILD_RETRIES,
    MAX_SEARCH_RESULTS_CHARS,
    PATTERN_YES_NO,
    is_context_length_error,
    normalize_query,
    regex_validate,
    truncate_result,
    truncate_text,
)
from src.url_utils import canonicalize_url
from src.model_utils import handle_missing_model

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.config import AgentConfig
    from src.embedding_client import EmbeddingClient


class SearchAbort(Exception):
    """Raised when search orchestration must halt due to a fatal error."""


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
        """Invoke an LLM chain with automatic context-length retry logic.

        Args:
            chain: The LangChain chain to invoke
            inputs: Input dictionary for the chain
            rebuild_key: Key for tracking rebuild count in self._rebuild_counts
            rebuild_label: Human-readable label for logging
            fallback_value: Value to return on non-critical failures
            raise_on_non_context_error: If True, raise SearchAbort on non-context ResponseErrors

        Returns:
            Tuple of (result_string, was_successful_llm_call)
            - result_string: The chain's output or fallback_value on errors
            - was_successful_llm_call: True if LLM was invoked successfully, False if fallback used

        Raises:
            SearchAbort: On model-not-found errors or non-context errors when raise_on_non_context_error=True
        """
        try:
            raw_output = chain.invoke(inputs)
            return str(raw_output), True
        except ResponseError as exc:  # pragma: no cover - network/model specific
            # Handle model not found (fatal error)
            if "not found" in str(exc).lower():
                handle_missing_model(self._mark_error, "Robot", self.cfg.robot_model)
                raise SearchAbort from exc

            # Handle context length errors with retry
            if is_context_length_error(str(exc)):
                if self._rebuild_counts[rebuild_key] < MAX_REBUILD_RETRIES:
                    self._reduce_context_and_rebuild(rebuild_key, rebuild_label)
                    try:
                        raw_output = chain.invoke(inputs)
                        return str(raw_output), True
                    except ResponseError:
                        logging.info(
                            "%s retry failed; using fallback value.",
                            rebuild_label.capitalize(),
                        )
                        return fallback_value, False
                else:
                    logging.info(
                        "Reached %s rebuild cap; using fallback value.",
                        rebuild_label,
                    )
                    return fallback_value, False

            # Handle other ResponseErrors
            if raise_on_non_context_error:
                logging.error("%s failed: %s", rebuild_label.capitalize(), exc)
                self._mark_error(f"{rebuild_label.capitalize()} failed; please retry.")
                raise SearchAbort from exc
            else:
                logging.info(
                    "%s failed; using fallback value. Error: %s",
                    rebuild_label.capitalize(),
                    exc,
                )
                return fallback_value, False

        except Exception as exc:  # pragma: no cover - defensive
            logging.info(
                "%s crashed; using fallback value. Error: %s",
                rebuild_label.capitalize(),
                exc,
            )
            return fallback_value, False

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

        Returns:
            Tuple of (is_relevant: bool, updated_llm_checks: int)
        """
        # Tier 1: Fast keyword matching
        relevant = is_relevant(result_text, topic_keywords)
        if relevant:
            return True, relevance_llm_checks

        # Tier 2: Embedding similarity check
        result_embedding = self._embedding_client.embed(keywords_source)
        similarity = self._context_similarity(
            result_embedding,
            question_embedding,
            topic_embedding_current,
        )
        if similarity >= self.cfg.embedding_result_similarity_threshold:
            return True, relevance_llm_checks

        # Tier 3: LLM judgment (with limit)
        if relevance_llm_checks >= self.cfg.max_relevance_llm_checks:
            return False, relevance_llm_checks

        kw_list = sorted(topic_keywords) if topic_keywords else []
        if len(kw_list) > 50:
            kw_list = kw_list[:50]
        topic_keywords_text = ", ".join(kw_list) if kw_list else "None"
        topic_keywords_text = truncate_text(topic_keywords_text, 1000)

        relevance_raw, llm_success = self._invoke_chain_with_retry(
            chain=chains["result_filter"],
            inputs=self._inputs(
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
        )
        relevance_decision = regex_validate(relevance_raw, PATTERN_YES_NO, "NO")
        if llm_success:
            relevance_llm_checks += 1

        if relevance_decision == "YES":
            return True, relevance_llm_checks
        else:
            return False, relevance_llm_checks

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
        is_relevant, updated_llm_checks = self._check_result_relevance(
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
        )

        if not is_relevant:
            return None, updated_llm_checks

        # Accept result: update tracking sets
        seen_result_hashes.add(result_hash)
        if norm_link:
            seen_urls.add(norm_link)
        topic_keywords.update(extract_keywords(keywords_source))

        return result_text, updated_llm_checks

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

        Returns:
            List of suggested query strings (may be empty on failure)
        """
        results_to_date = "\n\n".join(aggregated_results) or "No results yet."
        results_to_date = truncate_text(results_to_date, self._char_budget(MAX_SEARCH_RESULTS_CHARS))

        suggestions_raw, _llm_success = self._invoke_chain_with_retry(
            chain=chains["planning"],
            inputs=self._inputs(
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
        )

        return self._parse_suggestions(suggestions_raw, suggestion_limit)

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

        Returns:
            True if candidate query should be accepted, False otherwise
        """
        # Stage 1: Embedding similarity check
        candidate_embedding = self._embedding_client.embed(candidate)
        if candidate_embedding is not None and (question_embedding or topic_embedding_current):
            similarity = self._context_similarity(
                candidate_embedding,
                question_embedding,
                topic_embedding_current,
            )
            if similarity < self.cfg.embedding_query_similarity_threshold:
                logging.info(
                    "Skipping suggestion with low semantic similarity (%.2f): %s",
                    similarity,
                    candidate,
                )
                return False

        # Stage 2: LLM filter
        verdict_raw, _llm_success = self._invoke_chain_with_retry(
            chain=chains["query_filter"],
            inputs=self._inputs(
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
        )

        if verdict_raw == "SKIP":
            logging.info("Skipping suggestion: %s", candidate)
            return False

        verdict = regex_validate(verdict_raw, PATTERN_YES_NO, "NO")
        return verdict == "YES"

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

        Returns:
            List of accepted result texts from this search round

        Side Effects:
            - Updates seen_result_hashes, seen_urls, topic_keywords
        """
        results_list = self._ddg_results(current_query)
        accepted_results: List[str] = []
        relevance_llm_checks = 0

        for res in results_list or []:
            result_text, relevance_llm_checks = self._process_search_result(
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
            )
            if result_text is not None:
                accepted_results.append(result_text)

        return accepted_results

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

        Side Effects:
            - Appends validated queries to pending_queries
            - Updates seen_query_norms with normalized queries
        """
        for candidate in candidate_queries:
            norm_candidate = normalize_query(candidate)
            if norm_candidate in seen_query_norms or len(pending_queries) >= max_rounds:
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
            else:
                logging.info("Skipping off-topic follow-up suggestion: %s", candidate)

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


__all__ = ["SearchAbort", "SearchOrchestrator"]
