"""Search orchestration utilities for Local-AI-Agent."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, List, Set, TYPE_CHECKING

from src.exceptions import ResponseError
from src.keywords import _extract_keywords, _is_relevant
from src.text_utils import (
    MAX_REBUILD_RETRIES,
    MAX_SEARCH_RESULTS_CHARS,
    _PATTERN_YES_NO,
    _is_context_length_error,
    _normalize_query,
    _regex_validate,
    _truncate_result,
    _truncate_text,
)
from src.url_utils import _canonicalize_url
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
        aggregated_results: List[str] = []
        if not should_search:
            return aggregated_results, topic_keywords

        pending_queries: List[str] = [primary_search_query]
        seen_query_norms: Set[str] = {_normalize_query(primary_search_query)}
        seen_result_hashes: Set[str] = set()
        seen_urls: Set[str] = set()
        if not topic_keywords:
            topic_keywords.update(_extract_keywords(user_query))
            topic_keywords.update(_extract_keywords(primary_search_query))
        max_rounds = self.cfg.max_rounds
        round_index = 0
        iteration_guard = max(max_rounds * 4, 20)
        iterations = 0
        while round_index < len(pending_queries) and round_index < max_rounds:
            iterations += 1
            if iterations > iteration_guard:
                logging.warning(
                    "Search loop aborted after %d iterations without progress; breaking to avoid a stall.",
                    iteration_guard,
                )
                break
            current_query = pending_queries[round_index]
            results_list = self._ddg_results(current_query)
            accepted_any = False
            relevance_llm_checks = 0
            for res in results_list or []:
                title = str(res.get("title", "")).strip()
                link = str(res.get("link", "")).strip()
                snippet = str(res.get("snippet", "")).strip()
                if not any([title, snippet, link]):
                    continue
                norm_link = _canonicalize_url(link) if link else ""
                if norm_link and norm_link in seen_urls:
                    continue
                assembled = "\n".join(
                    part
                    for part in [
                        (f"Title: {title}" if title else ""),
                        (f"URL: {link}" if link else ""),
                        (f"Snippet: {snippet}" if snippet else ""),
                    ]
                    if part
                )
                result_hash = hashlib.sha256(assembled.encode("utf-8", errors="ignore")).hexdigest()
                if result_hash in seen_result_hashes:
                    continue
                result_text = _truncate_result(assembled)
                keywords_source = " ".join([part for part in [title, snippet] if part])
                relevant = _is_relevant(result_text, topic_keywords)
                if not relevant:
                    result_embedding = self._embedding_client.embed(keywords_source)
                    similarity = self._context_similarity(
                        result_embedding,
                        question_embedding,
                        topic_embedding_current,
                    )
                    if similarity >= self.cfg.embedding_result_similarity_threshold:
                        relevant = True
                    if not relevant:
                        if relevance_llm_checks >= self.cfg.max_relevance_llm_checks:
                            continue
                        kw_list = sorted(topic_keywords) if topic_keywords else []
                        if len(kw_list) > 50:
                            kw_list = kw_list[:50]
                        topic_keywords_text = ", ".join(kw_list) if kw_list else "None"
                        topic_keywords_text = _truncate_text(topic_keywords_text, 1000)
                        try:
                            relevance_raw = chains["result_filter"].invoke(
                                self._inputs(
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
                                )
                            )
                            relevance_decision = _regex_validate(str(relevance_raw), _PATTERN_YES_NO, "NO")
                            relevance_llm_checks += 1
                        except ResponseError as exc:  # pragma: no cover - network/model specific
                            if "not found" in str(exc).lower():
                                handle_missing_model(self._mark_error, "Robot", self.cfg.robot_model)
                                raise SearchAbort from exc
                            if _is_context_length_error(str(exc)):
                                if self._rebuild_counts["relevance"] < MAX_REBUILD_RETRIES:
                                    self._reduce_context_and_rebuild("relevance", "relevance")
                                    try:
                                        relevance_raw = chains["result_filter"].invoke(
                                            self._inputs(
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
                                            )
                                        )
                                        relevance_decision = _regex_validate(str(relevance_raw), _PATTERN_YES_NO, "NO")
                                        relevance_llm_checks += 1
                                    except ResponseError:
                                        logging.info(
                                            "Relevance retry failed; skipping one result for '%s'.",
                                            current_query,
                                        )
                                        relevance_decision = "NO"
                                else:
                                    logging.info("Reached relevance rebuild cap; marking result as low relevance.")
                                    relevance_decision = "NO"
                            else:
                                logging.info(
                                    "Relevance check failed; skipping one result for '%s'. Error: %s",
                                    current_query,
                                    exc,
                                )
                                relevance_decision = "NO"
                        except Exception as exc:  # pragma: no cover - defensive
                            logging.info(
                                "Relevance check crashed; skipping one result for '%s'. Error: %s",
                                current_query,
                                exc,
                            )
                            relevance_decision = "NO"
                        if relevance_decision == "YES":
                            relevant = True
                        else:
                            continue
                aggregated_results.append(result_text)
                seen_result_hashes.add(result_hash)
                if norm_link:
                    seen_urls.add(norm_link)
                topic_keywords.update(_extract_keywords(keywords_source))
                accepted_any = True
            if not accepted_any:
                logging.info("No relevant results for '%s'. Not counting toward limit.", current_query)
                if round_index < len(pending_queries):
                    pending_queries.pop(round_index)
            else:
                round_index += 1
                if round_index >= max_rounds:
                    break
            remaining_slots = max_rounds - round_index
            if remaining_slots > 0:
                suggestion_limit = min(self.cfg.max_followup_suggestions, remaining_slots)
                results_to_date = "\n\n".join(aggregated_results) or "No results yet."
                results_to_date = _truncate_text(results_to_date, self._char_budget(MAX_SEARCH_RESULTS_CHARS))
                try:
                    suggestions_raw = chains["planning"].invoke(
                        self._inputs(
                            current_datetime,
                            current_year,
                            current_month,
                            current_day,
                            conversation_text,
                            user_query,
                            results_to_date=results_to_date,
                            suggestion_limit=str(suggestion_limit),
                            known_answers=prior_responses_text,
                        )
                    )
                except ResponseError as exc:  # pragma: no cover - network/model specific
                    if "not found" in str(exc).lower():
                        handle_missing_model(self._mark_error, "Robot", self.cfg.robot_model)
                        raise SearchAbort from exc
                    if _is_context_length_error(str(exc)):
                        if self._rebuild_counts["planning"] < MAX_REBUILD_RETRIES:
                            self._reduce_context_and_rebuild("planning", "planning")
                            try:
                                suggestions_raw = chains["planning"].invoke(
                                    self._inputs(
                                        current_datetime,
                                        current_year,
                                        current_month,
                                        current_day,
                                        conversation_text,
                                        user_query,
                                        results_to_date=results_to_date,
                                        suggestion_limit=str(suggestion_limit),
                                        known_answers=prior_responses_text,
                                    )
                                )
                            except ResponseError:
                                logging.info("Planning retry failed; skipping follow-up suggestions this round.")
                                suggestions_raw = "NONE"
                        else:
                            logging.info("Reached planning rebuild cap; no new suggestions this round.")
                            suggestions_raw = "NONE"
                    else:
                        logging.error("Query planning failed: %s", exc)
                        self._mark_error("Query planning failed; please retry.")
                        raise SearchAbort from exc
                except Exception as exc:  # pragma: no cover - defensive
                    logging.info(
                        "Planning failed unexpectedly; skipping suggestions this round. Error: %s",
                        exc,
                    )
                    suggestions_raw = "NONE"
                new_queries = self._parse_suggestions(suggestions_raw, suggestion_limit)
                for candidate in new_queries:
                    norm_candidate = _normalize_query(candidate)
                    if norm_candidate in seen_query_norms or len(pending_queries) >= max_rounds:
                        continue
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
                            continue
                    try:
                        verdict_raw = chains["query_filter"].invoke(
                            self._inputs(
                                current_datetime,
                                current_year,
                                current_month,
                                current_day,
                                conversation_text,
                                user_query,
                                candidate_query=candidate,
                            )
                        )
                    except ResponseError as exc:  # pragma: no cover - network/model specific
                        if "not found" in str(exc).lower():
                            handle_missing_model(self._mark_error, "Robot", self.cfg.robot_model)
                            raise SearchAbort from exc
                        if _is_context_length_error(str(exc)):
                            if self._rebuild_counts["query_filter"] < MAX_REBUILD_RETRIES:
                                self._reduce_context_and_rebuild("query_filter", "query filter")
                                try:
                                    verdict_raw = chains["query_filter"].invoke(
                                        self._inputs(
                                            current_datetime,
                                            current_year,
                                            current_month,
                                            current_day,
                                            conversation_text,
                                            user_query,
                                            candidate_query=candidate,
                                        )
                                    )
                                except ResponseError:
                                    logging.info("Skipping suggestion after retry: %s", candidate)
                                    continue
                            else:
                                logging.info(
                                    "Reached query filter rebuild cap; skipping candidate: %s",
                                    candidate,
                                )
                                continue
                        else:
                            logging.info(
                                "Skipping suggestion due to filter error: %s (%s)",
                                candidate,
                                exc,
                            )
                            continue
                    except Exception as exc:  # pragma: no cover - defensive
                        logging.info(
                            "Skipping suggestion due to unexpected filter error: %s (%s)",
                            candidate,
                            exc,
                        )
                        continue
                    verdict = _regex_validate(str(verdict_raw), _PATTERN_YES_NO, "NO")
                    if verdict == "YES" and norm_candidate not in seen_query_norms:
                        pending_queries.append(candidate)
                        seen_query_norms.add(norm_candidate)
                    else:
                        logging.info("Skipping off-topic follow-up suggestion: %s", candidate)
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
            results_to_date = "\n\n".join(aggregated_results) or "No results yet."
            results_to_date = _truncate_text(results_to_date, MAX_SEARCH_RESULTS_CHARS)
            try:
                suggestions_raw = chains["planning"].invoke(
                    self._inputs(
                        current_datetime,
                        current_year,
                        current_month,
                        current_day,
                        conversation_text,
                        user_query,
                        results_to_date=results_to_date,
                        suggestion_limit=str(suggestion_limit),
                        known_answers=prior_responses_text,
                    )
                )
            except ResponseError as exc:  # pragma: no cover - network/model specific
                if "not found" in str(exc).lower():
                    handle_missing_model(self._mark_error, "Robot", self.cfg.robot_model)
                    raise SearchAbort from exc
                if _is_context_length_error(str(exc)):
                    if self._rebuild_counts["planning"] < MAX_REBUILD_RETRIES:
                        self._reduce_context_and_rebuild("planning", "planning")
                        try:
                            suggestions_raw = chains["planning"].invoke(
                                self._inputs(
                                    current_datetime,
                                    current_year,
                                    current_month,
                                    current_day,
                                    conversation_text,
                                    user_query,
                                    results_to_date=results_to_date,
                                    suggestion_limit=str(suggestion_limit),
                                    known_answers=prior_responses_text,
                                )
                            )
                        except ResponseError:
                            logging.info("Planning retry failed during fill; stopping additional planning.")
                            break
                    else:
                        logging.info("Reached planning rebuild cap during fill; stopping additional planning.")
                        break
                else:
                    logging.error("Additional query planning failed: %s", exc)
                    self._mark_error("Additional query planning failed; please retry.")
                    break
            except Exception as exc:  # pragma: no cover - defensive
                logging.info(
                    "Planning crashed during fill; stopping additional planning. Error: %s",
                    exc,
                )
                break
            fill_queries = self._parse_suggestions(suggestions_raw, suggestion_limit)
            for candidate in fill_queries:
                norm_candidate = _normalize_query(candidate)
                if norm_candidate in seen_query_norms or len(pending_queries) >= self.cfg.max_rounds:
                    continue
                candidate_embedding = self._embedding_client.embed(candidate)
                if candidate_embedding is not None and (question_embedding or topic_embedding_current):
                    similarity = self._context_similarity(
                        candidate_embedding,
                        question_embedding,
                        topic_embedding_current,
                    )
                    if similarity < self.cfg.embedding_query_similarity_threshold:
                        logging.info(
                            "Skipping fill suggestion with low semantic similarity (%.2f): %s",
                            similarity,
                            candidate,
                        )
                        continue
                try:
                    verdict_raw = chains["query_filter"].invoke(
                        self._inputs(
                            current_datetime,
                            current_year,
                            current_month,
                            current_day,
                            conversation_text,
                            user_query,
                            candidate_query=candidate,
                        )
                    )
                except ResponseError as exc:  # pragma: no cover - network/model specific
                    if "not found" in str(exc).lower():
                        handle_missing_model(self._mark_error, "Robot", self.cfg.robot_model)
                        raise SearchAbort from exc
                    if _is_context_length_error(str(exc)):
                        if self._rebuild_counts["query_filter"] < MAX_REBUILD_RETRIES:
                            self._reduce_context_and_rebuild("query_filter", "query filter")
                            try:
                                verdict_raw = chains["query_filter"].invoke(
                                    self._inputs(
                                        current_datetime,
                                        current_year,
                                        current_month,
                                        current_day,
                                        conversation_text,
                                        user_query,
                                        candidate_query=candidate,
                                    )
                                )
                            except ResponseError:
                                logging.info(
                                    "Skipping fill suggestion after retry: %s",
                                    candidate,
                                )
                                continue
                        else:
                            logging.info(
                                "Reached query filter rebuild cap; skipping fill candidate: %s",
                                candidate,
                            )
                            continue
                    else:
                        logging.info(
                            "Skipping fill suggestion due to filter error: %s (%s)",
                            candidate,
                            exc,
                        )
                        continue
                except Exception as exc:  # pragma: no cover - defensive
                    logging.info(
                        "Skipping fill suggestion due to unexpected filter error: %s (%s)",
                        candidate,
                        exc,
                    )
                    continue
                verdict = _regex_validate(str(verdict_raw), _PATTERN_YES_NO, "NO")
                if verdict == "YES" and norm_candidate not in seen_query_norms:
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
