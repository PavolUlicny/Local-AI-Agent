from __future__ import annotations

from typing import List, Set
import argparse
from datetime import datetime, timezone
import logging
import sys
import hashlib
import time
import random
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_ollama import OllamaLLM
except ImportError as exc:
    raise ImportError(
        "langchain-ollama is required. Install it with 'pip install -U langchain-ollama'."
    ) from exc

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

try:
    from ollama import ResponseError as _OllamaResponseError
    ResponseError = _OllamaResponseError
except ImportError:
    try:
        from ollama._types import ResponseError as _OllamaResponseError
        ResponseError = _OllamaResponseError
    except ImportError:
        class ResponseError(Exception):
            pass

# Imports from our split modules (prompts, helpers)
try:
    from src.prompts import (
        response_prompt_no_search,
        response_prompt,
        search_decision_prompt,
        planning_prompt,
        seed_prompt,
        query_filter_prompt,
        result_filter_prompt,
        context_mode_prompt,
    )
    from src.helpers import (
        Topic,
        _truncate_result,
        _normalize_query,
        _regex_validate,
        _truncate_text,
        _current_datetime_utc,
        _is_context_length_error,
        _pick_seed_query,
        _extract_keywords,
        _is_relevant,
        _format_turns,
        _collect_prior_responses,
        _select_topic,
        _prune_keywords,
        _canonicalize_url,
        _PATTERN_SEARCH_DECISION,
        _PATTERN_YES_NO,
        MAX_CONVERSATION_CHARS,
        MAX_PRIOR_RESPONSE_CHARS,
        MAX_SEARCH_RESULTS_CHARS,
        MAX_TURN_KEYWORD_SOURCE_CHARS,
        MAX_TOPICS,
        MAX_REBUILD_RETRIES,
    )
except ImportError:
    # Fallback if executed directly from src/ without package context
    from prompts import (
        response_prompt_no_search,
        response_prompt,
        search_decision_prompt,
        planning_prompt,
        seed_prompt,
        query_filter_prompt,
        result_filter_prompt,
        context_mode_prompt,
    )
    from helpers import (
        Topic,
        _truncate_result,
        _normalize_query,
        _regex_validate,
        _truncate_text,
        _current_datetime_utc,
        _is_context_length_error,
        _pick_seed_query,
        _extract_keywords,
        _is_relevant,
        _format_turns,
        _collect_prior_responses,
        _select_topic,
        _prune_keywords,
        _canonicalize_url,
        _PATTERN_SEARCH_DECISION,
        _PATTERN_YES_NO,
        MAX_CONVERSATION_CHARS,
        MAX_PRIOR_RESPONSE_CHARS,
        MAX_SEARCH_RESULTS_CHARS,
        MAX_TURN_KEYWORD_SOURCE_CHARS,
        MAX_TOPICS,
        MAX_REBUILD_RETRIES,
    )

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local AI Agent with optional web search")
    parser.add_argument("--model", default="cogito:8b", help="Ollama model to use")
    parser.add_argument("--no-auto-search", action="store_true", help="Disable automatic web search decision")
    parser.add_argument("--max-rounds", type=int, default=12, help="Maximum search rounds")
    parser.add_argument("--max-context-turns", type=int, default=8, help="Max turns from topic to keep in context")
    parser.add_argument("--max-followup-suggestions", type=int, default=6, help="Max planning suggestions per cycle")
    parser.add_argument("--max-fill-attempts", type=int, default=3, help="How many planning fill attempts")
    parser.add_argument("--max-relevance-llm-checks", type=int, default=2, help="LLM checks allowed for borderline relevance")
    parser.add_argument("--num-ctx", type=int, default=8192, help="Model context window tokens")
    parser.add_argument("--num-predict", type=int, default=8192, help="Max tokens to predict")
    parser.add_argument("--robot-temp", type=float, default=0.0, help="Temperature for robot LLM")
    parser.add_argument("--assistant-temp", type=float, default=0.7, help="Temperature for assistant LLM")
    parser.add_argument("--robot-top-p", type=float, default=0.4, help="Top-p for robot LLM")
    parser.add_argument("--assistant-top-p", type=float, default=0.8, help="Top-p for assistant LLM")
    parser.add_argument("--robot-top-k", type=int, default=20, help="Top-k for robot LLM")
    parser.add_argument("--assistant-top-k", type=int, default=80, help="Top-k for assistant LLM")
    parser.add_argument("--robot-repeat-penalty", type=float, default=1.1, help="Repeat penalty for robot LLM")
    parser.add_argument("--assistant-repeat-penalty", type=float, default=1.2, help="Repeat penalty for assistant LLM")
    parser.add_argument("--ddg-region", default="us-en", help="DuckDuckGo region, e.g. us-en, uk-en, de-de")
    parser.add_argument("--ddg-safesearch", default="moderate", choices=["off", "moderate", "strict"], help="DuckDuckGo safesearch level")
    parser.add_argument("--ddg-backend", default="html", choices=["html", "lite", "api"], help="DuckDuckGo backend to use")
    parser.add_argument("--search-max-results", type=int, default=5, help="Max results to fetch per query")
    parser.add_argument("--search-retries", type=int, default=4, help="Retry attempts for transient search errors")
    parser.add_argument("--log-level", default="WARNING", help="Logging level: DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--log-file", default=None, help="Optional log file path")
    parser.add_argument("--question", default=None, help="Run once with this question and exit")
    return parser


def _configure_logging(level: str, log_file: str | None) -> None:
    level_upper = (level or "INFO").upper()
    numeric = getattr(logging, level_upper, logging.INFO)
    handlers = []
    console_handler = logging.StreamHandler(sys.stderr)
    handlers.append(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = _build_arg_parser()
        args = parser.parse_args()
    _configure_logging(args.log_level, args.log_file)
    
    # Settings (configurable via CLI)
    max_context_turns = args.max_context_turns
    max_search_rounds = args.max_rounds
    max_followup_suggestions = args.max_followup_suggestions
    max_fill_attempts = args.max_fill_attempts
    max_relevance_llm_checks = args.max_relevance_llm_checks
    auto_search_decision = not args.no_auto_search
    used_model = args.model
    num_predict_ = args.num_predict
    num_ctx_ = args.num_ctx
    ddg_region = args.ddg_region
    ddg_safesearch = args.ddg_safesearch
    ddg_backend = args.ddg_backend
    search_max_results = args.search_max_results
    search_retries = args.search_retries
    
    robot_temperature = args.robot_temp
    assistant_temperature = args.assistant_temp
    robot_top_p_ = args.robot_top_p
    assistant_top_p_ = args.assistant_top_p
    robot_top_k_ = args.robot_top_k
    assistant_top_k_ = args.assistant_top_k
    robot_repeat_penalty_ = args.robot_repeat_penalty
    assistant_repeat_penalty_ = args.assistant_repeat_penalty
    
    llm_robot = OllamaLLM(
        model=used_model,
        temperature=robot_temperature,
        top_p=robot_top_p_,
        top_k=robot_top_k_,
        repeat_penalty=robot_repeat_penalty_,
        num_predict=num_predict_,
        num_ctx=num_ctx_,
    )
    llm_assistant = OllamaLLM(
        model=used_model,
        temperature=assistant_temperature,
        top_p=assistant_top_p_,
        top_k=assistant_top_k_,
        repeat_penalty=assistant_repeat_penalty_,
        num_predict=num_predict_,
        num_ctx=num_ctx_,
    )
    search_api = DuckDuckGoSearchAPIWrapper(region=ddg_region, safesearch=ddg_safesearch, backend=ddg_backend)

    rebuild_counts = {
        "search_decision": 0,
        "seed": 0,
        "relevance": 0,
        "planning": 0,
        "query_filter": 0,
        "answer": 0,
    }

    # Prompts are now imported from src.prompts

    context_chain = context_mode_prompt | llm_robot | StrOutputParser()
    seed_chain = seed_prompt | llm_robot | StrOutputParser()
    planning_chain = planning_prompt | llm_robot | StrOutputParser()
    result_filter_chain = result_filter_prompt | llm_robot | StrOutputParser()
    query_filter_chain = query_filter_prompt | llm_robot | StrOutputParser()
    search_decision_chain = search_decision_prompt | llm_robot | StrOutputParser()

    topics: List[Topic] = []

    def _reduce_context_and_rebuild(stage_key: str, label: str) -> None:
        nonlocal num_ctx_, num_predict_
        rebuild_counts[stage_key] += 1
        reduced_ctx = max(2048, num_ctx_ // 2)
        reduced_predict = max(512, min(num_predict_, reduced_ctx // 2))
        logging.info(
            f"Context too large ({label}); rebuild {rebuild_counts[stage_key]}/{MAX_REBUILD_RETRIES} with num_ctx={reduced_ctx}, num_predict={reduced_predict}."
        )
        _rebuild_llms(reduced_ctx, reduced_predict)

    def _rebuild_llms(new_num_ctx: int, new_num_predict: int) -> None:
        nonlocal llm_robot, llm_assistant, num_ctx_, num_predict_, context_chain, seed_chain, planning_chain, result_filter_chain, query_filter_chain, search_decision_chain
        num_ctx_ = new_num_ctx
        num_predict_ = new_num_predict
        llm_robot = OllamaLLM(
            model=used_model,
            temperature=robot_temperature,
            top_p=robot_top_p_,
            top_k=robot_top_k_,
            repeat_penalty=robot_repeat_penalty_,
            num_predict=num_predict_,
            num_ctx=num_ctx_,
        )
        llm_assistant = OllamaLLM(
            model=used_model,
            temperature=assistant_temperature,
            top_p=assistant_top_p_,
            top_k=assistant_top_k_,
            repeat_penalty=assistant_repeat_penalty_,
            num_predict=num_predict_,
            num_ctx=num_ctx_,
        )
        
        context_chain = context_mode_prompt | llm_robot | StrOutputParser()
        seed_chain = seed_prompt | llm_robot | StrOutputParser()
        planning_chain = planning_prompt | llm_robot | StrOutputParser()
        result_filter_chain = result_filter_prompt | llm_robot | StrOutputParser()
        query_filter_chain = query_filter_prompt | llm_robot | StrOutputParser()
        search_decision_chain = search_decision_prompt | llm_robot | StrOutputParser()

    def _ddg_results(query: str, max_results: int = 5, retries: int = 4) -> List[dict]:
        delay = 1.0
        for attempt in range(1, retries + 1):
            try:
                return search_api.results(query, max_results=max_results)
            except Exception as exc:
                msg = str(exc)
                is_rate = (
                    "Ratelimit" in msg
                    or "429" in msg
                    or " 202 " in msg
                    or "rate limit" in msg.lower()
                )
                if is_rate:
                    logging.warning(
                        f"Rate limited for '{query}' (attempt {attempt}/{retries}); sleeping {delay:.1f}s"
                    )
                else:
                    logging.warning(
                        f"Search error for '{query}' (attempt {attempt}/{retries}): {exc}"
                    )
                time.sleep(delay + (random.random() * 0.5))
                delay = min(delay * 2.0, 8.0)
        logging.warning(f"Search failed after {retries} attempts for '{query}'.")
        return []

    while True:
        try:
            if args.question:
                user_query = args.question.strip()
            else:
                prompt_text = "\n\033[92mEnter your request (or 'exit' to quit): \033[0m"
                user_query = input(prompt_text).strip()
        except (KeyboardInterrupt, EOFError):
            logging.info("Exiting due to interrupt/EOF.")
            return

        if not user_query:
            logging.info("No input provided.")
            continue

        if not args.question and user_query.lower() in {"exit", "quit"}:
            logging.info("Goodbye!")
            return

        current_datetime = _current_datetime_utc()
        # Discrete date components for prompt grounding
        dt_obj = datetime.now(timezone.utc)
        current_year = str(dt_obj.year)
        current_month = f"{dt_obj.month:02d}"
        current_day = f"{dt_obj.day:02d}"
        question_keywords = _extract_keywords(user_query)
        try:
            (
                selected_topic_index,
                recent_history,
                topic_keywords,
            ) = _select_topic(
                context_chain,
                topics,
                user_query,
                question_keywords,
                max_context_turns,
                current_datetime,
                current_year,
                current_month,
                current_day,
            )
        except ResponseError as exc:
            message = str(exc)
            if "not found" in message.lower():
                logging.error(
                    f"Model '{used_model}' not found. Run 'ollama pull {used_model}' and retry."
                )
                return
            selected_topic_index = None
            recent_history = []
            topic_keywords = set(question_keywords)
        except Exception as exc:
            logging.warning(f"Context classification failed; proceeding without topic selection. Error: {exc}")
            selected_topic_index = None
            recent_history = []
            topic_keywords = set(question_keywords)

        conversation_text = _format_turns(
            recent_history, "No prior relevant conversation."
        )
        conversation_text = _truncate_text(conversation_text, MAX_CONVERSATION_CHARS)

        prior_responses_text = (
            _collect_prior_responses(topics[selected_topic_index], max_chars=MAX_PRIOR_RESPONSE_CHARS)
            if selected_topic_index is not None
            else "No prior answers for this topic."
        )
        prior_responses_text = _truncate_text(prior_responses_text, MAX_PRIOR_RESPONSE_CHARS)

        aggregated_results: List[str] = []
        seen_results: Set[str] = set()
        seen_result_hashes: Set[str] = set()
        seen_urls: Set[str] = set()

        # Default: if auto decision is off, do not search unless explicitly enabled
        should_search = False
        if auto_search_decision:
            try:
                decision_raw = search_decision_chain.invoke(
                    {
                        "current_datetime": current_datetime,
                        "current_year": current_year,
                        "current_month": current_month,
                        "current_day": current_day,
                        "conversation_history": conversation_text,
                        "user_question": user_query,
                        "known_answers": prior_responses_text,
                    }
                )
                decision_validated = _regex_validate(str(decision_raw), _PATTERN_SEARCH_DECISION, "SEARCH")
                should_search = decision_validated == "SEARCH"
            except ResponseError as exc:
                if "not found" in str(exc).lower():
                    logging.error(
                        f"Model '{used_model}' not found. Run 'ollama pull {used_model}' and retry."
                    )
                    return
                if _is_context_length_error(str(exc)):
                    if rebuild_counts["search_decision"] < MAX_REBUILD_RETRIES:
                        _reduce_context_and_rebuild("search_decision", "search decision")
                        try:
                            decision_raw = search_decision_chain.invoke(
                                {
                                    "current_datetime": current_datetime,
                                    "current_year": current_year,
                                    "current_month": current_month,
                                    "current_day": current_day,
                                    "conversation_history": conversation_text,
                                    "user_question": user_query,
                                    "known_answers": prior_responses_text,
                                }
                            )
                            decision_validated = _regex_validate(str(decision_raw), _PATTERN_SEARCH_DECISION, "SEARCH")
                            should_search = decision_validated == "SEARCH"
                        except ResponseError:
                            should_search = False
                    else:
                        logging.info("Reached search decision rebuild cap; defaulting to NO_SEARCH fallback.")
                        should_search = False
                else:
                    logging.warning("Search decision failed with non-context error; defaulting to NO_SEARCH.")
                    should_search = False
            except Exception as exc:
                logging.warning(f"Search decision failed unexpectedly; defaulting to NO_SEARCH. Error: {exc}")
                should_search = False

        if should_search:
            primary_search_query = user_query
            try:
                seed_response = seed_chain.invoke(
                    {
                        "current_datetime": current_datetime,
                        "current_year": current_year,
                        "current_month": current_month,
                        "current_day": current_day,
                        "conversation_history": conversation_text,
                        "user_question": user_query,
                        "known_answers": prior_responses_text,
                    }
                )
                seed_text = str(seed_response).strip()
                primary_search_query = _pick_seed_query(seed_text, user_query)
            except ResponseError as exc:
                if "not found" in str(exc).lower():
                    logging.error(
                        f"Model '{used_model}' not found. Run 'ollama pull {used_model}' and retry."
                    )
                    return
                if _is_context_length_error(str(exc)):
                    if rebuild_counts["seed"] < MAX_REBUILD_RETRIES:
                        _reduce_context_and_rebuild("seed", "seed")
                        try:
                            seed_response = seed_chain.invoke(
                                {
                                    "current_datetime": current_datetime,
                                    "current_year": current_year,
                                    "current_month": current_month,
                                    "current_day": current_day,
                                    "conversation_history": conversation_text,
                                    "user_question": user_query,
                                    "known_answers": prior_responses_text,
                                }
                            )
                            seed_text = str(seed_response).strip()
                            primary_search_query = _pick_seed_query(seed_text, user_query)
                        except ResponseError:
                            primary_search_query = user_query
                    else:
                        logging.info("Reached seed rebuild cap; using original user query as search seed.")
                        primary_search_query = user_query
                else:
                    logging.warning("Seed generation failed; using original user query as search seed.")
                    primary_search_query = user_query
            except Exception as exc:
                logging.warning(f"Seed generation failed unexpectedly; using original query. Error: {exc}")
                primary_search_query = user_query

            pending_queries: List[str] = [primary_search_query]
            seen_query_norms: Set[str] = { _normalize_query(primary_search_query) }
            if not topic_keywords:
                topic_keywords.update(_extract_keywords(user_query))
                topic_keywords.update(_extract_keywords(primary_search_query))
            max_rounds = max_search_rounds

            round_index = 0
            while round_index < len(pending_queries) and round_index < max_rounds:
                current_query = pending_queries[round_index]
                # Multi-result search: fetch top N results and evaluate each
                results_list = _ddg_results(current_query, max_results=search_max_results, retries=search_retries)

                # Iterate each result item
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
                        part for part in [
                            (f"Title: {title}" if title else ""),
                            (f"URL: {link}" if link else ""),
                            (f"Snippet: {snippet}" if snippet else ""),
                        ]
                        if part
                    )
                    # dedupe using stable hash of full assembled text (pre-truncation)
                    result_hash = hashlib.sha1(assembled.encode("utf-8", errors="ignore")).hexdigest()
                    if result_hash in seen_result_hashes:
                        continue
                    result_text = _truncate_result(assembled)
                    normalized_result = result_text.strip()
                    if normalized_result and normalized_result in seen_results:
                        continue

                    relevant = _is_relevant(result_text, topic_keywords)
                    if not relevant:
                        if relevance_llm_checks >= max_relevance_llm_checks:
                            continue
                        kw_list = sorted(topic_keywords) if topic_keywords else []
                        if len(kw_list) > 50:
                            kw_list = kw_list[:50]
                        topic_keywords_text = ", ".join(kw_list) if kw_list else "None"
                        topic_keywords_text = _truncate_text(topic_keywords_text, 1000)
                        try:
                            relevance_raw = result_filter_chain.invoke(
                                {
                                    "current_datetime": current_datetime,
                                    "current_year": current_year,
                                    "current_month": current_month,
                                    "current_day": current_day,
                                    "user_question": user_query,
                                    "search_query": current_query,
                                    "raw_result": result_text,
                                    "known_answers": prior_responses_text,
                                    "topic_keywords": topic_keywords_text,
                                }
                            )
                            relevance_decision = _regex_validate(str(relevance_raw), _PATTERN_YES_NO, "NO")
                            relevance_llm_checks += 1
                        except ResponseError as exc:
                            if "not found" in str(exc).lower():
                                logging.error(
                                    f"Model '{used_model}' not found. Run 'ollama pull {used_model}' and retry."
                                )
                                return
                            if _is_context_length_error(str(exc)):
                                if rebuild_counts["relevance"] < MAX_REBUILD_RETRIES:
                                    _reduce_context_and_rebuild("relevance", "relevance")
                                    try:
                                        relevance_raw = result_filter_chain.invoke(
                                            {
                                                "current_datetime": current_datetime,
                                                "current_year": current_year,
                                                "current_month": current_month,
                                                "current_day": current_day,
                                                "user_question": user_query,
                                                "search_query": current_query,
                                                "raw_result": result_text,
                                                "known_answers": prior_responses_text,
                                                "topic_keywords": topic_keywords_text,
                                            }
                                        )
                                        relevance_decision = _regex_validate(str(relevance_raw), _PATTERN_YES_NO, "NO")
                                        relevance_llm_checks += 1
                                    except ResponseError:
                                        logging.info(f"Relevance retry failed; skipping one result for '{current_query}'.")
                                        relevance_decision = "NO"
                                else:
                                    logging.info("Reached relevance rebuild cap; marking result as low relevance.")
                                    relevance_decision = "NO"
                            else:
                                logging.info(f"Relevance check failed; skipping one result for '{current_query}'. Error: {exc}")
                                relevance_decision = "NO"
                        except Exception as exc:
                            logging.info(f"Relevance check crashed; skipping one result for '{current_query}'. Error: {exc}")
                            relevance_decision = "NO"
                        if relevance_decision == "YES":
                            relevant = True
                        else:
                            continue

                    aggregated_results.append(result_text)
                    seen_results.add(normalized_result)
                    seen_result_hashes.add(result_hash)
                    if norm_link:
                        seen_urls.add(norm_link)
                    # Avoid label tokens by extracting from title+snippet only
                    keywords_source = " ".join([part for part in [title, snippet] if part])
                    topic_keywords.update(_extract_keywords(keywords_source))
                    accepted_any = True

                # If no relevant results were accepted, do not consume a round; drop this query
                if not accepted_any:
                    logging.info(f"No relevant results for '{current_query}'. Not counting toward limit.")
                    if round_index < len(pending_queries):
                        pending_queries.pop(round_index)
                else:
                    round_index += 1
                    if round_index >= max_rounds:
                        break

                remaining_slots = max_rounds - len(pending_queries)
                if remaining_slots <= 0:
                    continue

                suggestion_limit = min(max_followup_suggestions, remaining_slots)
                results_to_date = "\n\n".join(aggregated_results) or "No results yet."
                results_to_date = _truncate_text(
                    results_to_date, MAX_SEARCH_RESULTS_CHARS
                )

                try:
                    suggestions_raw = planning_chain.invoke(
                        {
                            "current_datetime": current_datetime,
                            "current_year": current_year,
                            "current_month": current_month,
                            "current_day": current_day,
                            "conversation_history": conversation_text,
                            "user_question": user_query,
                            "results_to_date": results_to_date,
                            "suggestion_limit": str(suggestion_limit),
                            "known_answers": prior_responses_text,
                        }
                    )
                except ResponseError as exc:
                    if "not found" in str(exc).lower():
                        logging.error(
                            f"Model '{used_model}' not found. Run 'ollama pull {used_model}' and retry."
                        )
                        return
                    if _is_context_length_error(str(exc)):
                        if rebuild_counts["planning"] < MAX_REBUILD_RETRIES:
                            rebuild_counts["planning"] += 1
                            reduced_ctx = max(2048, num_ctx_ // 2)
                            reduced_predict = max(512, min(num_predict_, reduced_ctx // 2))
                            logging.info(
                                f"Context too large (planning); rebuild {rebuild_counts['planning']}/{MAX_REBUILD_RETRIES} with num_ctx={reduced_ctx}, num_predict={reduced_predict}."
                            )
                            _rebuild_llms(reduced_ctx, reduced_predict)
                            try:
                                suggestions_raw = planning_chain.invoke(
                                    {
                                        "current_datetime": current_datetime,
                                        "current_year": current_year,
                                        "current_month": current_month,
                                        "current_day": current_day,
                                        "conversation_history": conversation_text,
                                        "user_question": user_query,
                                        "results_to_date": results_to_date,
                                        "suggestion_limit": str(suggestion_limit),
                                        "known_answers": prior_responses_text,
                                    }
                                )
                            except ResponseError:
                                logging.info("Planning retry failed; skipping follow-up suggestions this round.")
                                suggestions_raw = "NONE"
                        else:
                            logging.info("Reached planning rebuild cap; no new suggestions this round.")
                            suggestions_raw = "NONE"
                    else:
                        logging.info(f"Could not plan follow-up searches: {exc}")
                        suggestions_raw = "NONE"
                except Exception as exc:
                    logging.info(f"Planning failed unexpectedly; skipping suggestions this round. Error: {exc}")
                    suggestions_raw = "NONE"

                new_queries: List[str] = []
                for line in str(suggestions_raw).splitlines():
                    normalized = line.strip().strip("-*\"'").strip()
                    if not normalized:
                        continue
                    if normalized.lower() == "none":
                        new_queries = []
                        break
                    new_queries.append(normalized)

                if new_queries:
                    new_queries = new_queries[:suggestion_limit]

                for candidate in new_queries:
                    # Early normalize to skip near-duplicates before LLM filtering
                    norm_candidate = _normalize_query(candidate)
                    if norm_candidate in seen_query_norms or len(pending_queries) >= max_rounds:
                        continue
                    try:
                        verdict_raw = query_filter_chain.invoke(
                            {
                                "current_datetime": current_datetime,
                                "current_year": current_year,
                                "current_month": current_month,
                                "current_day": current_day,
                                "candidate_query": candidate,
                                "conversation_history": conversation_text,
                                "user_question": user_query,
                            }
                        )
                    except ResponseError as exc:
                        if "not found" in str(exc).lower():
                            logging.error(
                                f"Model '{used_model}' not found. Run 'ollama pull {used_model}' and retry."
                            )
                            return
                        if _is_context_length_error(str(exc)):
                            if rebuild_counts["query_filter"] < MAX_REBUILD_RETRIES:
                                _reduce_context_and_rebuild("query_filter", "query filter")
                                try:
                                    verdict_raw = query_filter_chain.invoke(
                                        {
                                            "current_datetime": current_datetime,
                                            "current_year": current_year,
                                            "current_month": current_month,
                                            "current_day": current_day,
                                            "candidate_query": candidate,
                                            "conversation_history": conversation_text,
                                            "user_question": user_query,
                                        }
                                    )
                                except ResponseError:
                                    logging.info(f"Skipping suggestion after retry: {candidate}")
                                    continue
                            else:
                                logging.info(f"Reached query filter rebuild cap; skipping candidate: {candidate}")
                                continue
                        else:
                            logging.info(f"Skipping suggestion due to filter error: {candidate} ({exc})")
                            continue
                    except Exception as exc:
                        logging.info(f"Skipping suggestion due to unexpected filter error: {candidate} ({exc})")
                        continue
                    verdict = _regex_validate(str(verdict_raw), _PATTERN_YES_NO, "NO")
                    if verdict == "YES" and norm_candidate not in seen_query_norms:
                        pending_queries.append(candidate)
                        seen_query_norms.add(norm_candidate)
                    else:
                        logging.info(f"Skipping off-topic follow-up suggestion: {candidate}")

                # Ensure query slots are filled: keep planning until pending is full or no progress
                fill_attempts = 0
                while len(pending_queries) < max_rounds and fill_attempts < max_fill_attempts:
                    fill_attempts += 1
                    remaining_slots = max_rounds - len(pending_queries)
                    if remaining_slots <= 0:
                        break
                    suggestion_limit = min(max_followup_suggestions, remaining_slots)
                    results_to_date = "\n\n".join(aggregated_results) or "No results yet."
                    results_to_date = _truncate_text(results_to_date, MAX_SEARCH_RESULTS_CHARS)

                    try:
                        suggestions_raw = planning_chain.invoke(
                            {
                                "current_datetime": current_datetime,
                                "current_year": current_year,
                                "current_month": current_month,
                                "current_day": current_day,
                                "conversation_history": conversation_text,
                                "user_question": user_query,
                                "results_to_date": results_to_date,
                                "suggestion_limit": str(suggestion_limit),
                                "known_answers": prior_responses_text,
                            }
                        )
                    except ResponseError as exc:
                        if "not found" in str(exc).lower():
                            logging.error(
                                f"Model '{used_model}' not found. Run 'ollama pull {used_model}' and retry."
                            )
                            break
                        if _is_context_length_error(str(exc)):
                            if rebuild_counts["planning"] < MAX_REBUILD_RETRIES:
                                _reduce_context_and_rebuild("planning", "planning")
                                try:
                                    suggestions_raw = planning_chain.invoke(
                                        {
                                            "current_datetime": current_datetime,
                                            "current_year": current_year,
                                            "current_month": current_month,
                                            "current_day": current_day,
                                            "conversation_history": conversation_text,
                                            "user_question": user_query,
                                            "results_to_date": results_to_date,
                                            "suggestion_limit": str(suggestion_limit),
                                            "known_answers": prior_responses_text,
                                        }
                                    )
                                except ResponseError:
                                    logging.info("Planning retry failed during fill; stopping additional planning.")
                                    break
                            else:
                                logging.info("Reached planning rebuild cap during fill; stopping additional planning.")
                                break
                        else:
                            logging.info(f"Could not plan additional fill queries: {exc}")
                            break
                    except Exception as exc:
                        logging.info(f"Planning crashed during fill; stopping additional planning. Error: {exc}")
                        break

                    fill_queries: List[str] = []
                    for line in str(suggestions_raw).splitlines():
                        normalized = line.strip().strip("-*\"'").strip()
                        if not normalized:
                            continue
                        if normalized.lower() == "none":
                            fill_queries = []
                            break
                        fill_queries.append(normalized)

                    if fill_queries:
                        fill_queries = fill_queries[:suggestion_limit]

                    for candidate in fill_queries:
                        # Early normalize to skip near-duplicates before LLM filtering
                        norm_candidate = _normalize_query(candidate)
                        if norm_candidate in seen_query_norms or len(pending_queries) >= max_rounds:
                            continue
                        try:
                            verdict_raw = query_filter_chain.invoke(
                                {
                                    "current_datetime": current_datetime,
                                    "current_year": current_year,
                                    "current_month": current_month,
                                    "current_day": current_day,
                                    "candidate_query": candidate,
                                    "conversation_history": conversation_text,
                                    "user_question": user_query,
                                }
                            )
                        except ResponseError as exc:
                            if "not found" in str(exc).lower():
                                logging.error(
                                    f"Model '{used_model}' not found. Run 'ollama pull {used_model}' and retry."
                                )
                                continue
                            if _is_context_length_error(str(exc)):
                                if rebuild_counts["query_filter"] < MAX_REBUILD_RETRIES:
                                    _reduce_context_and_rebuild("query_filter", "query filter")
                                    try:
                                        verdict_raw = query_filter_chain.invoke(
                                            {
                                                "current_datetime": current_datetime,
                                                "current_year": current_year,
                                                "current_month": current_month,
                                                "current_day": current_day,
                                                "candidate_query": candidate,
                                                "conversation_history": conversation_text,
                                                "user_question": user_query,
                                            }
                                        )
                                    except ResponseError:
                                        logging.info(f"Skipping fill suggestion after retry: {candidate}")
                                        continue
                                else:
                                    logging.info(f"Reached query filter rebuild cap; skipping fill candidate: {candidate}")
                                    continue
                            else:
                                logging.info(f"Skipping fill suggestion due to filter error: {candidate} ({exc})")
                                continue
                        except Exception as exc:
                            logging.info(f"Skipping fill suggestion due to unexpected filter error: {candidate} ({exc})")
                            continue
                        verdict = _regex_validate(str(verdict_raw), _PATTERN_YES_NO, "NO")
                        if verdict == "YES" and norm_candidate not in seen_query_norms:
                            pending_queries.append(candidate)
                            seen_query_norms.add(norm_candidate)
                        else:
                            continue
        if should_search:
            search_results_text = (
                "\n\n".join(aggregated_results)
                if aggregated_results
                else "No search results collected."
            )
            search_results_text = _truncate_text(
                search_results_text, MAX_SEARCH_RESULTS_CHARS
            )
        else:
            search_results_text = "No web search performed."

        if should_search:
            inputs = {
                "current_datetime": current_datetime,
                "current_year": current_year,
                "current_month": current_month,
                "current_day": current_day,
                "conversation_history": conversation_text,
                "search_results": search_results_text,
                "user_question": user_query,
                "prior_responses": prior_responses_text,
            }
            chain = response_prompt | llm_assistant | StrOutputParser()
        else:
            inputs = {
                "current_datetime": current_datetime,
                "current_year": current_year,
                "current_month": current_month,
                "current_day": current_day,
                "conversation_history": conversation_text,
                "user_question": user_query,
                "prior_responses": prior_responses_text,
            }
            chain = response_prompt_no_search | llm_assistant | StrOutputParser()

        try:
            response_stream = chain.stream(inputs)
        except ResponseError as exc:
            if "not found" in str(exc).lower():
                logging.error(
                    f"Model '{used_model}' not found. Run 'ollama pull {used_model}' and retry."
                )
                return
            if _is_context_length_error(str(exc)):
                if rebuild_counts["answer"] < MAX_REBUILD_RETRIES:
                    _reduce_context_and_rebuild("answer", "answer")
                    try:
                        chain = (response_prompt if should_search else response_prompt_no_search) | llm_assistant | StrOutputParser()
                        response_stream = chain.stream(inputs)
                    except ResponseError as exc2:
                        logging.error(f"Answer generation failed after retry: {exc2}")
                        return
                else:
                    logging.error("Reached answer generation rebuild cap; please shorten your query or reset session.")
                    return
            else:
                raise
        except Exception as exc:
            logging.error(f"Answer generation failed unexpectedly: {exc}")
            return

        print("\n\033[91m[Answer]\033[0m")
        response_chunks: List[str] = []
        try:
            for chunk in response_stream:
                response_chunks.append(chunk)
                print(chunk, end="", flush=True)
        except KeyboardInterrupt:
            logging.info("Streaming interrupted by user.")
        except Exception as exc:
            logging.error(f"Streaming error: {exc}")

        if response_chunks and not response_chunks[-1].endswith("\n"):
            print()

        response_text = "".join(response_chunks)

        if selected_topic_index is None:
            topics.append(Topic(keywords=set(topic_keywords)))
            selected_topic_index = len(topics) - 1

        if selected_topic_index != len(topics) - 1:
            moved_topic = topics.pop(selected_topic_index)
            topics.append(moved_topic)
            selected_topic_index = len(topics) - 1

        while len(topics) > MAX_TOPICS:
            topics.pop(0)
            selected_topic_index = max(0, selected_topic_index - 1)

        topic_entry = topics[selected_topic_index]
        topic_entry.turns.append((user_query, response_text))
        if len(topic_entry.turns) > max_context_turns * 2:
            topic_entry.turns = topic_entry.turns[-max_context_turns * 2 :]

        aggregated_keyword_source = _truncate_text(
            " ".join(aggregated_results), MAX_TURN_KEYWORD_SOURCE_CHARS
        )
        turn_keywords = _extract_keywords(
            " ".join([user_query, response_text, aggregated_keyword_source])
        )
        if not turn_keywords:
            turn_keywords = set(question_keywords)

        topic_entry.keywords.update(turn_keywords)
        topic_entry.keywords.update(topic_keywords)
        _prune_keywords(topic_entry)

        # If running in one-shot mode via --question, exit after first answer
        if args.question:
            return

if __name__ == "__main__":
    main()