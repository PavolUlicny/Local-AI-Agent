from __future__ import annotations

from typing import Any, List, Set, TYPE_CHECKING, cast
import hashlib
import importlib
import logging
import random
import sys
import time
from datetime import datetime, timezone
from typing import TextIO

from ddgs import DDGS
from ddgs.exceptions import DDGSException, TimeoutException
from langchain_ollama import OllamaEmbeddings

PromptSession: Any | None
ANSI: Any | None
InMemoryHistory: Any | None

try:
    from prompt_toolkit import PromptSession as _PromptSession
    from prompt_toolkit.formatted_text import ANSI as _ANSI
    from prompt_toolkit.history import InMemoryHistory as _InMemoryHistory
except ImportError:  # pragma: no cover - fallback when prompt_toolkit missing
    PromptSession = None
    ANSI = None
    InMemoryHistory = None
else:
    PromptSession = _PromptSession
    ANSI = _ANSI
    InMemoryHistory = _InMemoryHistory

if TYPE_CHECKING:
    from src.config import AgentConfig
    from src.helpers import Topic

try:
    _config = importlib.import_module("src.config")
    _chains = importlib.import_module("src.chains")
    _exceptions = importlib.import_module("src.exceptions")
except ModuleNotFoundError as exc:  # fallback when imported as top-level module
    missing_root = getattr(exc, "name", "").split(".")[0]
    if missing_root != "src":
        raise
    _config = importlib.import_module("config")
    _chains = importlib.import_module("chains")
    _exceptions = importlib.import_module("exceptions")

_AgentConfig = _config.AgentConfig
build_llms = _chains.build_llms
build_chains = _chains.build_chains
ResponseError = _exceptions.ResponseError

# Import helpers as a module and bind required symbols to avoid duplicate
# name-definition errors during static analysis (mypy sees both branches
# if using `from ... import ...` in try/except).
try:
    _helpers = importlib.import_module("src.helpers")
except ModuleNotFoundError as exc:  # pragma: no cover - local dev fallback
    missing_root = getattr(exc, "name", "").split(".")[0]
    if missing_root != "src":
        raise
    _helpers = importlib.import_module("helpers")

_Topic = _helpers.Topic
_tail_turns = _helpers._tail_turns
_truncate_result = _helpers._truncate_result
_normalize_query = _helpers._normalize_query
_regex_validate = _helpers._regex_validate
_truncate_text = _helpers._truncate_text
_current_datetime_utc = _helpers._current_datetime_utc
_is_context_length_error = _helpers._is_context_length_error
_pick_seed_query = _helpers._pick_seed_query
_extract_keywords = _helpers._extract_keywords
_is_relevant = _helpers._is_relevant
_format_turns = _helpers._format_turns
_collect_prior_responses = _helpers._collect_prior_responses
_select_topic = _helpers._select_topic
_prune_keywords = _helpers._prune_keywords
_canonicalize_url = _helpers._canonicalize_url
_summarize_answer = _helpers._summarize_answer
_topic_brief = _helpers._topic_brief
_blend_embeddings = _helpers._blend_embeddings
_cosine_similarity = _helpers._cosine_similarity
_PATTERN_SEARCH_DECISION = _helpers._PATTERN_SEARCH_DECISION
_PATTERN_YES_NO = _helpers._PATTERN_YES_NO
MAX_CONVERSATION_CHARS = _helpers.MAX_CONVERSATION_CHARS
MAX_PRIOR_RESPONSE_CHARS = _helpers.MAX_PRIOR_RESPONSE_CHARS
MAX_SEARCH_RESULTS_CHARS = _helpers.MAX_SEARCH_RESULTS_CHARS
MAX_TURN_KEYWORD_SOURCE_CHARS = _helpers.MAX_TURN_KEYWORD_SOURCE_CHARS
MAX_TOPICS = _helpers.MAX_TOPICS
MAX_REBUILD_RETRIES = _helpers.MAX_REBUILD_RETRIES
MAX_PROMPT_RECENT_TURNS = _helpers.MAX_PROMPT_RECENT_TURNS


class Agent:
    def __init__(self, cfg: AgentConfig, *, output_stream: TextIO | None = None, is_tty: bool | None = None):
        self.cfg = cfg
        self.llm_robot, self.llm_assistant = build_llms(cfg)
        self.chains = build_chains(self.llm_robot, self.llm_assistant)
        # Maintain a DDGS client attribute for smoke/health checks; keep it closed to avoid lingering sockets.
        self.search_client: Any = self._new_closed_ddgs(cfg.search_timeout)
        self.rebuild_counts = {
            "search_decision": 0,
            "seed": 0,
            "relevance": 0,
            "planning": 0,
            "query_filter": 0,
            "answer": 0,
        }
        self.topics: List["Topic"] = []
        self._prompt_session = None
        self._embedding_client: OllamaEmbeddings | None = None
        self._embedding_warning_logged = False
        self._last_error: str | None = None
        self._out: TextIO = output_stream or sys.stdout
        self._is_tty: bool = bool(is_tty if is_tty is not None else getattr(self._out, "isatty", lambda: False)())
        self._base_llm_params = {
            "assistant_num_ctx": cfg.assistant_num_ctx,
            "robot_num_ctx": cfg.robot_num_ctx,
            "assistant_num_predict": cfg.assistant_num_predict,
            "robot_num_predict": cfg.robot_num_predict,
        }

    def _write(self, text: str) -> None:
        try:
            self._out.write(text)
            if hasattr(self._out, "flush"):
                self._out.flush()
        except Exception as exc:
            logging.debug("Output stream write failed: %s", exc)

    def _writeln(self, text: str = "") -> None:
        self._write(f"{text}\n")

    @staticmethod
    def _safe_close(client: Any) -> None:
        if client is None:
            return
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:
                logging.debug("Client close failed: %s", exc)

    @staticmethod
    def _new_closed_ddgs(timeout: float | None = None) -> Any:
        try:
            client = DDGS(timeout=cast(int | None, timeout))
            Agent._safe_close(client)
            return client
        except Exception:
            return None

    def _reset_rebuild_counts(self) -> None:
        for key in self.rebuild_counts:
            self.rebuild_counts[key] = 0

    def _char_budget(self, base: int) -> int:
        # Roughly map tokens→chars (~4 chars/token) and keep a safety margin.
        ctx_tokens = min(self.cfg.assistant_num_ctx, self.cfg.robot_num_ctx)
        return min(base, max(1024, int(ctx_tokens * 4 * 0.8)))

    def _mark_error(self, message: str) -> str:
        self._last_error = message
        return message

    def _clear_error(self) -> None:
        self._last_error = None

    def _close_clients(self) -> None:
        client = getattr(self, "search_client", None)
        self._safe_close(client)
        self.search_client = None

    def _print_welcome_banner(self) -> None:
        message = "\n".join(
            [
                "Welcome to Local AI Agent.",
                "Made by Pavol Ulicny.",
                "Enter submits your message. Type 'exit' to quit.",
            ]
        )
        if self._is_tty:
            self._writeln(f"\n\033[96m{message}\033[0m")
        else:
            self._writeln(message)

    # Dynamic config updates after rebuild
    def _rebuild_llms(self, new_ctx: int, new_predict: int) -> None:
        self.cfg.assistant_num_ctx = new_ctx
        self.cfg.robot_num_ctx = new_ctx
        self.cfg.assistant_num_predict = new_predict
        self.cfg.robot_num_predict = min(self.cfg.robot_num_predict, new_predict)
        self.llm_robot, self.llm_assistant = build_llms(self.cfg)
        self.chains = build_chains(self.llm_robot, self.llm_assistant)

    def _restore_llm_params(self) -> None:
        base = self._base_llm_params
        cfg = self.cfg
        needs_restore = any(
            [
                cfg.assistant_num_ctx != base["assistant_num_ctx"],
                cfg.robot_num_ctx != base["robot_num_ctx"],
                cfg.assistant_num_predict != base["assistant_num_predict"],
                cfg.robot_num_predict != base["robot_num_predict"],
            ]
        )
        if not needs_restore:
            return
        cfg.assistant_num_ctx = base["assistant_num_ctx"]
        cfg.robot_num_ctx = base["robot_num_ctx"]
        cfg.assistant_num_predict = base["assistant_num_predict"]
        cfg.robot_num_predict = base["robot_num_predict"]
        self.llm_robot, self.llm_assistant = build_llms(cfg)
        self.chains = build_chains(self.llm_robot, self.llm_assistant)

    def _reduce_context_and_rebuild(self, stage_key: str, label: str) -> None:
        self.rebuild_counts[stage_key] += 1
        current_ctx = min(self.cfg.assistant_num_ctx, self.cfg.robot_num_ctx)
        current_predict = self.cfg.assistant_num_predict
        reduced_ctx = max(2048, current_ctx // 2)
        reduced_predict = max(512, min(current_predict, reduced_ctx // 2))
        logging.info(
            "Context too large (%s); rebuild %s/%s with num_ctx=%s, num_predict=%s.",
            label,
            self.rebuild_counts[stage_key],
            MAX_REBUILD_RETRIES,
            reduced_ctx,
            reduced_predict,
        )
        self._rebuild_llms(reduced_ctx, reduced_predict)

    def _ddg_results(self, query: str) -> List[dict]:
        delay = 1.0
        for attempt in range(1, self.cfg.search_retries + 1):
            self.search_client = DDGS(timeout=cast(int | None, self.cfg.search_timeout))
            try:
                raw_results = self.search_client.text(
                    query,
                    region=self.cfg.ddg_region,
                    safesearch=self.cfg.ddg_safesearch,
                    backend=self.cfg.ddg_backend,
                    max_results=self.cfg.search_max_results,
                )
                results: List[dict] = []
                for entry in raw_results or []:
                    normalized = self._normalize_search_result(entry)
                    if normalized:
                        results.append(normalized)
                return results
            except TimeoutException as exc:  # pragma: no cover - network
                logging.warning(
                    f"Search timeout for '{query}' (attempt {attempt}/{self.cfg.search_retries}); sleeping {delay:.1f}s"
                )
                logging.debug("Timeout details: %s", exc)
            except DDGSException as exc:  # pragma: no cover - network
                logging.warning(f"DDGS search error for '{query}' (attempt {attempt}/{self.cfg.search_retries}): {exc}")
            except Exception as exc:  # pragma: no cover - network
                logging.warning(
                    f"Unexpected search error for '{query}' (attempt {attempt}/{self.cfg.search_retries}): {exc}"
                )
            finally:
                self._safe_close(self.search_client)
                # Keep an already-closed client around so health checks see the attribute without leaking descriptors.
                self.search_client = self._new_closed_ddgs(self.cfg.search_timeout)
            if attempt < self.cfg.search_retries:
                time.sleep(delay + (random.random() * 0.5))
                delay = min(delay * 2.0, 8.0)
        logging.warning(f"Search failed after {self.cfg.search_retries} attempts for '{query}'.")
        return []

    def _normalize_search_result(self, raw_result: dict[str, Any]) -> dict[str, str] | None:
        title = str(raw_result.get("title") or "").strip()
        snippet = str(
            raw_result.get("body")
            or raw_result.get("snippet")
            or raw_result.get("description")
            or raw_result.get("content")
            or ""
        ).strip()
        link = str(
            raw_result.get("href")
            or raw_result.get("url")
            or raw_result.get("image")
            or raw_result.get("embed_url")
            or ""
        ).strip()
        if not any([title, snippet, link]):
            return None
        return {"title": title, "link": link, "snippet": snippet}

    def _inputs(
        self,
        current_datetime: str,
        current_year: str,
        current_month: str,
        current_day: str,
        conversation_text: str,
        user_query: str,
        **overrides,
    ):
        base = {
            "current_datetime": current_datetime,
            "current_year": current_year,
            "current_month": current_month,
            "current_day": current_day,
            "conversation_history": conversation_text,
            "user_question": user_query,
        }
        base.update(overrides)
        return base

    def _prompt_messages(self) -> tuple[Any, str]:
        plain_prompt = "> "
        if ANSI is not None and self._is_tty:
            formatted = ANSI("\n\033[92m> \033[0m")
            return formatted, plain_prompt
        return plain_prompt, plain_prompt

    def _build_prompt_session(self):
        if PromptSession is None or InMemoryHistory is None:
            return None
        return PromptSession(
            history=InMemoryHistory(),
            multiline=False,
            wrap_lines=True,
        )

    def _ensure_prompt_session(self):
        if self._prompt_session is None:
            self._prompt_session = self._build_prompt_session()
        return self._prompt_session

    def _ensure_embedding_client(self) -> OllamaEmbeddings | None:
        if not self.cfg.embedding_model:
            return None
        if self._embedding_client is None:
            try:
                self._embedding_client = OllamaEmbeddings(model=self.cfg.embedding_model)
            except Exception as exc:
                if not self._embedding_warning_logged:
                    logging.warning(
                        "Embedding model '%s' unavailable; semantic similarity checks are disabled (%s)",
                        self.cfg.embedding_model,
                        exc,
                    )
                    self._embedding_warning_logged = True
                self._embedding_client = None
        return self._embedding_client

    def _embed_text(self, text: str) -> List[float] | None:
        normalized = (text or "").strip()
        if not normalized:
            return None
        client = self._ensure_embedding_client()
        if client is None:
            return None
        try:
            return list(client.embed_query(normalized))
        except Exception as exc:  # pragma: no cover - network/model specific
            if "not found" in str(exc).lower():
                logging.warning(
                    "Embedding model '%s' not found. Run 'ollama pull %s' to enable semantic topic tracking.",
                    self.cfg.embedding_model,
                    self.cfg.embedding_model,
                )
            else:
                logging.warning("Embedding generation failed (%s): %s", self.cfg.embedding_model, exc)
            return None

    @staticmethod
    def _context_similarity(
        candidate_embedding: List[float] | None,
        question_embedding: List[float] | None,
        topic_embedding: List[float] | None,
    ) -> float:
        if not candidate_embedding:
            return 0.0
        scores: List[float] = []
        if question_embedding:
            scores.append(_cosine_similarity(candidate_embedding, question_embedding))
        if topic_embedding:
            scores.append(_cosine_similarity(candidate_embedding, topic_embedding))
        return max(scores) if scores else 0.0

    def _read_user_query(self) -> str:
        formatted_prompt, _ = self._prompt_messages()
        session = self._ensure_prompt_session()
        if session is None:
            # Fallback to built-in input when prompt_toolkit is unavailable.
            return input(formatted_prompt)  # noqa: A001 - shadowing built-in acceptable for prompt
        return session.prompt(formatted_prompt)

    def answer_once(self, question: str) -> str | None:
        try:
            return self._handle_query(question, one_shot=True)
        finally:
            self._close_clients()

    def run(self) -> None:  # interactive loop
        self._print_welcome_banner()
        try:
            while True:
                try:
                    user_query = self._read_user_query().strip()
                except (KeyboardInterrupt, EOFError):
                    logging.info("Exiting due to interrupt/EOF.")
                    return
                if not user_query:
                    logging.info("No input provided.")
                    continue
                if user_query.lower() in {"exit", "quit"}:
                    logging.info("Goodbye!")
                    return
                self._handle_query(user_query, one_shot=False)
                if self._last_error:
                    self._writeln(self._last_error)
                    if not self.cfg.log_console:
                        logging.error(self._last_error)
                    self._clear_error()
        finally:
            self._close_clients()

    def _handle_query(self, user_query: str, one_shot: bool) -> str | None:
        try:
            return self._handle_query_core(user_query, one_shot)
        finally:
            self._restore_llm_params()

    def _handle_query_core(self, user_query: str, one_shot: bool) -> str | None:
        self._clear_error()
        self._reset_rebuild_counts()
        cfg = self.cfg
        current_datetime = _current_datetime_utc()
        dt_obj = datetime.now(timezone.utc)
        current_year = str(dt_obj.year)
        current_month = f"{dt_obj.month:02d}"
        current_day = f"{dt_obj.day:02d}"
        question_keywords = _extract_keywords(user_query)
        question_embedding = self._embed_text(user_query)
        try:
            selected_topic_index, recent_history, topic_keywords = _select_topic(
                self.chains["context"],
                self.topics,
                user_query,
                question_keywords,
                cfg.max_context_turns,
                current_datetime,
                current_year,
                current_month,
                current_day,
                question_embedding=question_embedding,
                embedding_threshold=cfg.embedding_similarity_threshold,
            )
        except ResponseError as exc:  # model not found, etc.
            message = str(exc)
            if "not found" in message.lower():
                logging.error(f"Model '{cfg.model}' not found. Run 'ollama pull {cfg.model}' and retry.")
                self._mark_error(f"Model '{cfg.model}' not found. Run 'ollama pull {cfg.model}' and retry.")
                return None
            selected_topic_index = None
            recent_history = []
            topic_keywords = set(question_keywords)
        except Exception as exc:  # graceful fallback
            logging.warning(f"Context classification failed; proceeding without topic selection. Error: {exc}")
            selected_topic_index = None
            recent_history = []
            topic_keywords = set(question_keywords)
        topic_brief_text = ""
        if selected_topic_index is not None and selected_topic_index < len(self.topics):
            topic_brief_text = _topic_brief(self.topics[selected_topic_index])
        topic_embedding_current: List[float] | None = None
        if selected_topic_index is not None and selected_topic_index < len(self.topics):
            topic_embedding_current = self.topics[selected_topic_index].embedding
        recent_for_prompt = _tail_turns(recent_history, MAX_PROMPT_RECENT_TURNS)
        conversation_text = _format_turns(recent_for_prompt, "No prior relevant conversation.")
        if topic_brief_text:
            conversation_text = f"Topic brief:\n{topic_brief_text}\n\nRecent turns:\n{conversation_text}"
        conversation_text = _truncate_text(conversation_text, self._char_budget(MAX_CONVERSATION_CHARS))
        prior_responses_text = (
            _collect_prior_responses(self.topics[selected_topic_index], max_chars=MAX_PRIOR_RESPONSE_CHARS)
            if selected_topic_index is not None
            else "No prior answers for this topic."
        )
        if topic_brief_text:
            prior_responses_text = f"{topic_brief_text}\n\nRecent answers:\n{prior_responses_text}"
        prior_responses_text = _truncate_text(prior_responses_text, self._char_budget(MAX_PRIOR_RESPONSE_CHARS))

        aggregated_results: List[str] = []
        seen_result_hashes: Set[str] = set()
        seen_urls: Set[str] = set()

        should_search = bool(cfg.force_search)
        if not should_search and cfg.auto_search_decision:
            try:
                decision_raw = self.chains["search_decision"].invoke(
                    self._inputs(
                        current_datetime,
                        current_year,
                        current_month,
                        current_day,
                        conversation_text,
                        user_query,
                        known_answers=prior_responses_text,
                    )
                )
                decision_validated = _regex_validate(str(decision_raw), _PATTERN_SEARCH_DECISION, "SEARCH")
                should_search = decision_validated == "SEARCH"
            except ResponseError as exc:
                if "not found" in str(exc).lower():
                    logging.error(f"Model '{cfg.model}' not found. Run 'ollama pull {cfg.model}' and retry.")
                    self._mark_error(f"Model '{cfg.model}' not found. Run 'ollama pull {cfg.model}' and retry.")
                    return None
                if _is_context_length_error(str(exc)):
                    if self.rebuild_counts["search_decision"] < MAX_REBUILD_RETRIES:
                        self._reduce_context_and_rebuild("search_decision", "search decision")
                        try:
                            decision_raw = self.chains["search_decision"].invoke(
                                self._inputs(
                                    current_datetime,
                                    current_year,
                                    current_month,
                                    current_day,
                                    conversation_text,
                                    user_query,
                                    known_answers=prior_responses_text,
                                )
                            )
                            decision_validated = _regex_validate(str(decision_raw), _PATTERN_SEARCH_DECISION, "SEARCH")
                            should_search = decision_validated == "SEARCH"
                        except ResponseError:
                            should_search = False
                    else:
                        logging.info("Reached search decision rebuild cap; defaulting to NO_SEARCH fallback.")
                        should_search = False
                else:
                    logging.error("Search decision failed: %s", exc)
                    self._mark_error("Search decision failed; please retry.")
                    return None
            except Exception as exc:
                logging.warning(f"Search decision failed unexpectedly; defaulting to NO_SEARCH. Error: {exc}")
                should_search = False
        elif not cfg.auto_search_decision and not cfg.force_search:
            should_search = False

        if should_search:
            primary_search_query = user_query
            try:
                seed_raw = self.chains["seed"].invoke(
                    self._inputs(
                        current_datetime,
                        current_year,
                        current_month,
                        current_day,
                        conversation_text,
                        user_query,
                        known_answers=prior_responses_text,
                    )
                )
                seed_text = str(seed_raw).strip()
                primary_search_query = _pick_seed_query(seed_text, user_query)
            except ResponseError as exc:
                if "not found" in str(exc).lower():
                    logging.error(f"Model '{cfg.model}' not found. Run 'ollama pull {cfg.model}' and retry.")
                    self._mark_error(f"Model '{cfg.model}' not found. Run 'ollama pull {cfg.model}' and retry.")
                    return None
                if _is_context_length_error(str(exc)):
                    if self.rebuild_counts["seed"] < MAX_REBUILD_RETRIES:
                        self._reduce_context_and_rebuild("seed", "seed")
                        try:
                            seed_raw = self.chains["seed"].invoke(
                                self._inputs(
                                    current_datetime,
                                    current_year,
                                    current_month,
                                    current_day,
                                    conversation_text,
                                    user_query,
                                    known_answers=prior_responses_text,
                                )
                            )
                            seed_text = str(seed_raw).strip()
                            primary_search_query = _pick_seed_query(seed_text, user_query)
                        except ResponseError:
                            primary_search_query = user_query
                    else:
                        logging.info("Reached seed rebuild cap; using original user query as search seed.")
                        primary_search_query = user_query
                else:
                    logging.error("Seed generation failed: %s", exc)
                    self._mark_error("Seed query generation failed; please retry.")
                    return None
            except Exception as exc:
                logging.warning(f"Seed generation failed unexpectedly; using original query. Error: {exc}")
                primary_search_query = user_query

            pending_queries: List[str] = [primary_search_query]
            seen_query_norms: Set[str] = {_normalize_query(primary_search_query)}
            if not topic_keywords:
                topic_keywords.update(_extract_keywords(user_query))
                topic_keywords.update(_extract_keywords(primary_search_query))
            max_rounds = cfg.max_rounds
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
                        result_embedding = self._embed_text(keywords_source)
                        similarity = self._context_similarity(
                            result_embedding,
                            question_embedding,
                            topic_embedding_current,
                        )
                        if similarity >= cfg.embedding_result_similarity_threshold:
                            relevant = True
                        if not relevant:
                            if relevance_llm_checks >= cfg.max_relevance_llm_checks:
                                continue
                            kw_list = sorted(topic_keywords) if topic_keywords else []
                            if len(kw_list) > 50:
                                kw_list = kw_list[:50]
                            topic_keywords_text = ", ".join(kw_list) if kw_list else "None"
                            topic_keywords_text = _truncate_text(topic_keywords_text, 1000)
                            try:
                                relevance_raw = self.chains["result_filter"].invoke(
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
                            except ResponseError as exc:
                                if "not found" in str(exc).lower():
                                    logging.error(
                                        f"Model '{cfg.model}' not found. Run 'ollama pull {cfg.model}' and retry."
                                    )
                                    self._mark_error(
                                        f"Model '{cfg.model}' not found. Run 'ollama pull {cfg.model}' and retry."
                                    )
                                    return None
                                if _is_context_length_error(str(exc)):
                                    if self.rebuild_counts["relevance"] < MAX_REBUILD_RETRIES:
                                        self._reduce_context_and_rebuild("relevance", "relevance")
                                        try:
                                            relevance_raw = self.chains["result_filter"].invoke(
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
                                            relevance_decision = _regex_validate(
                                                str(relevance_raw), _PATTERN_YES_NO, "NO"
                                            )
                                            relevance_llm_checks += 1
                                        except ResponseError:
                                            logging.info(
                                                f"Relevance retry failed; skipping one result for '{current_query}'."
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
                            except Exception as exc:
                                logging.info(
                                    f"Relevance check crashed; skipping one result for '{current_query}'. Error: {exc}"
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
                suggestion_limit = min(cfg.max_followup_suggestions, remaining_slots)
                results_to_date = "\n\n".join(aggregated_results) or "No results yet."
                results_to_date = _truncate_text(results_to_date, self._char_budget(MAX_SEARCH_RESULTS_CHARS))
                try:
                    suggestions_raw = self.chains["planning"].invoke(
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
                except ResponseError as exc:
                    if "not found" in str(exc).lower():
                        logging.error(f"Model '{cfg.model}' not found. Run 'ollama pull {cfg.model}' and retry.")
                        self._mark_error(f"Model '{cfg.model}' not found. Run 'ollama pull {cfg.model}' and retry.")
                        return None
                    if _is_context_length_error(str(exc)):
                        if self.rebuild_counts["planning"] < MAX_REBUILD_RETRIES:
                            self._reduce_context_and_rebuild("planning", "planning")
                            try:
                                suggestions_raw = self.chains["planning"].invoke(
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
                        return None
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
                    norm_candidate = _normalize_query(candidate)
                    if norm_candidate in seen_query_norms or len(pending_queries) >= max_rounds:
                        continue
                    candidate_embedding = self._embed_text(candidate)
                    if candidate_embedding is not None and (question_embedding or topic_embedding_current):
                        similarity = self._context_similarity(
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
                            continue
                    try:
                        verdict_raw = self.chains["query_filter"].invoke(
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
                    except ResponseError as exc:
                        if "not found" in str(exc).lower():
                            logging.error(f"Model '{cfg.model}' not found. Run 'ollama pull {cfg.model}' and retry.")
                            self._mark_error(f"Model '{cfg.model}' not found. Run 'ollama pull {cfg.model}' and retry.")
                            return None
                        if _is_context_length_error(str(exc)):
                            if self.rebuild_counts["query_filter"] < MAX_REBUILD_RETRIES:
                                self._reduce_context_and_rebuild("query_filter", "query filter")
                                try:
                                    verdict_raw = self.chains["query_filter"].invoke(
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
                fill_attempts = 0
                while len(pending_queries) < max_rounds and fill_attempts < cfg.max_fill_attempts:
                    fill_attempts += 1
                    remaining_slots = max_rounds - len(pending_queries)
                    if remaining_slots <= 0:
                        break
                    suggestion_limit = min(cfg.max_followup_suggestions, remaining_slots)
                    results_to_date = "\n\n".join(aggregated_results) or "No results yet."
                    results_to_date = _truncate_text(results_to_date, MAX_SEARCH_RESULTS_CHARS)
                    try:
                        suggestions_raw = self.chains["planning"].invoke(
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
                    except ResponseError as exc:
                        if "not found" in str(exc).lower():
                            logging.error(f"Model '{cfg.model}' not found. Run 'ollama pull {cfg.model}' and retry.")
                            self._mark_error(f"Model '{cfg.model}' not found. Run 'ollama pull {cfg.model}' and retry.")
                            return None
                        if _is_context_length_error(str(exc)):
                            if self.rebuild_counts["planning"] < MAX_REBUILD_RETRIES:
                                self._reduce_context_and_rebuild("planning", "planning")
                                try:
                                    suggestions_raw = self.chains["planning"].invoke(
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
                        norm_candidate = _normalize_query(candidate)
                        if norm_candidate in seen_query_norms or len(pending_queries) >= max_rounds:
                            continue
                        candidate_embedding = self._embed_text(candidate)
                        if candidate_embedding is not None and (question_embedding or topic_embedding_current):
                            similarity = self._context_similarity(
                                candidate_embedding,
                                question_embedding,
                                topic_embedding_current,
                            )
                            if similarity < cfg.embedding_query_similarity_threshold:
                                logging.info(
                                    "Skipping fill suggestion with low semantic similarity (%.2f): %s",
                                    similarity,
                                    candidate,
                                )
                                continue
                        try:
                            verdict_raw = self.chains["query_filter"].invoke(
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
                        except ResponseError as exc:
                            if "not found" in str(exc).lower():
                                logging.error(
                                    f"Model '{cfg.model}' not found. Run 'ollama pull {cfg.model}' and retry."
                                )
                                self._mark_error(
                                    f"Model '{cfg.model}' not found. Run 'ollama pull {cfg.model}' and retry."
                                )
                                return None
                            if _is_context_length_error(str(exc)):
                                if self.rebuild_counts["query_filter"] < MAX_REBUILD_RETRIES:
                                    self._reduce_context_and_rebuild("query_filter", "query filter")
                                    try:
                                        verdict_raw = self.chains["query_filter"].invoke(
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
                                        logging.info(f"Skipping fill suggestion after retry: {candidate}")
                                        continue
                                else:
                                    logging.info(
                                        f"Reached query filter rebuild cap; skipping fill candidate: {candidate}"
                                    )
                                    continue
                            else:
                                logging.info(f"Skipping fill suggestion due to filter error: {candidate} ({exc})")
                                continue
                        except Exception as exc:
                            logging.info(
                                f"Skipping fill suggestion due to unexpected filter error: {candidate} ({exc})"
                            )
                            continue
                        verdict = _regex_validate(str(verdict_raw), _PATTERN_YES_NO, "NO")
                        if verdict == "YES" and norm_candidate not in seen_query_norms:
                            pending_queries.append(candidate)
                            seen_query_norms.add(norm_candidate)
                        else:
                            continue
        search_results_text = (
            "\n\n".join(aggregated_results)
            if should_search and aggregated_results
            else ("No search results collected." if should_search else "No web search performed.")
        )
        search_results_text = _truncate_text(search_results_text, self._char_budget(MAX_SEARCH_RESULTS_CHARS))
        if should_search:
            resp_inputs = self._inputs(
                current_datetime,
                current_year,
                current_month,
                current_day,
                conversation_text,
                user_query,
                search_results=search_results_text,
                prior_responses=prior_responses_text,
            )
            chain = self.chains["response"]
        else:
            resp_inputs = self._inputs(
                current_datetime,
                current_year,
                current_month,
                current_day,
                conversation_text,
                user_query,
                prior_responses=prior_responses_text,
            )
            chain = self.chains["response_no_search"]
        try:
            response_stream = chain.stream(resp_inputs)
        except ResponseError as exc:
            if "not found" in str(exc).lower():
                logging.error(f"Model '{cfg.model}' not found. Run 'ollama pull {cfg.model}' and retry.")
                self._mark_error(f"Model '{cfg.model}' not found. Run 'ollama pull {cfg.model}' and retry.")
                return None
            if _is_context_length_error(str(exc)):
                if self.rebuild_counts["answer"] < MAX_REBUILD_RETRIES:
                    self._reduce_context_and_rebuild("answer", "answer")
                    try:
                        chain = self.chains["response"] if should_search else self.chains["response_no_search"]
                        response_stream = chain.stream(resp_inputs)
                    except ResponseError as exc2:
                        logging.error(f"Answer generation failed after retry: {exc2}")
                        self._mark_error("Answer generation failed after retry; see logs for details.")
                        return None
                else:
                    logging.error("Reached answer generation rebuild cap; please shorten your query or reset session.")
                    self._mark_error(
                        "Answer generation failed: exceeded rebuild attempts; "
                        "please shorten your query or reset session."
                    )
                    return None
            else:
                logging.error(f"Answer generation failed: {exc}")
                self._mark_error("Answer generation failed; see logs for details.")
                return None
        except Exception as exc:
            logging.error(f"Answer generation failed unexpectedly: {exc}")
            self._mark_error("Answer generation failed unexpectedly; see logs for details.")
            return None
        if self._is_tty:
            self._writeln("\n\033[91m[Answer]\033[0m")
        else:
            self._writeln("\n[Answer]")
        response_chunks: List[str] = []
        stream_error: Exception | None = None
        try:
            for chunk in response_stream:
                response_chunks.append(chunk)
                self._write(chunk)
        except KeyboardInterrupt:
            logging.info("Streaming interrupted by user.")
        except Exception as exc:
            stream_error = exc
            logging.error(f"Streaming error: {exc}")
        if response_chunks and not response_chunks[-1].endswith("\n"):
            self._writeln()
        if one_shot:
            self._writeln()
        if stream_error:
            self._mark_error("Answer streaming failed; please retry.")
            return None
        response_text = "".join(response_chunks)
        if selected_topic_index is None:
            initial_embedding = list(question_embedding) if question_embedding else None
            self.topics.append(_Topic(keywords=set(topic_keywords), embedding=initial_embedding))
            selected_topic_index = len(self.topics) - 1
        if selected_topic_index != len(self.topics) - 1:
            moved_topic = self.topics.pop(selected_topic_index)
            self.topics.append(moved_topic)
            selected_topic_index = len(self.topics) - 1
        while len(self.topics) > MAX_TOPICS:
            self.topics.pop(0)
            selected_topic_index = max(0, selected_topic_index - 1)
        topic_entry = self.topics[selected_topic_index]
        topic_entry.turns.append((user_query, response_text))
        history_window = max(0, cfg.max_context_turns) * 2
        if history_window == 0:
            topic_entry.turns = []
        elif len(topic_entry.turns) > history_window:
            topic_entry.turns = topic_entry.turns[-history_window:]
        aggregated_keyword_source = _truncate_text(
            " ".join(aggregated_results),
            self._char_budget(MAX_TURN_KEYWORD_SOURCE_CHARS),
        )
        turn_keywords = _extract_keywords(" ".join([user_query, response_text, aggregated_keyword_source]))
        if not turn_keywords:
            turn_keywords = set(question_keywords)
        topic_entry.keywords.update(turn_keywords)
        topic_entry.keywords.update(topic_keywords)
        new_summary = _summarize_answer(response_text)
        if new_summary:
            topic_entry.summary = new_summary
        turn_embedding = self._embed_text(f"User: {user_query}\nAssistant: {response_text}")
        if turn_embedding:
            topic_entry.embedding = _blend_embeddings(
                topic_entry.embedding,
                turn_embedding,
                cfg.embedding_history_decay,
            )
        _prune_keywords(topic_entry)
        return response_text


__all__ = ["Agent"]
