from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any, Callable, List, Set, TextIO, cast

from . import agent_utils as _agent_utils_mod
from . import chains as _chains
from . import context as _context_mod
from . import embedding_client as _embedding_client_mod
from . import exceptions as _exceptions
from . import input_handler as _input_handler_mod
from . import model_utils as _model_utils_mod
from . import response as _response_mod
from . import search as _search_mod
from . import search_client as _search_client_mod
from . import search_orchestrator as _search_orchestrator_mod
from . import text_utils as _text_utils_mod
from . import topic_manager as _topic_manager_mod
from . import topic_utils as _topic_utils_mod
from . import topics as _topics_mod

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession as PromptSessionType

    from .config import AgentConfig
    from .context import QueryContext as QueryContextType
    from .search_orchestrator import SearchOrchestrator as SearchOrchestratorType
    from .topic_utils import Topic as TopicType
else:
    PromptSessionType = Any
    SearchOrchestratorType = Any
    TopicType = Any
    QueryContextType = Any

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

QueryContext = _context_mod.QueryContext
build_query_context = _context_mod.build_query_context
agent_utils = _agent_utils_mod
search = _search_mod
response = _response_mod
topics = _topics_mod

# Character budget calculation constants
MIN_CHAR_BUDGET = 1024  # Minimum character budget regardless of context size
CHARS_PER_TOKEN_ESTIMATE = 4  # Rough estimate of characters per LLM token
CONTEXT_SAFETY_MARGIN = 0.8  # Use 80% of context to leave safety margin


class Agent:
    """Main agent orchestrator for conversational AI with web search capabilities.

    Coordinates LLM interactions, topic management, search operations, and response
    generation. Manages conversation context, handles user queries, and maintains
    topic-based conversation history with semantic similarity matching.
    """

    def __init__(self, cfg: AgentConfig, *, output_stream: TextIO | None = None, is_tty: bool | None = None):
        self.cfg = cfg
        self.llm_robot, self.llm_assistant = _chains.build_llms(cfg)
        self.chains = _chains.build_chains(self.llm_robot, self.llm_assistant)
        self.search_client = _search_client_mod.SearchClient(
            cfg,
            normalizer=self._normalize_search_result,
            notify_retry=self._notify_search_retry,
        )
        self.embedding_client = _embedding_client_mod.EmbeddingClient(cfg.embedding_model)
        self.topic_manager = _topic_manager_mod.TopicManager(
            cfg,
            embedding_client=self.embedding_client,
            char_budget=self._char_budget,
        )
        self.rebuild_counts = {
            "question_expansion": 0,
            "search_decision": 0,
            "seed": 0,
            "relevance": 0,
            "planning": 0,
            "query_filter": 0,
            "answer": 0,
        }
        self.topics: List[TopicType] = []
        self._prompt_session: PromptSessionType | None = None
        self._last_error: str | None = None
        self._out: TextIO = output_stream or sys.stdout
        self._is_tty: bool = bool(is_tty if is_tty is not None else getattr(self._out, "isatty", lambda: False)())
        self._base_llm_params = {
            "assistant_num_ctx": cfg.assistant_num_ctx,
            "robot_num_ctx": cfg.robot_num_ctx,
            "assistant_num_predict": cfg.assistant_num_predict,
            "robot_num_predict": cfg.robot_num_predict,
        }
        # Initialize input handler for prompt/session related helpers.
        # Provide a handler instance; prompt session may be created lazily.
        # Use direct attribute access rather than getattr with constant names.
        self.input_handler = _input_handler_mod.InputHandler(self._is_tty, None)
        self.build_inputs = _input_handler_mod.build_inputs

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

    def _notify_search_retry(self, attempt: int, total_attempts: int, delay: float, reason: Exception) -> None:
        message = f"[search retry {attempt}/{total_attempts}] waiting {delay:.1f}s before retry ({reason})"
        logging.info(message)
        if self._is_tty:
            self._writeln(message)

    def _reset_rebuild_counts(self) -> None:
        for key in self.rebuild_counts:
            self.rebuild_counts[key] = 0

    def _char_budget(self, base: int) -> int:
        # Roughly map tokens→chars and keep a safety margin.
        ctx_tokens = min(self.cfg.assistant_num_ctx, self.cfg.robot_num_ctx)
        estimated_chars = int(ctx_tokens * CHARS_PER_TOKEN_ESTIMATE * CONTEXT_SAFETY_MARGIN)
        return min(base, max(MIN_CHAR_BUDGET, estimated_chars))

    def _mark_error(self, message: str) -> str:
        self._last_error = message
        return message

    def _clear_error(self) -> None:
        self._last_error = None

    def _close_clients(self) -> None:
        client = getattr(self, "search_client", None)
        self._safe_close(client)
        self.search_client = None  # type: ignore[assignment]
        embedding_client = getattr(self, "embedding_client", None)
        if embedding_client is not None:
            embedding_client.close()

    def _build_search_orchestrator(self) -> SearchOrchestratorType:
        # Delegate to `src.search.build_search_orchestrator` to keep orchestration
        # code outside the `Agent` class for easier testing and separation.
        return cast(SearchOrchestratorType, search.build_search_orchestrator(self))

    def _print_welcome_banner(self) -> None:
        message = "\n".join(
            [
                "Welcome to Local AI Agent.",
                "Made by Pavol Ulicny.",
                "Enter submits your message. Type 'exit' to quit.",
            ]
        )
        if ANSI is not None and self._is_tty:
            self._writeln(f"\n\033[96m{message}\033[0m")
        else:
            self._writeln(message)

    # Dynamic config updates after rebuild
    def _rebuild_llms(self, new_ctx: int, new_predict: int) -> None:
        self.cfg.assistant_num_ctx = new_ctx
        self.cfg.robot_num_ctx = new_ctx
        self.cfg.assistant_num_predict = new_predict
        self.cfg.robot_num_predict = min(self.cfg.robot_num_predict, new_predict)
        self.llm_robot, self.llm_assistant = _chains.build_llms(self.cfg)
        self.chains = _chains.build_chains(self.llm_robot, self.llm_assistant)

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
        self.llm_robot, self.llm_assistant = _chains.build_llms(cfg)
        self.chains = _chains.build_chains(self.llm_robot, self.llm_assistant)

    def _reduce_context_and_rebuild(self, stage_key: str, label: str) -> None:
        self.rebuild_counts[stage_key] += 1
        current_ctx = min(self.cfg.assistant_num_ctx, self.cfg.robot_num_ctx)
        current_predict = self.cfg.assistant_num_predict
        # Compute a reduced context that does not increase the current context.
        # Use a conservative lower bound (1024 chars-worth tokens) and halve the
        # context, but never grow it: reduced_ctx = min(current_ctx, max(1024, current_ctx // 2)).
        reduced_ctx_candidate = max(1024, current_ctx // 2)
        reduced_ctx = min(current_ctx, reduced_ctx_candidate)
        reduced_predict = max(512, min(current_predict, reduced_ctx // 2))
        logging.info(
            "Context too large (%s); rebuild %s/%s with num_ctx=%s, num_predict=%s.",
            label,
            self.rebuild_counts[stage_key],
            _text_utils_mod.MAX_REBUILD_RETRIES,
            reduced_ctx,
            reduced_predict,
        )
        self._rebuild_llms(reduced_ctx, reduced_predict)

    def _invoke_chain_safe(self, chain_name: str, inputs: dict[str, Any], rebuild_key: str | None = None) -> Any:
        """Delegate to `src.agent_utils.invoke_chain_safe` for centralized handling."""
        return agent_utils.invoke_chain_safe(self, chain_name, inputs, rebuild_key)

    def _expand_question(self, ctx: "QueryContextType", user_query: str, prior_responses_text: str) -> str:
        """Expand user question to resolve pronouns and references from conversation history.

        Returns the expanded question, or the original if expansion fails.
        """
        inputs = {
            "user_question": user_query,
            "conversation_history": ctx.conversation_text,
            "known_answers": prior_responses_text,
            "current_year": ctx.current_year,
            "current_month": ctx.current_month,
            "current_day": ctx.current_day,
        }

        try:
            expanded = agent_utils.invoke_chain_safe(
                self,
                "question_expansion",
                inputs,
                rebuild_key="question_expansion",
                fallback_on_context_error=user_query,
                fallback_on_generic_error=user_query,
            )

            if expanded:
                expanded_clean = str(expanded).strip()
                if expanded_clean:
                    logging.debug(f"Question expanded: '{user_query}' → '{expanded_clean}'")
                    return expanded_clean

            logging.debug(f"Question expansion failed or empty, using original: '{user_query}'")
            return user_query

        except Exception as exc:
            logging.warning(f"Question expansion error: {exc}, using original question")
            return user_query

    def _decide_should_search(self, ctx: "QueryContextType", user_query: str, prior_responses_text: str) -> bool:
        """Run the search_decision classifier and return True if SEARCH decided.

        Delegates to `agent_utils.decide_should_search` for implementation.
        """
        return cast(bool, agent_utils.decide_should_search(self, ctx, user_query, prior_responses_text))

    def _generate_search_seed(self, ctx: "QueryContextType", user_query: str, prior_responses_text: str) -> str:
        """Generate a refined search query via the seed chain.

        Delegates to `agent_utils.generate_search_seed` for implementation.
        """
        return cast(str, agent_utils.generate_search_seed(self, ctx, user_query, prior_responses_text))

    def _run_search_rounds(
        self,
        ctx: "QueryContextType",
        user_query: str,
        should_search: bool,
        primary_search_query: str,
        question_embedding: List[float] | None,
        topic_embedding_current: List[float] | None,
        topic_keywords: Set[str],
    ) -> tuple[List[str], Set[str]]:
        """Run search orchestration via SearchOrchestrator and return results + keywords.

        This is a thin wrapper around `SearchOrchestrator.run` so the orchestration
        callsite in `_handle_query_core` is easier to test and reason about.
        It may raise `_search_orchestrator_mod.SearchAbort` which callers should handle.
        """
        # Delegate search orchestration to the `src.search` module.
        return cast(
            tuple[List[str], Set[str]],
            search.run_search_rounds(
                self,
                ctx,
                user_query,
                should_search,
                primary_search_query,
                question_embedding,
                topic_embedding_current,
                topic_keywords,
            ),
        )

    def _build_resp_inputs(
        self,
        current_datetime: str,
        current_year: str,
        current_month: str,
        current_day: str,
        conversation_text: str,
        user_query: str,
        should_search: bool,
        prior_responses_text: str,
        search_results_text: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Delegate to `src.agent_utils.build_resp_inputs`."""
        return cast(
            tuple[dict[str, Any], str],
            agent_utils.build_resp_inputs(
                self,
                current_datetime,
                current_year,
                current_month,
                current_day,
                conversation_text,
                user_query,
                should_search,
                prior_responses_text,
                search_results_text,
            ),
        )

    def _generate_and_stream_response(
        self,
        resp_inputs: dict[str, Any],
        chain_name: str,
        one_shot: bool,
        write_fn: Callable[[str], None] | None = None,
    ) -> str | None:
        """Invoke `chain.stream` for `chain_name`, stream output via `write_fn` and
        return the aggregated response text. Returns None on fatal errors (matching
        previous behavior) so callers can propagate `None`.
        """
        try:
            return cast(
                str | None, response.generate_and_stream_response(self, resp_inputs, chain_name, one_shot, write_fn)
            )
        except Exception as exc:
            logging.error("Response generation delegation failed: %s", exc)
            self._mark_error("Answer generation failed unexpectedly; see logs for details.")
            return None

    def _update_topics(
        self,
        selected_topic_index: int | None,
        topic_keywords: Set[str],
        question_keywords: List[str],
        aggregated_results: List[str],
        user_query: str,
        response_text: str,
        question_embedding: List[float] | None,
    ) -> int | None:
        """Wrap `topic_manager.update_topics` for easier testing and isolation.

        Returns the selected topic index (or None) as the underlying method does.
        """
        # Delegate to `src.topics.update_topics` to keep topic-related logic
        # isolated and easier to unit test.
        return cast(
            int | None,
            topics.update_topics(
                self,
                selected_topic_index,
                topic_keywords,
                question_keywords,
                aggregated_results,
                user_query,
                response_text,
                question_embedding,
            ),
        )

    def _ddg_results(self, query: str) -> Any:
        if self.search_client is None:
            self.search_client = _search_client_mod.SearchClient(
                self.cfg,
                normalizer=self._normalize_search_result,
                notify_retry=self._notify_search_retry,
            )
        return self.search_client.fetch(query)

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
        **overrides: Any,
    ) -> dict[str, Any]:
        # delegate to agent_utils.inputs to centralize the inputs builder
        return cast(
            dict[str, Any],
            agent_utils.inputs(
                self,
                current_datetime,
                current_year,
                current_month,
                current_day,
                conversation_text,
                user_query,
                **overrides,
            ),
        )

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
            scores.append(_topic_utils_mod.cosine_similarity(candidate_embedding, question_embedding))
        if topic_embedding:
            scores.append(_topic_utils_mod.cosine_similarity(candidate_embedding, topic_embedding))
        return max(scores) if scores else 0.0

    def _read_user_query(self) -> str:
        return cast(str, self.input_handler.read_user_query(self._prompt_session))

    def answer_once(self, question: str) -> str | None:
        """Process a single question and return the response.

        Args:
            question: User's question to answer.

        Returns:
            The generated response, or None if processing failed.
        """
        try:
            return self._handle_query(question, one_shot=True)
        finally:
            self._close_clients()

    def run(self) -> None:
        """Start interactive conversation loop with the user.

        Reads user input, processes queries, manages conversation context,
        and handles graceful exit on quit/interrupt signals.
        """
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
        # Phase 1: Context & State Initialization
        self._clear_error()
        self._reset_rebuild_counts()
        ctx = build_query_context(self, user_query)
        cfg = self.cfg
        current_datetime = ctx.current_datetime
        current_year = ctx.current_year
        current_month = ctx.current_month
        current_day = ctx.current_day
        question_keywords = ctx.question_keywords
        question_embedding = ctx.question_embedding
        selected_topic_index = ctx.selected_topic_index
        topic_keywords = ctx.topic_keywords
        topic_embedding_current = ctx.topic_embedding_current
        conversation_text = ctx.conversation_text
        prior_responses_text = ctx.prior_responses_text

        aggregated_results: List[str] = []

        # Phase 1.5: Question Expansion
        # Expand the question to resolve pronouns/references from conversation
        # Keep original user_query for final response, use expanded_question for search/logic
        expanded_question = self._expand_question(ctx, user_query, prior_responses_text)

        # Phase 2: Search Decision
        should_search = bool(cfg.force_search)
        if not should_search and cfg.auto_search_decision:
            try:
                should_search = self._decide_should_search(ctx, expanded_question, prior_responses_text)
            except _exceptions.ResponseError as exc:
                if "not found" in str(exc).lower():
                    _model_utils_mod.handle_missing_model(self._mark_error, "Robot", cfg.robot_model)
                    return None
                # Context length and other errors already handled by agent_utils.decide_should_search
                logging.error("Search decision failed: %s", exc)
                self._mark_error("Search decision failed; please retry.")
                return None
        elif not cfg.auto_search_decision and not cfg.force_search:
            should_search = False

        # Phase 3: Search Seed Generation & Search Orchestration
        if should_search:
            try:
                primary_search_query = self._generate_search_seed(ctx, expanded_question, prior_responses_text)
            except _exceptions.ResponseError as exc:
                if "not found" in str(exc).lower():
                    _model_utils_mod.handle_missing_model(self._mark_error, "Robot", cfg.robot_model)
                    return None
                # Context length and other errors already handled by agent_utils.generate_search_seed
                logging.error("Seed generation failed: %s", exc)
                self._mark_error("Seed query generation failed; please retry.")
                return None

            # Phase 4: Search Orchestration
            try:
                aggregated_results, topic_keywords = self._run_search_rounds(
                    ctx,
                    expanded_question,
                    should_search,
                    primary_search_query,
                    question_embedding,
                    topic_embedding_current,
                    topic_keywords,
                )
            except _search_orchestrator_mod.SearchAbort:
                return None

        # Phase 5: Response Generation
        search_results_text = (
            "\n\n".join(aggregated_results)
            if should_search and aggregated_results
            else ("No search results collected." if should_search else "No web search performed.")
        )
        search_results_text = _text_utils_mod.truncate_text(
            search_results_text, self._char_budget(_text_utils_mod.MAX_SEARCH_RESULTS_CHARS)
        )
        resp_inputs, chain_name = self._build_resp_inputs(
            current_datetime,
            current_year,
            current_month,
            current_day,
            conversation_text,
            user_query,
            should_search,
            prior_responses_text,
            search_results_text if should_search else None,
        )
        response_text = self._generate_and_stream_response(resp_inputs, chain_name, one_shot)
        if response_text is None:
            return None

        # Phase 6: Topic Update
        selected_topic_index = self._update_topics(
            selected_topic_index,
            topic_keywords,
            question_keywords,
            aggregated_results,
            user_query,
            response_text,
            question_embedding,
        )
        return response_text


__all__ = ["Agent"]
