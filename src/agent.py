"""Agent orchestration and helper decompositions.

This module implements the top-level `Agent` class which orchestrates user
queries, context selection, optional web search orchestration, and final
response generation. The implementation has been refactored into small,
testable helpers to keep control flow clear and to improve unit test
coverage:

- `QueryContext`: a dataclass capturing computed context values (datetime,
    conversation summary, embeddings, selected topic, etc.).
- `_invoke_chain_safe`: centralized chain invocation with rebuild-on-context-
    length handling and consistent ResponseError propagation.
- `_decide_should_search`, `_generate_search_seed`: encapsulate LLM-driven
    classification and seed selection logic.
- `_run_search_rounds`: thin wrapper around `SearchOrchestrator.run`.
- `_generate_and_stream_response`: centralized response streaming with
    injectable `write_fn` for tests.
- `_build_resp_inputs`: builds the inputs for the final response chain and
    chooses between `response` and `response_no_search` chains.

Interactive prompt/session handling is delegated to `src.input_handler.InputHandler`.
The refactor preserves runtime behavior while enabling focused unit tests
for each extracted piece.
"""

from __future__ import annotations

from typing import Any, List, Set, TYPE_CHECKING, cast, Callable
import importlib
import logging
import sys
from typing import TextIO

PromptSession: Any | None
ANSI: Any | None
InMemoryHistory: Any | None
if TYPE_CHECKING:
    from prompt_toolkit import PromptSession as PromptSessionType
    from prompt_toolkit.history import InMemoryHistory as InMemoryHistoryType
    from src.search_orchestrator import SearchOrchestrator as SearchOrchestratorType
    from src.topic_utils import Topic as TopicType
    from src.context import QueryContext as QueryContextType
else:
    PromptSessionType = Any
    InMemoryHistoryType = Any
    SearchOrchestratorType = Any
    TopicType = Any
    QueryContextType = Any

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

try:
    _chains = importlib.import_module("src.chains")
    _exceptions = importlib.import_module("src.exceptions")
    _search_client_mod = importlib.import_module("src.search_client")
    _embedding_client_mod = importlib.import_module("src.embedding_client")
    _search_orchestrator_mod = importlib.import_module("src.search_orchestrator")
    _topic_manager_mod = importlib.import_module("src.topic_manager")
    _text_utils_mod = importlib.import_module("src.text_utils")
    _keywords_mod = importlib.import_module("src.keywords")
    _topic_utils_mod = importlib.import_module("src.topic_utils")
    _model_utils_mod = importlib.import_module("src.model_utils")
except ModuleNotFoundError as exc:  # fallback when imported as top-level module
    missing_root = getattr(exc, "name", "").split(".")[0]
    if missing_root != "src":
        raise
    _chains = importlib.import_module("chains")
    _exceptions = importlib.import_module("exceptions")
    _search_client_mod = importlib.import_module("search_client")
    _embedding_client_mod = importlib.import_module("embedding_client")
    _search_orchestrator_mod = importlib.import_module("search_orchestrator")
    _topic_manager_mod = importlib.import_module("topic_manager")
    _text_utils_mod = importlib.import_module("text_utils")
    _keywords_mod = importlib.import_module("keywords")
    _topic_utils_mod = importlib.import_module("topic_utils")
    _model_utils_mod = importlib.import_module("model_utils")

try:
    _context_mod = importlib.import_module("src.context")
except ModuleNotFoundError:
    _context_mod = importlib.import_module("context")
QueryContext = _context_mod.QueryContext
build_query_context = _context_mod.build_query_context

try:
    _agent_utils_mod = importlib.import_module("src.agent_utils")
except ModuleNotFoundError:
    _agent_utils_mod = importlib.import_module("agent_utils")
agent_utils = _agent_utils_mod


class Agent:
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
        try:
            _input_handler_mod = importlib.import_module("src.input_handler")
        except ModuleNotFoundError:
            _input_handler_mod = importlib.import_module("input_handler")
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
        embedding_client = getattr(self, "embedding_client", None)
        if embedding_client is not None:
            embedding_client.close()

    def _build_search_orchestrator(self) -> SearchOrchestratorType:
        return cast(
            SearchOrchestratorType,
            _search_orchestrator_mod.SearchOrchestrator(
                self.cfg,
                ddg_results=self._ddg_results,
                embedding_client=self.embedding_client,
                context_similarity=self._context_similarity,
                inputs_builder=self._inputs,
                reduce_context_and_rebuild=self._reduce_context_and_rebuild,
                rebuild_counts=self.rebuild_counts,
                char_budget=self._char_budget,
                mark_error=self._mark_error,
            ),
        )

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
        reduced_ctx = max(2048, current_ctx // 2)
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

    def _decide_should_search(self, ctx: "QueryContextType", user_query: str, prior_responses_text: str) -> bool:
        """Run the `search_decision` classifier and return True when it decides SEARCH.

        This helper performs the chain invoke and validation. It does not swallow
        `_exceptions.ResponseError` so the caller can preserve existing control-flow.
        """
        decision_raw = self._invoke_chain_safe(
            "search_decision",
            self._inputs(
                ctx.current_datetime,
                ctx.current_year,
                ctx.current_month,
                ctx.current_day,
                ctx.conversation_text,
                user_query,
                known_answers=prior_responses_text,
            ),
            rebuild_key="search_decision",
        )
        decision_validated = cast(
            str, _text_utils_mod.regex_validate(str(decision_raw), _text_utils_mod.PATTERN_SEARCH_DECISION, "SEARCH")
        )
        return decision_validated == "SEARCH"

    def _generate_search_seed(self, ctx: "QueryContextType", user_query: str, prior_responses_text: str) -> str:
        """Invoke the `seed` chain and pick a seed query. May raise ResponseError to be
        handled by the caller in the same way as the original code.
        """
        seed_raw = self._invoke_chain_safe(
            "seed",
            self._inputs(
                ctx.current_datetime,
                ctx.current_year,
                ctx.current_month,
                ctx.current_day,
                ctx.conversation_text,
                user_query,
                known_answers=prior_responses_text,
            ),
            rebuild_key="seed",
        )
        seed_text = str(seed_raw).strip()
        return cast(str, _text_utils_mod.pick_seed_query(seed_text, user_query))

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
        orchestrator: SearchOrchestratorType = self._build_search_orchestrator()
        aggregated_results, topic_keywords = orchestrator.run(
            chains=self.chains,
            should_search=should_search,
            user_query=user_query,
            current_datetime=ctx.current_datetime,
            current_year=ctx.current_year,
            current_month=ctx.current_month,
            current_day=ctx.current_day,
            conversation_text=ctx.conversation_text,
            prior_responses_text=ctx.prior_responses_text,
            question_embedding=question_embedding,
            topic_embedding_current=topic_embedding_current,
            topic_keywords=topic_keywords,
            primary_search_query=primary_search_query,
        )
        return aggregated_results, topic_keywords

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
        if write_fn is None:
            write_fn = self._write
        cfg = self.cfg
        chain = self.chains[chain_name]
        try:
            response_stream = chain.stream(resp_inputs)
        except _exceptions.ResponseError as exc:
            if "not found" in str(exc).lower():
                _model_utils_mod.handle_missing_model(self._mark_error, "Assistant", cfg.assistant_model)
                return None
            if _text_utils_mod.is_context_length_error(str(exc)):
                if self.rebuild_counts["answer"] < _text_utils_mod.MAX_REBUILD_RETRIES:
                    self._reduce_context_and_rebuild("answer", "answer")
                    try:
                        chain = (
                            self.chains["response"] if chain_name == "response" else self.chains["response_no_search"]
                        )
                        response_stream = chain.stream(resp_inputs)
                    except _exceptions.ResponseError as exc2:
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

        if ANSI is not None and self._is_tty:
            self._writeln("\n\033[91m[Answer]\033[0m")
        else:
            self._writeln("\n[Answer]")
        response_chunks: List[str] = []
        stream_error: Exception | None = None
        try:
            for chunk in response_stream:
                response_chunks.append(chunk)
                write_fn(chunk)
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
        return "".join(response_chunks)

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
        return cast(
            int | None,
            self.topic_manager.update_topics(
                topics=self.topics,
                selected_topic_index=selected_topic_index,
                topic_keywords=topic_keywords,
                question_keywords=question_keywords,
                aggregated_results=aggregated_results,
                user_query=user_query,
                response_text=response_text,
                question_embedding=question_embedding,
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

    def _prompt_messages(self) -> tuple[Any, str]:
        return cast(tuple[Any, str], self.input_handler.prompt_messages())

    def _build_prompt_session(self) -> PromptSessionType | None:
        return cast(PromptSessionType, self.input_handler.build_prompt_session())

    def _ensure_prompt_session(self) -> PromptSessionType | None:
        # Ensure the session is available; allow InputHandler to create it lazily.
        self._prompt_session = cast(PromptSessionType, self.input_handler.ensure_prompt_session(self._prompt_session))
        return self._prompt_session

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

        should_search = bool(cfg.force_search)
        if not should_search and cfg.auto_search_decision:
            try:
                should_search = self._decide_should_search(ctx, user_query, prior_responses_text)
            except _exceptions.ResponseError as exc:
                if "not found" in str(exc).lower():
                    _model_utils_mod.handle_missing_model(self._mark_error, "Robot", cfg.robot_model)
                    return None
                if _text_utils_mod.is_context_length_error(str(exc)):
                    if self.rebuild_counts["search_decision"] < _text_utils_mod.MAX_REBUILD_RETRIES:
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
                            decision_validated = _text_utils_mod.regex_validate(
                                str(decision_raw), _text_utils_mod.PATTERN_SEARCH_DECISION, "SEARCH"
                            )
                            should_search = decision_validated == "SEARCH"
                        except _exceptions.ResponseError:
                            should_search = False
                    else:
                        logging.info("Reached search decision rebuild cap; defaulting to NO_SEARCH fallback.")
                        should_search = False
                else:
                    logging.error("Search decision failed: %s", exc)
                    self._mark_error("Search decision failed; please retry.")
                    return None
            except Exception as exc:
                # Default to SEARCH on unexpected classifier errors to avoid suppressing needed lookups.
                logging.warning("Search decision crashed; defaulting to SEARCH. Error: %s", exc)
                should_search = True
        elif not cfg.auto_search_decision and not cfg.force_search:
            should_search = False

        if should_search:
            primary_search_query = user_query
            try:
                primary_search_query = self._generate_search_seed(ctx, user_query, prior_responses_text)
            except _exceptions.ResponseError as exc:
                if "not found" in str(exc).lower():
                    _model_utils_mod.handle_missing_model(self._mark_error, "Robot", cfg.robot_model)
                    return None
                if _text_utils_mod.is_context_length_error(str(exc)):
                    if self.rebuild_counts["seed"] < _text_utils_mod.MAX_REBUILD_RETRIES:
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
                            primary_search_query = _text_utils_mod.pick_seed_query(seed_text, user_query)
                        except _exceptions.ResponseError:
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
            try:
                aggregated_results, topic_keywords = self._run_search_rounds(
                    ctx,
                    user_query,
                    should_search,
                    primary_search_query,
                    question_embedding,
                    topic_embedding_current,
                    topic_keywords,
                )
            except _search_orchestrator_mod.SearchAbort:
                return None
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
        # Generate and stream the response; helper returns final text or None on failure
        response_text = self._generate_and_stream_response(resp_inputs, chain_name, one_shot)
        if response_text is None:
            return None
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
