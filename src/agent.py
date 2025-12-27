from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, List, TextIO, cast

from . import agent_utils as _agent_utils_mod
from . import chains as _chains  # noqa: F401 - used by tests for monkeypatching
from . import input_validation as _input_validation_mod
from .constants import (
    ChainName,
    RebuildKey,
    MIN_CHAR_BUDGET,
    CHARS_PER_TOKEN_ESTIMATE,
    CONTEXT_SAFETY_MARGIN,
    MAX_QUERY_LENGTH,
    MAX_SEARCH_RESULTS_CHARS,
)
from . import commands as _commands_mod
from . import conversation as _conversation_mod
from . import embedding_client as _embedding_client_mod
from . import exceptions as _exceptions
from . import input_handler as _input_handler_mod
from . import llm_lifecycle as _llm_lifecycle_mod
from . import model_utils as _model_utils_mod
from . import response as _response_mod
from . import search as _search_mod
from . import search_client as _search_client_mod
from . import search_orchestrator as _search_orchestrator_mod
from . import text_utils as _text_utils_mod

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession as PromptSessionType

    from .config import AgentConfig
    from .search_orchestrator import SearchOrchestrator as SearchOrchestratorType
else:
    PromptSessionType = Any
    SearchOrchestratorType = Any

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

agent_utils = _agent_utils_mod
search = _search_mod
response = _response_mod


class Agent:
    """Main agent orchestrator for conversational AI with web search capabilities.

    Coordinates LLM interactions, conversation management, search operations, and response
    generation. Uses a simple conversation list that leverages 128k context windows of
    modern LLMs instead of complex topic tracking.
    """

    def __init__(self, cfg: AgentConfig, *, output_stream: TextIO | None = None, is_tty: bool | None = None):
        self.cfg = cfg

        # Use LLMManager for lifecycle management
        self._llm_manager = _llm_lifecycle_mod.LLMManager(cfg)
        self.llm_robot, self.llm_assistant = self._llm_manager.get_llms()
        self.chains = self._llm_manager.get_chains()

        self.search_client = _search_client_mod.SearchClient(
            cfg,
            normalizer=self._normalize_search_result,
            notify_retry=self._notify_search_retry,
        )
        # Keep embedding client for search result filtering only
        self.embedding_client = _embedding_client_mod.EmbeddingClient(cfg.embedding_model)

        # New conversation management
        self.conversation = _conversation_mod.ConversationManager(max_context_chars=cfg.max_conversation_chars)

        # Session tracking
        self._session_start_time = datetime.now(timezone.utc)
        self._session_search_count = 0

        self.command_handler = _commands_mod.CommandHandler(self.conversation, self)

        self.rebuild_counts: dict[str, int] = {
            str(RebuildKey.SEARCH_DECISION): 0,
            str(RebuildKey.RELEVANCE): 0,
            str(RebuildKey.PLANNING): 0,
            str(RebuildKey.QUERY_FILTER): 0,
            str(RebuildKey.QUERY_REWRITE): 0,
            str(RebuildKey.ANSWER): 0,
        }
        self._prompt_session: PromptSessionType | None = None
        self._last_error: str | None = None
        self._out: TextIO = output_stream or sys.stdout
        self._is_tty: bool = bool(is_tty if is_tty is not None else getattr(self._out, "isatty", lambda: False)())

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
        except OSError as exc:
            # Output stream errors (IOError, BrokenPipeError are OSError subclasses in Python 3)
            logging.error("Output stream write failed: %s", exc)

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
                "Enter submits your message. Type '/quit' to exit or '/help' for commands.",
            ]
        )
        if ANSI is not None and self._is_tty:
            self._writeln(f"\n\033[96m{message}\033[0m")
        else:
            self._writeln(message)

    # Dynamic config updates after rebuild
    def _reduce_context_and_rebuild(self, stage_key: str, label: str) -> None:
        """Reduce context and rebuild LLMs - delegates to LLMManager."""
        self._llm_manager.rebuild_with_reduced_context(stage_key, label, self.rebuild_counts)
        # Update references to rebuilt LLMs and chains
        self.llm_robot, self.llm_assistant = self._llm_manager.get_llms()
        self.chains = self._llm_manager.get_chains()

    def _restore_llm_params(self) -> None:
        """Restore LLMs to original parameters - delegates to LLMManager."""
        self._llm_manager.restore_original_params()
        # Update references after restoration
        self.llm_robot, self.llm_assistant = self._llm_manager.get_llms()
        self.chains = self._llm_manager.get_chains()

    def _invoke_chain_safe(self, chain_name: str, inputs: dict[str, Any], rebuild_key: str | None = None) -> Any:
        """Delegate to `src.agent_utils.invoke_chain_safe` for centralized handling."""
        return agent_utils.invoke_chain_safe(self, chain_name, inputs, rebuild_key)

    def _sanitize_user_input(self, user_query: str) -> str:
        """Sanitize user input (delegates to input_validation module).

        Args:
            user_query: Raw user query

        Returns:
            Sanitized query string

        Raises:
            InputValidationError: If input contains suspicious injection patterns
        """
        return _input_validation_mod.sanitize_user_input(user_query)

    def _rewrite_user_query(self, sanitized_query: str, write_fn: Callable[[str], None] | None = None) -> str:
        """Rewrite user query for clarity and better search/response quality.

        Resolves pronouns, expands acronyms, adds temporal context.
        Falls back to original query on any errors.

        Args:
            sanitized_query: Sanitized user query
            write_fn: Optional function to display rewritten query to user

        Returns:
            Rewritten query string (or original if rewriting fails)
        """
        # Skip rewriting for very short queries (greetings, acknowledgments)
        if len(sanitized_query.strip()) < 5:
            return sanitized_query

        # Skip for pure command-like queries
        lower_query = sanitized_query.lower().strip()
        skip_patterns = ["hi", "hello", "thanks", "thank you", "ok", "okay", "bye"]
        if lower_query in skip_patterns:
            return sanitized_query

        try:
            # Build inputs for rewrite chain
            from datetime import datetime, timezone

            dt_obj = datetime.now(timezone.utc)
            current_year = str(dt_obj.year)
            current_month = f"{dt_obj.month:02d}"
            current_day = f"{dt_obj.day:02d}"

            conversation_history = self.conversation.format_for_prompt()
            max_conv_chars = self._char_budget(self.cfg.max_conversation_chars)
            if len(conversation_history) > max_conv_chars:
                conversation_history = "...\n\n" + conversation_history[-max_conv_chars:]

            rewrite_inputs = {
                "current_year": current_year,
                "current_month": current_month,
                "current_day": current_day,
                "conversation_history": conversation_history,
                "user_question": sanitized_query,
            }

            # Invoke rewrite chain with context overflow handling
            rewritten = self._invoke_chain_safe(
                ChainName.QUERY_REWRITE,
                rewrite_inputs,
                str(RebuildKey.QUERY_REWRITE),
            )

            if rewritten is None:
                logging.warning("Query rewrite returned None, using original query")
                return sanitized_query

            rewritten_str = str(rewritten).strip()

            # Validate rewritten query
            if not rewritten_str or len(rewritten_str) > MAX_QUERY_LENGTH:
                logging.warning("Query rewrite invalid (empty or too long), using original")
                return sanitized_query

            # If rewritten query is substantially different, show it to user
            if rewritten_str.lower() != sanitized_query.lower():
                logging.info("Query rewritten: '%s' -> '%s'", sanitized_query, rewritten_str)
                if write_fn and self._is_tty:
                    write_fn(f"→ Understood as: {rewritten_str}\n")

            return rewritten_str

        except Exception as exc:
            logging.warning("Query rewrite failed: %s, using original query", exc)
            return sanitized_query

    def _build_query_inputs(self, user_query: str) -> dict[str, Any]:
        """Build input dictionary for LLM chains - simplified without topic system.

        Args:
            user_query: Current user query

        Returns:
            Dictionary of inputs for prompt templates
        """
        from datetime import datetime, timezone

        # Get current datetime info
        current_datetime = _text_utils_mod.current_datetime_utc()
        dt_obj = datetime.now(timezone.utc)
        current_year = str(dt_obj.year)
        current_month = f"{dt_obj.month:02d}"
        current_day = f"{dt_obj.day:02d}"

        # Format conversation history (simple!)
        conversation_history = self.conversation.format_for_prompt()

        # Truncate if needed to fit in budget
        max_conv_chars = self._char_budget(self.cfg.max_conversation_chars)
        if len(conversation_history) > max_conv_chars:
            # Truncate from the beginning (keep recent context)
            conversation_history = "...\n\n" + conversation_history[-max_conv_chars:]

        # Build inputs dict
        return {
            "current_datetime": current_datetime,
            "current_year": current_year,
            "current_month": current_month,
            "current_day": current_day,
            "conversation_history": conversation_history,
            "user_question": user_query,
            # Provide conversation history under multiple keys for different prompt templates
            "known_answers": conversation_history,  # Used by some prompts
            "prior_responses": conversation_history,  # Used by response prompts
        }

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
        """Calculate cosine similarity between candidate and context embeddings.

        Used by search result filtering to determine semantic relevance.
        Returns max similarity if multiple context embeddings provided.
        """
        if not candidate_embedding:
            return 0.0

        def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
            """Calculate cosine similarity between two vectors."""
            if not vec_a or not vec_b or len(vec_a) != len(vec_b):
                return 0.0
            dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
            magnitude_a = sum(a * a for a in vec_a) ** 0.5
            magnitude_b = sum(b * b for b in vec_b) ** 0.5
            if magnitude_a == 0.0 or magnitude_b == 0.0:
                return 0.0
            return float(dot_product / (magnitude_a * magnitude_b))

        scores: List[float] = []
        if question_embedding:
            scores.append(cosine_similarity(candidate_embedding, question_embedding))
        if topic_embedding:
            scores.append(cosine_similarity(candidate_embedding, topic_embedding))
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
                    self._writeln("\nInterrupted. Type '/quit' to exit.")
                    continue
                if not user_query:
                    continue

                # Check for commands first
                is_command, response_msg, should_exit = self.command_handler.handle(user_query)
                if is_command:
                    if response_msg:
                        self._writeln(response_msg)
                    if should_exit:
                        return
                    continue

                # Handle exit/quit keywords
                if user_query.lower() in {"exit", "quit"}:
                    self._writeln("Goodbye! (Tip: use '/quit' or '/exit' for cleaner exit)")
                    return

                # Process query
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

    def _determine_search_necessity(self, query_inputs: dict[str, Any]) -> bool | None:
        """Determine whether search is needed for this query.

        Args:
            query_inputs: Prompt inputs for LLM

        Returns:
            True if search is needed, False if not, None on error

        Side Effects:
            - May call _mark_error if search decision fails
        """
        cfg = self.cfg

        # Check force_search flag
        if cfg.force_search:
            logging.info("Agent chose to search (force_search enabled)")
            return True

        # Check auto_search_decision setting
        if not cfg.auto_search_decision:
            logging.info("Agent chose not to search (auto_search disabled)")
            return False

        # Invoke search decision chain
        try:
            decision_inputs = query_inputs.copy()
            decision_raw = self._invoke_chain_safe(
                ChainName.SEARCH_DECISION, decision_inputs, str(RebuildKey.SEARCH_DECISION)
            )
            if decision_raw is None:
                return None

            # Normalize to SEARCH or NO_SEARCH
            decision_str = str(decision_raw).strip().upper().replace("_", "").replace("-", "")
            should_search = "SEARCH" in decision_str and "NO" not in decision_str
            if should_search:
                logging.info("Agent chose to search")
            else:
                logging.info("Agent chose not to search")
            return should_search

        except _exceptions.ResponseError as exc:
            if "not found" in str(exc).lower():
                _model_utils_mod.handle_missing_model(self._mark_error, "Robot", cfg.robot_model)
                return None
            logging.error("Search decision failed: %s", exc)
            self._mark_error("Search decision failed; please retry.")
            return None

    def _execute_search_pipeline(self, query_inputs: dict[str, Any], user_query: str) -> List[str]:
        """Execute the search orchestration pipeline.

        Args:
            query_inputs: Prompt inputs for LLM
            user_query: Original user query

        Returns:
            List of aggregated search result texts

        Side Effects:
            - May raise SearchAbort if search must be abandoned
        """
        # Track session-wide search count
        self._session_search_count += 1

        orchestrator = self._build_search_orchestrator()
        return orchestrator.run(
            query_inputs=query_inputs,
            user_query=user_query,
        )

    def _generate_response(
        self,
        query_inputs: dict[str, Any],
        aggregated_results: List[str],
        should_search: bool,
        one_shot: bool,
    ) -> str | None:
        """Generate the final response text.

        Args:
            query_inputs: Prompt inputs for LLM
            aggregated_results: Search results (if any)
            should_search: Whether search was performed
            one_shot: Whether this is a one-shot query

        Returns:
            Response text, or None on error

        Side Effects:
            - Streams response to stdout
        """
        # Prepare search results text
        search_results_text = "\n\n".join(aggregated_results) if should_search and aggregated_results else ""
        if search_results_text:
            search_results_text = _text_utils_mod.truncate_text(
                search_results_text, self._char_budget(MAX_SEARCH_RESULTS_CHARS)
            )

        # Build response inputs
        resp_inputs = query_inputs.copy()
        if should_search and search_results_text:
            resp_inputs["search_results"] = search_results_text
            chain_name = ChainName.RESPONSE
            logging.info("Generating response with search results")
        else:
            chain_name = ChainName.RESPONSE_NO_SEARCH
            logging.info("Generating response without search results")

        # Generate and stream response
        return self._generate_and_stream_response(resp_inputs, chain_name, one_shot)

    def _handle_query_core(self, user_query: str, one_shot: bool) -> str | None:
        """Core query handling logic - simplified for conversation system.

        Args:
            user_query: User's question
            one_shot: Whether this is a one-shot query (for --question mode)

        Returns:
            Response text or None on error
        """
        # Phase 1: Initialize and sanitize
        self._clear_error()
        self._reset_rebuild_counts()

        try:
            sanitized_query = self._sanitize_user_input(user_query)
        except ValueError as e:
            # Input validation failed (e.g., prompt injection detected)
            self._mark_error(str(e))
            return None

        # Rewrite query for clarity (resolves pronouns, expands acronyms, adds temporal context)
        write_fn = self._write if not one_shot else None
        rewritten_query = self._rewrite_user_query(sanitized_query, write_fn=write_fn)

        query_inputs = self._build_query_inputs(rewritten_query)

        # Phase 2: Determine if search is needed
        should_search = self._determine_search_necessity(query_inputs)
        if should_search is None:  # Error occurred
            return None

        # Phase 3: Execute search if needed
        aggregated_results: List[str] = []
        if should_search:
            try:
                aggregated_results = self._execute_search_pipeline(query_inputs, sanitized_query)
            except _search_orchestrator_mod.SearchAbort:
                return None

        # Phase 4: Generate response
        response_text = self._generate_response(query_inputs, aggregated_results, should_search, one_shot)
        if response_text is None:
            return None

        # Phase 5: Update conversation history
        # Store ORIGINAL query in history (not rewritten) to prevent query drift
        self.conversation.add_turn(
            user_query=sanitized_query,
            assistant_response=response_text,
            search_used=bool(aggregated_results),
        )

        return response_text


__all__ = ["Agent"]
