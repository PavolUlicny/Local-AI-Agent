"""Chain invocation utilities with retry logic for search orchestration."""

from __future__ import annotations

import logging
from typing import Any, Callable, TYPE_CHECKING

from src.exceptions import ResponseError
from src.text_utils import MAX_REBUILD_RETRIES, is_context_length_error
from src.model_utils import handle_missing_model

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.config import AgentConfig


class SearchAbort(Exception):
    """Raised when search orchestration must halt due to a fatal error."""


def invoke_chain_with_retry(
    *,
    chain: Any,
    inputs: dict[str, Any],
    rebuild_key: str,
    rebuild_label: str,
    fallback_value: str = "NO",
    raise_on_non_context_error: bool = False,
    cfg: "AgentConfig",
    reduce_context_and_rebuild: Callable[[str, str], None],
    rebuild_counts: dict[str, int],
    mark_error: Callable[[str], str],
) -> tuple[str, bool]:
    """Invoke an LLM chain with automatic context-length retry logic.

    Args:
        chain: The LangChain chain to invoke
        inputs: Input dictionary for the chain
        rebuild_key: Key for tracking rebuild count in rebuild_counts
        rebuild_label: Human-readable label for logging
        fallback_value: Value to return on non-critical failures
        raise_on_non_context_error: If True, raise SearchAbort on non-context ResponseErrors
        cfg: Agent configuration
        reduce_context_and_rebuild: Callback to reduce context and rebuild chains
        rebuild_counts: Dictionary tracking rebuild attempts by key
        mark_error: Callback to mark errors

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
            handle_missing_model(mark_error, "Robot", cfg.robot_model)
            raise SearchAbort from exc

        # Handle context length errors with retry
        if is_context_length_error(str(exc)):
            if rebuild_counts[rebuild_key] < MAX_REBUILD_RETRIES:
                reduce_context_and_rebuild(rebuild_key, rebuild_label)
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
            mark_error(f"{rebuild_label.capitalize()} failed; please retry.")
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


__all__ = ["SearchAbort", "invoke_chain_with_retry"]
