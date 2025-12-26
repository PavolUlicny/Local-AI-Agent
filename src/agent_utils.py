from __future__ import annotations

from typing import Any, cast, TYPE_CHECKING

from . import exceptions as _exceptions
from . import text_utils as _text_utils_mod
from .constants import ChainName, RebuildKey, MAX_REBUILD_RETRIES

if TYPE_CHECKING:  # pragma: no cover
    from .protocols import AgentProtocol


def invoke_chain_safe(
    agent: "AgentProtocol",
    chain_name: ChainName | str,
    inputs: dict[str, Any],
    rebuild_key: RebuildKey | str | None = None,
    fallback_on_context_error: Any = None,
    fallback_on_generic_error: Any = None,
) -> Any:
    """Invoke a chain with automatic retry on context length errors and optional fallbacks.

    Args:
        agent: The Agent instance
        chain_name: Name of the chain to invoke (ChainName enum or string)
        inputs: Input dictionary for the chain
        rebuild_key: Key for tracking rebuild count (RebuildKey enum or string, enables context retry if provided)
        fallback_on_context_error: Value to return if context retries exhausted (instead of raising)
        fallback_on_generic_error: Value to return on non-context ResponseError (instead of raising)

    Returns:
        Chain result, or fallback value if error handling triggered

    Raises:
        ResponseError: On "model not found" errors, or if no fallback provided for other errors
    """
    import logging

    # Allow both enums and strings for flexibility
    if isinstance(chain_name, str):
        try:
            chain_name = ChainName(chain_name)
        except ValueError:
            # If not a valid enum, try to find by string value
            chain_name_obj = cast(ChainName, chain_name)
        else:
            chain_name_obj = chain_name
    else:
        chain_name_obj = chain_name

    rebuild_key_str: str | None = None
    if rebuild_key is not None:
        if isinstance(rebuild_key, str):
            rebuild_key_str = rebuild_key
        else:
            rebuild_key_str = str(rebuild_key)

    try:
        return agent.chains[chain_name_obj].invoke(inputs)
    except _exceptions.ResponseError as exc:
        msg = str(exc)
        # Always propagate "model not found" errors
        if "not found" in msg.lower():
            raise
        # Handle context length errors with retry + optional fallback
        if rebuild_key_str and _text_utils_mod.is_context_length_error(msg):
            if agent.rebuild_counts.get(rebuild_key_str, 0) < MAX_REBUILD_RETRIES:
                agent._reduce_context_and_rebuild(rebuild_key_str, rebuild_key_str)
                return agent.chains[chain_name_obj].invoke(inputs)
            # Retries exhausted: use fallback or raise
            if fallback_on_context_error is not None:
                logging.info(
                    "Reached %s rebuild cap; using fallback value.",
                    rebuild_key_str,
                )
                return fallback_on_context_error
            raise
        # Handle other ResponseErrors with optional fallback
        if fallback_on_generic_error is not None:
            return fallback_on_generic_error
        raise


def inputs(
    agent: "AgentProtocol",
    current_datetime: str,
    current_year: str,
    current_month: str,
    current_day: str,
    conversation_text: str,
    user_query: str,
    **overrides: Any,
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        agent.build_inputs(
            current_datetime, current_year, current_month, current_day, conversation_text, user_query, **overrides
        ),
    )


def build_resp_inputs(
    agent: "AgentProtocol",
    current_datetime: str,
    current_year: str,
    current_month: str,
    current_day: str,
    conversation_text: str,
    user_query: str,
    should_search: bool,
    prior_responses_text: str,
    search_results_text: str | None = None,
) -> tuple[dict[str, Any], ChainName]:
    if should_search:
        resp_inputs = inputs(
            agent,
            current_datetime,
            current_year,
            current_month,
            current_day,
            conversation_text,
            user_query,
            search_results=search_results_text or "",
            prior_responses=prior_responses_text,
        )
        chain_name = ChainName.RESPONSE
    else:
        resp_inputs = inputs(
            agent,
            current_datetime,
            current_year,
            current_month,
            current_day,
            conversation_text,
            user_query,
            prior_responses=prior_responses_text,
        )
        chain_name = ChainName.RESPONSE_NO_SEARCH
    return resp_inputs, chain_name
