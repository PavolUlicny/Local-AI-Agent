from __future__ import annotations

from typing import Any, cast
import importlib

try:
    _exceptions = importlib.import_module("src.exceptions")
    _text_utils_mod = importlib.import_module("src.text_utils")
except ModuleNotFoundError:
    _exceptions = importlib.import_module("exceptions")
    _text_utils_mod = importlib.import_module("text_utils")


def invoke_chain_safe(
    agent: Any,
    chain_name: str,
    inputs: dict[str, Any],
    rebuild_key: str | None = None,
    fallback_on_context_error: Any = None,
    fallback_on_generic_error: Any = None,
) -> Any:
    """Invoke a chain with automatic retry on context length errors and optional fallbacks.

    Args:
        agent: The Agent instance
        chain_name: Name of the chain to invoke
        inputs: Input dictionary for the chain
        rebuild_key: Key for tracking rebuild count (enables context retry if provided)
        fallback_on_context_error: Value to return if context retries exhausted (instead of raising)
        fallback_on_generic_error: Value to return on non-context ResponseError (instead of raising)

    Returns:
        Chain result, or fallback value if error handling triggered

    Raises:
        ResponseError: On "model not found" errors, or if no fallback provided for other errors
    """
    import logging

    try:
        return agent.chains[chain_name].invoke(inputs)
    except _exceptions.ResponseError as exc:
        msg = str(exc)
        # Always propagate "model not found" errors
        if "not found" in msg.lower():
            raise
        # Handle context length errors with retry + optional fallback
        if rebuild_key and _text_utils_mod.is_context_length_error(msg):
            if agent.rebuild_counts.get(rebuild_key, 0) < _text_utils_mod.MAX_REBUILD_RETRIES:
                agent._reduce_context_and_rebuild(rebuild_key, rebuild_key)
                return agent.chains[chain_name].invoke(inputs)
            # Retries exhausted: use fallback or raise
            if fallback_on_context_error is not None:
                logging.info(
                    "Reached %s rebuild cap; using fallback value.",
                    rebuild_key,
                )
                return fallback_on_context_error
            raise
        # Handle other ResponseErrors with optional fallback
        if fallback_on_generic_error is not None:
            return fallback_on_generic_error
        raise


def inputs(
    agent: Any,
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
    agent: Any,
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
        chain_name = "response"
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
        chain_name = "response_no_search"
    return resp_inputs, chain_name


def decide_should_search(
    agent: Any,
    ctx: Any,
    user_query: str,
    prior_responses_text: str,
) -> bool:
    """Run the search_decision classifier and return True if SEARCH decided.

    Handles:
    - Model not found errors (raises ResponseError to propagate to caller)
    - Context length errors (retry with reduced context up to MAX_REBUILD_RETRIES)
    - Generic ResponseError (raises to propagate to caller)
    - Unexpected errors (defaults to True for safety - better to search than miss info)

    Args:
        agent: The Agent instance
        ctx: QueryContext with datetime and conversation info
        user_query: The user's question
        prior_responses_text: Previous responses for context

    Returns:
        bool: True if should search, False otherwise

    Raises:
        ResponseError: On model not found or other chain errors (propagates to caller)
    """
    import logging

    try:
        decision_raw = invoke_chain_safe(
            agent,
            "search_decision",
            inputs(
                agent,
                ctx.current_datetime,
                ctx.current_year,
                ctx.current_month,
                ctx.current_day,
                ctx.conversation_text,
                user_query,
                known_answers=prior_responses_text,
            ),
            rebuild_key="search_decision",
            fallback_on_context_error="NO_SEARCH",  # Safe default when retries exhausted
        )
        decision_validated = cast(
            str, _text_utils_mod.regex_validate(str(decision_raw), _text_utils_mod.PATTERN_SEARCH_DECISION, "SEARCH")
        )
        return decision_validated == "SEARCH"
    except _exceptions.ResponseError:
        # Propagate ResponseError (model not found, etc.) to caller
        raise
    except Exception as exc:
        # Default to SEARCH on unexpected errors (safety: don't suppress needed lookups)
        logging.warning("Search decision crashed; defaulting to SEARCH. Error: %s", exc)
        return True


def generate_search_seed(
    agent: Any,
    ctx: Any,
    user_query: str,
    prior_responses_text: str,
) -> str:
    """Generate a refined search query via the seed chain.

    Handles:
    - Model not found errors (raises ResponseError to propagate to caller)
    - Context length errors (retry with reduced context up to MAX_REBUILD_RETRIES)
    - Generic ResponseError (raises to propagate to caller)
    - Unexpected errors (fallback to original user_query)

    Args:
        agent: The Agent instance
        ctx: QueryContext with datetime and conversation info
        user_query: The user's question
        prior_responses_text: Previous responses for context

    Returns:
        str: Refined search query, or original user_query on failure

    Raises:
        ResponseError: On model not found or other chain errors (propagates to caller)
    """
    import logging

    try:
        seed_raw = invoke_chain_safe(
            agent,
            "seed",
            inputs(
                agent,
                ctx.current_datetime,
                ctx.current_year,
                ctx.current_month,
                ctx.current_day,
                ctx.conversation_text,
                user_query,
                known_answers=prior_responses_text,
            ),
            rebuild_key="seed",
            fallback_on_context_error=user_query,  # Use original query if retries exhausted
        )
        seed_text = str(seed_raw).strip()
        return cast(str, _text_utils_mod.pick_seed_query(seed_text, user_query))
    except _exceptions.ResponseError:
        # Propagate ResponseError (model not found, etc.) to caller
        raise
    except Exception as exc:
        # Fallback to original query on unexpected errors
        logging.warning("Seed generation failed unexpectedly; using original query. Error: %s", exc)
        return user_query
