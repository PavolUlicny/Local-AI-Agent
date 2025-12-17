from __future__ import annotations

from typing import Any, cast
import importlib

try:
    _exceptions = importlib.import_module("src.exceptions")
    _text_utils_mod = importlib.import_module("src.text_utils")
    _model_utils_mod = importlib.import_module("src.model_utils")
except ModuleNotFoundError:
    _exceptions = importlib.import_module("exceptions")
    _text_utils_mod = importlib.import_module("text_utils")
    _model_utils_mod = importlib.import_module("model_utils")


def invoke_chain_safe(agent: Any, chain_name: str, inputs: dict[str, Any], rebuild_key: str | None = None) -> Any:
    try:
        return agent.chains[chain_name].invoke(inputs)
    except _exceptions.ResponseError as exc:
        msg = str(exc)
        if "not found" in msg.lower():
            raise
        if rebuild_key and _text_utils_mod.is_context_length_error(msg):
            if agent.rebuild_counts.get(rebuild_key, 0) < _text_utils_mod.MAX_REBUILD_RETRIES:
                agent._reduce_context_and_rebuild(rebuild_key, rebuild_key)
                return agent.chains[chain_name].invoke(inputs)
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
