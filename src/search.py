from __future__ import annotations

from typing import Any, List, Set, cast
import importlib

try:
    _search_orchestrator_mod = importlib.import_module("src.search_orchestrator")
    _text_utils_mod = importlib.import_module("src.text_utils")
except ModuleNotFoundError:
    _search_orchestrator_mod = importlib.import_module("search_orchestrator")
    _text_utils_mod = importlib.import_module("text_utils")


def build_search_orchestrator(agent: Any) -> Any:
    """Create and return a SearchOrchestrator instance wired to the agent.

    The helper mirrors the previous `Agent._build_search_orchestrator` logic
    but takes an `agent` instance so it can be moved out of the class.
    """
    # Allow tests to monkeypatch `agent._build_search_orchestrator` and have
    # this function respect that. If the agent provides a custom builder,
    # call it; otherwise construct the orchestrator here.
    if hasattr(agent, "_build_search_orchestrator"):
        return agent._build_search_orchestrator()
    return cast(
        Any,
        _search_orchestrator_mod.SearchOrchestrator(
            agent.cfg,
            ddg_results=agent._ddg_results,
            embedding_client=agent.embedding_client,
            context_similarity=agent._context_similarity,
            inputs_builder=agent._inputs,
            reduce_context_and_rebuild=agent._reduce_context_and_rebuild,
            rebuild_counts=agent.rebuild_counts,
            char_budget=agent._char_budget,
            mark_error=agent._mark_error,
        ),
    )


def run_search_rounds(
    agent: Any,
    ctx: Any,
    user_query: str,
    should_search: bool,
    primary_search_query: str,
    question_embedding: List[float] | None,
    topic_embedding_current: List[float] | None,
    topic_keywords: Set[str],
) -> tuple[List[str], Set[str]]:
    """Run search orchestration using a freshly-built SearchOrchestrator.

    Returns the tuple `(aggregated_results, topic_keywords)`.
    """
    orchestrator = build_search_orchestrator(agent)
    aggregated_results, topic_keywords = orchestrator.run(
        chains=agent.chains,
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
