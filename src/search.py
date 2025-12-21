from __future__ import annotations

from typing import Any, List, Set, cast
import importlib

try:
    _search_orchestrator_mod = importlib.import_module("src.search_orchestrator")
    _search_context_mod = importlib.import_module("src.search_context")
except ModuleNotFoundError:
    _search_orchestrator_mod = importlib.import_module("search_orchestrator")
    _search_context_mod = importlib.import_module("search_context")


def build_search_orchestrator(agent: Any) -> Any:
    """Create and return a SearchOrchestrator instance wired to the agent.

    The helper mirrors the previous `Agent._build_search_orchestrator` logic
    but takes an `agent` instance so it can be moved out of the class.
    """
    # Allow tests to monkeypatch `agent._build_search_orchestrator` and have
    # this function respect that. If the agent provides a custom builder,
    # call it; otherwise construct the orchestrator here.
    # Only honor an instance-level override (e.g., monkeypatch on the agent
    # instance). If the attribute only exists on the Agent class (our own
    # method that delegates here) calling it would recurse, so avoid that.
    if "_build_search_orchestrator" in getattr(agent, "__dict__", {}):
        return agent._build_search_orchestrator()

    # Build SearchServices bundle
    services = _search_context_mod.SearchServices(
        cfg=agent.cfg,
        chains=agent.chains,
        embedding_client=agent.embedding_client,
        ddg_results=agent._ddg_results,
        inputs_builder=agent._inputs,
        reduce_context_and_rebuild=agent._reduce_context_and_rebuild,
        mark_error=agent._mark_error,
        context_similarity=agent._context_similarity,
        char_budget=agent._char_budget,
        rebuild_counts=agent.rebuild_counts,
    )

    return cast(
        Any,
        _search_orchestrator_mod.SearchOrchestrator(services),
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

    # Build SearchContext from QueryContext and parameters
    search_context = _search_context_mod.SearchContext(
        current_datetime=ctx.current_datetime,
        current_year=ctx.current_year,
        current_month=ctx.current_month,
        current_day=ctx.current_day,
        user_query=user_query,
        conversation_text=ctx.conversation_text,
        prior_responses_text=ctx.prior_responses_text,
        question_embedding=question_embedding,
        topic_embedding_current=topic_embedding_current,
    )

    aggregated_results, updated_keywords = orchestrator.run(
        context=search_context,
        should_search=should_search,
        primary_search_query=primary_search_query,
    )

    return aggregated_results, updated_keywords
