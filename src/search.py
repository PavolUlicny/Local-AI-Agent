from __future__ import annotations

from typing import Any, cast

from . import search_orchestrator as _search_orchestrator_mod
from . import search_context as _search_context_mod


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
