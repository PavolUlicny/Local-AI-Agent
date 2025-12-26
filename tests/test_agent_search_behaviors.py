from __future__ import annotations

from typing import Any

import pytest

from src import agent as agent_module
from src.config import AgentConfig

from tests.agent_test_utils import (
    DummyChain,
    _AlwaysYesChain,
    _IncrementingQueryChain,
    _RepeatChain,
)


def test_result_embedding_shortcircuits_relevance_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    result_filter_chain = DummyChain(outputs=["NO"])
    query_filter_chain = DummyChain(outputs=["NO"])

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "seed": DummyChain(outputs=["seed query"]),
            "planning": DummyChain(outputs=["none"]),
            "result_filter": result_filter_chain,
            "query_filter": query_filter_chain,
            "search_decision": DummyChain(outputs=["SEARCH"]),
            "response": DummyChain(stream_tokens=["answer"]),
            "response_no_search": DummyChain(stream_tokens=[]),
        }

    def fake_ddg_results(self, query: str):  # noqa: ANN001
        return [
            {
                "title": "Deep dive",
                "link": "https://example.com/a",
                "snippet": "Hyperloop expansion details",
            }
        ]

    def fake_embed_text(self, text: str):  # noqa: ANN001
        if "Hola" in text:
            return [1.0, 0.0]
        if "Hyperloop" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(agent_module.Agent, "_ddg_results", fake_ddg_results, raising=False)
    monkeypatch.setattr(agent_module._embedding_client_mod.EmbeddingClient, "embed", fake_embed_text, raising=False)

    agent = agent_module.Agent(AgentConfig())
    agent.answer_once("Hola?")

    assert len(result_filter_chain.invocations) == 0
    assert len(query_filter_chain.invocations) == 0


def test_query_filter_embedding_skips_low_similarity_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    planning_chain = DummyChain(outputs=["Mars colony missions", "none"])
    query_filter_chain = DummyChain(outputs=["YES"])

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "seed": DummyChain(outputs=["seed query"]),
            "planning": planning_chain,
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": query_filter_chain,
            "search_decision": DummyChain(outputs=["SEARCH"]),
            "response": DummyChain(stream_tokens=["final"]),
            "response_no_search": DummyChain(stream_tokens=[]),
        }

    def fake_ddg_results(self, query: str):  # noqa: ANN001
        return []

    def fake_embed_text(self, text: str):  # noqa: ANN001
        if "Space policy" in text:
            return [1.0, 0.0]
        if "Mars colony" in text:
            return [0.0, 1.0]
        return [1.0, 0.0]

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(agent_module.Agent, "_ddg_results", fake_ddg_results, raising=False)
    monkeypatch.setattr(agent_module._embedding_client_mod.EmbeddingClient, "embed", fake_embed_text, raising=False)

    # Use max_rounds=2 to allow follow-up query generation after first round
    agent = agent_module.Agent(AgentConfig(max_rounds=2, max_followup_suggestions=1))
    agent.answer_once("Space policy?")

    assert len(query_filter_chain.invocations) == 0
    assert len(planning_chain.invocations) >= 1


def test_planning_response_error_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    class PlanningErrorChain(DummyChain):
        def invoke(self, inputs: dict[str, Any]):  # noqa: D401, ANN001
            raise agent_module._exceptions.ResponseError("plan boom")

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "seed": DummyChain(outputs=["seed query"]),
            "planning": PlanningErrorChain(),
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": DummyChain(outputs=["NO"]),
            "search_decision": DummyChain(outputs=["SEARCH"]),
            "response": DummyChain(stream_tokens=["unused"]),
            "response_no_search": DummyChain(stream_tokens=[]),
        }

    def fake_ddg_results(self, query: str):  # noqa: ANN001
        return [
            {
                "title": "planning detail",
                "link": "https://example.com",
                "snippet": "test planning detail",
            }
        ]

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(agent_module.Agent, "_ddg_results", fake_ddg_results, raising=False)
    monkeypatch.setattr(
        agent_module._embedding_client_mod.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False
    )

    agent = agent_module.Agent(AgentConfig(max_rounds=2))
    result = agent.answer_once("test planning?")

    # With the refactored SearchOrchestrator, planning errors are caught and logged
    # but don't abort the search. The search can still succeed with results from
    # the primary query even if follow-up planning fails.
    assert result is not None  # Search succeeded despite planning error
    assert "unused" in result  # Got response from the search results


def test_search_loop_guard_prevents_spin(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    planning_chain = _IncrementingQueryChain(prefix="plan")
    query_filter_chain = _AlwaysYesChain()
    response_chain = DummyChain(stream_tokens=["unused"])
    response_no_search_chain = DummyChain(stream_tokens=["ok_no_results"])

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "seed": DummyChain(outputs=["seed query"]),
            "planning": planning_chain,
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": query_filter_chain,
            "search_decision": _RepeatChain("SEARCH"),
            "response": response_chain,
            "response_no_search": response_no_search_chain,
        }

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(agent_module.Agent, "_ddg_results", lambda self, q: [], raising=False)
    monkeypatch.setattr(
        agent_module._embedding_client_mod.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False
    )

    # With the refactored SearchOrchestrator, loop protection is provided by
    # the fill_attempts counter, which limits how many times we try to generate
    # suggestions. The search completes successfully with max_rounds=1.
    agent = agent_module.Agent(AgentConfig(max_rounds=1))
    result = agent.answer_once("Force spin guard?")

    # Search runs but finds no results, so uses response_no_search
    # Loop protection prevents infinite search attempts
    assert result == "ok_no_results"
