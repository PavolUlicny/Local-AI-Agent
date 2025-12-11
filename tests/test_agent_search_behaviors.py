from __future__ import annotations

from typing import Any
import logging

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
            "context": DummyChain(outputs=["NEW_TOPIC"]),
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

    monkeypatch.setattr(agent_module, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module, "build_chains", fake_build_chains)
    monkeypatch.setattr(agent_module.Agent, "_ddg_results", fake_ddg_results, raising=False)
    monkeypatch.setattr(agent_module.EmbeddingClient, "embed", fake_embed_text, raising=False)

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
            "context": DummyChain(outputs=["NEW_TOPIC"]),
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

    monkeypatch.setattr(agent_module, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module, "build_chains", fake_build_chains)
    monkeypatch.setattr(agent_module.Agent, "_ddg_results", fake_ddg_results, raising=False)
    monkeypatch.setattr(agent_module.EmbeddingClient, "embed", fake_embed_text, raising=False)

    agent = agent_module.Agent(AgentConfig(max_rounds=1, max_followup_suggestions=1))
    agent.answer_once("Space policy?")

    assert len(query_filter_chain.invocations) == 0
    assert len(planning_chain.invocations) >= 1


def test_planning_response_error_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    class PlanningErrorChain(DummyChain):
        def invoke(self, inputs: dict[str, Any]):  # noqa: D401, ANN001
            raise agent_module.ResponseError("plan boom")

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "context": DummyChain(outputs=["NEW_TOPIC"]),
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

    monkeypatch.setattr(agent_module, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module, "build_chains", fake_build_chains)
    monkeypatch.setattr(agent_module.Agent, "_ddg_results", fake_ddg_results, raising=False)
    monkeypatch.setattr(agent_module.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False)

    agent = agent_module.Agent(AgentConfig(max_rounds=2))
    result = agent.answer_once("test planning?")

    assert result is None
    assert agent._last_error and "Query planning failed" in agent._last_error


def test_search_loop_guard_prevents_spin(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    planning_chain = _IncrementingQueryChain(prefix="plan")
    query_filter_chain = _AlwaysYesChain()
    response_chain = DummyChain(stream_tokens=["ok"])

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "context": DummyChain(outputs=["NEW_TOPIC"]),
            "seed": DummyChain(outputs=["seed query"]),
            "planning": planning_chain,
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": query_filter_chain,
            "search_decision": _RepeatChain("SEARCH"),
            "response": response_chain,
            "response_no_search": DummyChain(stream_tokens=[]),
        }

    monkeypatch.setattr(agent_module, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module, "build_chains", fake_build_chains)
    monkeypatch.setattr(agent_module.Agent, "_ddg_results", lambda self, q: [], raising=False)
    monkeypatch.setattr(agent_module.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False)

    agent = agent_module.Agent(AgentConfig(max_rounds=1))
    with caplog.at_level(logging.WARNING):
        result = agent.answer_once("Force spin guard?")

    assert result == "ok"
    assert any("Search loop aborted" in msg for msg in caplog.messages)
