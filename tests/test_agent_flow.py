from __future__ import annotations

from typing import Any, Iterable

import pytest

from src import agent as agent_module
from src.config import AgentConfig


class DummyChain:
    def __init__(self, *, outputs: Iterable[str] | None = None, stream_tokens: Iterable[str] | None = None):
        self.outputs = list(outputs or [])
        self.stream_tokens = list(stream_tokens or [])
        self.invocations: list[dict[str, Any]] = []

    def invoke(self, inputs: dict[str, Any]) -> str:
        self.invocations.append(inputs)
        if self.outputs:
            return self.outputs.pop(0)
        return ""

    def stream(self, inputs: dict[str, Any]):
        self.invocations.append(inputs)
        for token in self.stream_tokens:
            yield token


def test_answer_once_without_search_uses_response_no_search(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    constructed_chains: list[dict[str, DummyChain]] = []

    def fake_build_llms(cfg: AgentConfig):
        return "robot", "assistant"

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        chains = {
            "context": DummyChain(outputs=["NEW_TOPIC"]),
            "seed": DummyChain(outputs=["seed query"]),
            "planning": DummyChain(outputs=["none"]),
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": DummyChain(outputs=["NO"]),
            "search_decision": DummyChain(outputs=["NO_SEARCH"]),
            "response": DummyChain(stream_tokens=["unused"]),
            "response_no_search": DummyChain(stream_tokens=["chunk1", "chunk2"]),
        }
        constructed_chains.append(chains)
        return chains

    def fail_if_search_runs(self, query: str):  # noqa: ANN001
        raise AssertionError(f"Search should not run for query: {query}")

    monkeypatch.setattr(agent_module, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module, "build_chains", fake_build_chains)
    monkeypatch.setattr(agent_module.Agent, "_ddg_results", fail_if_search_runs, raising=False)
    monkeypatch.setattr(agent_module.Agent, "_embed_text", lambda self, text: [1.0, 0.0])

    agent = agent_module.Agent(AgentConfig())
    result = agent.answer_once("Hola?")
    captured = capsys.readouterr()

    assert result == "chunk1chunk2"
    assert "chunk1chunk2" in captured.out

    assert constructed_chains, "build_chains should have been invoked"
    chains = constructed_chains[0]
    assert len(chains["search_decision"].invocations) == 1
    assert len(chains["response_no_search"].invocations) == 1


def test_result_embedding_shortcircuits_relevance_llm(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
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
    monkeypatch.setattr(agent_module.Agent, "_embed_text", fake_embed_text, raising=False)

    agent = agent_module.Agent(AgentConfig())
    agent.answer_once("Hola?")
    capsys.readouterr()

    assert len(result_filter_chain.invocations) == 0
    assert len(query_filter_chain.invocations) == 0


def test_query_filter_embedding_skips_low_similarity_candidates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
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
    monkeypatch.setattr(agent_module.Agent, "_embed_text", fake_embed_text, raising=False)

    agent = agent_module.Agent(AgentConfig(max_rounds=1, max_followup_suggestions=1))
    agent.answer_once("Space policy?")
    capsys.readouterr()

    assert len(query_filter_chain.invocations) == 0
    assert len(planning_chain.invocations) >= 1


def test_zero_context_turns_drop_history(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    response_chain = DummyChain(stream_tokens=["out"])

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "context": DummyChain(outputs=["FOLLOW_UP"]),
            "seed": DummyChain(outputs=["seed"]),
            "planning": DummyChain(outputs=["none"]),
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": DummyChain(outputs=["NO"]),
            "search_decision": DummyChain(outputs=["NO_SEARCH", "NO_SEARCH"]),
            "response": DummyChain(stream_tokens=["unused"]),
            "response_no_search": response_chain,
        }

    monkeypatch.setattr(agent_module, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module, "build_chains", fake_build_chains)
    monkeypatch.setattr(agent_module.Agent, "_embed_text", lambda self, text: [1.0, 0.0], raising=False)

    agent = agent_module.Agent(AgentConfig(no_auto_search=True, max_context_turns=0))
    agent.answer_once("First?")
    agent.answer_once("Second?")
    capsys.readouterr()

    assert agent.topics, "topic list should not be empty"
    assert all(not topic.turns for topic in agent.topics)
