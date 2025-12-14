from __future__ import annotations

from typing import Any

import pytest

from src import agent as agent_module
from src.config import AgentConfig

from tests.agent_test_utils import DummyChain


def test_answer_once_without_search_uses_response_no_search(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    constructed_chains: list[dict[str, DummyChain]] = []

    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
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

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(agent_module.Agent, "_ddg_results", fail_if_search_runs, raising=False)
    monkeypatch.setattr(agent_module._embedding_client_mod.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0])

    agent = agent_module.Agent(AgentConfig())
    result = agent.answer_once("Hola?")
    captured = capsys.readouterr()

    assert result == "chunk1chunk2"
    assert "chunk1chunk2" in captured.out

    assert constructed_chains, "build_chains should have been invoked"
    chains = constructed_chains[0]
    assert len(chains["search_decision"].invocations) == 1
    assert len(chains["response_no_search"].invocations) == 1


def test_zero_context_turns_drop_history(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(
        agent_module._embedding_client_mod.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False
    )

    agent = agent_module.Agent(AgentConfig(no_auto_search=True, max_context_turns=0))
    agent.answer_once("First?")
    agent.answer_once("Second?")

    assert agent.topics, "topic list should not be empty"
    assert all(not topic.turns for topic in agent.topics)


def test_rebuild_counts_reset_each_query(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    response_chain = DummyChain(stream_tokens=["ok"])

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "context": DummyChain(outputs=["NEW_TOPIC"]),
            "seed": DummyChain(outputs=["seed"]),
            "planning": DummyChain(outputs=["none"]),
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": DummyChain(outputs=["NO"]),
            "search_decision": DummyChain(outputs=["NO_SEARCH"]),
            "response": DummyChain(stream_tokens=["unused"]),
            "response_no_search": response_chain,
        }

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(
        agent_module._embedding_client_mod.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False
    )

    agent = agent_module.Agent(AgentConfig(no_auto_search=True))
    agent.rebuild_counts = dict.fromkeys(agent.rebuild_counts, 2)
    agent.answer_once("Hello?")

    assert all(count == 0 for count in agent.rebuild_counts.values())


def test_fatal_error_bubbles_via_last_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    class ErrorChain(DummyChain):
        def stream(self, inputs: dict[str, Any]):  # noqa: D401, ANN001
            raise agent_module._exceptions.ResponseError("assistant model not found")

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "context": DummyChain(outputs=["NEW_TOPIC"]),
            "seed": DummyChain(outputs=["seed"]),
            "planning": DummyChain(outputs=["none"]),
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": DummyChain(outputs=["NO"]),
            "search_decision": DummyChain(outputs=["NO_SEARCH"]),
            "response": ErrorChain(),
            "response_no_search": ErrorChain(),
        }

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(
        agent_module._embedding_client_mod.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False
    )

    agent = agent_module.Agent(AgentConfig(no_auto_search=True))
    result = agent.answer_once("Hola?")

    assert result is None
    expected = f"Assistant model '{agent.cfg.assistant_model}' not found. Run 'ollama pull {agent.cfg.assistant_model}' and retry."
    assert agent._last_error == expected


def test_force_search_skips_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    search_decision_chain = DummyChain(outputs=["NO_SEARCH"])
    response_chain = DummyChain(stream_tokens=["forced"])

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "context": DummyChain(outputs=["NEW_TOPIC"]),
            "seed": DummyChain(outputs=["seed"]),
            "planning": DummyChain(outputs=["none"]),
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": DummyChain(outputs=["NO"]),
            "search_decision": search_decision_chain,
            "response": response_chain,
            "response_no_search": DummyChain(stream_tokens=[]),
        }

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(agent_module.Agent, "_ddg_results", lambda self, q: [], raising=False)
    monkeypatch.setattr(
        agent_module._embedding_client_mod.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False
    )

    agent = agent_module.Agent(AgentConfig(force_search=True))
    result = agent.answer_once("Should search")

    assert result == "forced"
    assert len(search_decision_chain.invocations) == 0


def test_stream_error_does_not_persist_state(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    class BrokenChain(DummyChain):
        def stream(self, inputs: dict[str, Any]):  # noqa: D401, ANN001
            yield "partial"
            raise RuntimeError("stream boom")

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "context": DummyChain(outputs=["NEW_TOPIC"]),
            "seed": DummyChain(outputs=["seed"]),
            "planning": DummyChain(outputs=["none"]),
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": DummyChain(outputs=["NO"]),
            "search_decision": DummyChain(outputs=["NO_SEARCH"]),
            "response": BrokenChain(),
            "response_no_search": BrokenChain(),
        }

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(
        agent_module._embedding_client_mod.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False
    )

    agent = agent_module.Agent(AgentConfig(no_auto_search=True))
    result = agent.answer_once("Hello?")

    assert result is None
    assert agent.topics == []
