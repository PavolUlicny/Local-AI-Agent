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

    agent = agent_module.Agent(AgentConfig())
    result = agent.answer_once("Hola?")
    captured = capsys.readouterr()

    assert result == "chunk1chunk2"
    assert "chunk1chunk2" in captured.out

    assert constructed_chains, "build_chains should have been invoked"
    chains = constructed_chains[0]
    assert len(chains["search_decision"].invocations) == 1
    assert len(chains["response_no_search"].invocations) == 1
