from __future__ import annotations

from typing import Any

import pytest

from src import agent as agent_module
from src.config import AgentConfig

from tests.agent_test_utils import DummyChain


def test_search_decision_response_error_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    class ErrorChain(DummyChain):
        def invoke(self, inputs: dict[str, Any]):  # noqa: D401, ANN001
            raise agent_module._exceptions.ResponseError("classifier boom")

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "planning": DummyChain(outputs=["none"]),
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": DummyChain(outputs=["NO"]),
            "search_decision": ErrorChain(),
            "response": DummyChain(stream_tokens=["unused"]),
            "response_no_search": DummyChain(stream_tokens=["unused"]),
        }

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(
        agent_module._embedding_client_mod.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False
    )

    agent = agent_module.Agent(AgentConfig())
    result = agent.answer_once("Hola?")

    assert result is None
    assert agent._last_error and "Search decision failed" in agent._last_error
