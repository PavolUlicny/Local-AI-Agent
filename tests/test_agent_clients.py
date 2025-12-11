from __future__ import annotations

from typing import Any

import pytest

from src import agent as agent_module
from src import search_client as search_client_module
from src.config import AgentConfig

from tests.agent_test_utils import DummyChain


def test_search_client_closed_after_answer_once(monkeypatch: pytest.MonkeyPatch) -> None:
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

    class _Closeable:
        def __init__(self):
            self.closed = False

        def close(self):  # noqa: D401
            self.closed = True

    monkeypatch.setattr(agent_module, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module, "build_chains", fake_build_chains)
    monkeypatch.setattr(agent_module.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False)

    agent = agent_module.Agent(AgentConfig(no_auto_search=True))
    closable: Any = _Closeable()
    agent.search_client = closable
    agent.answer_once("Hi?")

    assert agent.search_client is None
    assert closable.closed is True


def test_ddg_results_closes_client_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeDDGS:
        def __init__(self):  # noqa: D401
            self.closed = False

        def text(self, *_, **__):  # noqa: D401, ANN002, ANN003
            return [
                {
                    "title": "Example",
                    "link": "https://example.com",
                    "snippet": "Sample snippet",
                }
            ]

        def close(self):  # noqa: D401
            self.closed = True

    fake_clients: list[_FakeDDGS] = []

    def _factory(*_, **__):  # noqa: ANN002, ANN003
        client = _FakeDDGS()
        fake_clients.append(client)
        return client

    monkeypatch.setattr(search_client_module, "DDGS", _factory)

    agent = agent_module.Agent(AgentConfig())
    results = agent._ddg_results("test")

    assert results
    assert fake_clients, "DDGS factory should have been called"
    assert all(client.closed for client in fake_clients)
    assert agent.search_client is not None
    placeholder_client = getattr(agent.search_client, "_client", None)
    if placeholder_client is not None:
        assert getattr(placeholder_client, "closed", True) is True


def test_ddg_timeout_respects_fractional(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[float] = []

    class _FakeDDGS:
        def __init__(self, timeout=None):  # noqa: D401, ANN001
            recorded.append(timeout)
            self.closed = False

        def text(self, *_, **__):  # noqa: D401, ANN002, ANN003
            return []

        def close(self):  # noqa: D401
            self.closed = True

    monkeypatch.setattr(search_client_module, "DDGS", _FakeDDGS)
    monkeypatch.setattr(agent_module.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False)

    agent = agent_module.Agent(AgentConfig(search_timeout=0.5, no_auto_search=True))
    agent._ddg_results("hello")

    assert recorded and all(val == 0.5 for val in recorded)
