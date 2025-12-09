from __future__ import annotations

from typing import Any, Iterable
import logging

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


class _RepeatChain(DummyChain):
    def __init__(self, value: str):
        super().__init__()
        self.value = value

    def invoke(self, inputs: dict[str, Any]) -> str:  # noqa: D401
        self.invocations.append(inputs)
        return self.value


class _AlwaysYesChain(DummyChain):
    def invoke(self, inputs: dict[str, Any]) -> str:  # noqa: D401
        self.invocations.append(inputs)
        return "YES"


class _IncrementingQueryChain(DummyChain):
    def __init__(self, prefix: str = "query"):
        super().__init__()
        self.prefix = prefix
        self.counter = 0

    def invoke(self, inputs: dict[str, Any]) -> str:  # noqa: D401
        self.invocations.append(inputs)
        self.counter += 1
        return f"{self.prefix} {self.counter}"


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


def test_rebuild_counts_reset_each_query(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
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

    monkeypatch.setattr(agent_module, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module, "build_chains", fake_build_chains)
    monkeypatch.setattr(agent_module.Agent, "_embed_text", lambda self, text: [1.0, 0.0], raising=False)

    agent = agent_module.Agent(AgentConfig(no_auto_search=True))
    # Seed nonzero rebuild counters to ensure they are cleared per turn
    agent.rebuild_counts = {k: 2 for k in agent.rebuild_counts}
    agent.answer_once("Hello?")
    capsys.readouterr()

    assert all(count == 0 for count in agent.rebuild_counts.values())


def test_fatal_error_bubbles_via_last_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    class ErrorChain(DummyChain):
        def stream(self, inputs: dict[str, Any]):  # noqa: D401, ANN001
            raise agent_module.ResponseError("model not found")

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

    monkeypatch.setattr(agent_module, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module, "build_chains", fake_build_chains)
    monkeypatch.setattr(agent_module.Agent, "_embed_text", lambda self, text: [1.0, 0.0], raising=False)

    agent = agent_module.Agent(AgentConfig(no_auto_search=True))
    result = agent.answer_once("Hola?")
    capsys.readouterr()

    assert result is None
    assert agent._last_error is not None and "not found" in agent._last_error


def test_search_client_closed_after_answer_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
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
    monkeypatch.setattr(agent_module.Agent, "_embed_text", lambda self, text: [1.0, 0.0], raising=False)

    agent = agent_module.Agent(AgentConfig(no_auto_search=True))
    closable: Any = _Closeable()
    agent.search_client = closable
    agent.answer_once("Hi?")
    capsys.readouterr()

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

    def _factory(*_, **__):
        client = _FakeDDGS()
        fake_clients.append(client)
        return client

    monkeypatch.setattr(agent_module, "DDGS", _factory)

    agent = agent_module.Agent(AgentConfig())
    results = agent._ddg_results("test")

    assert results
    assert fake_clients, "DDGS factory should have been called"
    assert all(client.closed for client in fake_clients)
    # Agent keeps a placeholder client for health checks, but it should be closed already.
    assert agent.search_client is not None
    assert getattr(agent.search_client, "closed", True) is True


def test_force_search_skips_classifier(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    search_decision_chain = DummyChain(outputs=["NO_SEARCH"])  # should be ignored when forced
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

    monkeypatch.setattr(agent_module, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module, "build_chains", fake_build_chains)
    monkeypatch.setattr(agent_module.Agent, "_ddg_results", lambda self, q: [], raising=False)
    monkeypatch.setattr(agent_module.Agent, "_embed_text", lambda self, text: [1.0, 0.0], raising=False)

    agent = agent_module.Agent(AgentConfig(force_search=True))
    result = agent.answer_once("Should search")
    capsys.readouterr()

    assert result == "forced"
    assert len(search_decision_chain.invocations) == 0


def test_search_decision_response_error_is_fatal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    class ErrorChain(DummyChain):
        def invoke(self, inputs: dict[str, Any]):  # noqa: D401, ANN001
            raise agent_module.ResponseError("classifier boom")

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "context": DummyChain(outputs=["NEW_TOPIC"]),
            "seed": DummyChain(outputs=["seed"]),
            "planning": DummyChain(outputs=["none"]),
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": DummyChain(outputs=["NO"]),
            "search_decision": ErrorChain(),
            "response": DummyChain(stream_tokens=["unused"]),
            "response_no_search": DummyChain(stream_tokens=["unused"]),
        }

    monkeypatch.setattr(agent_module, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module, "build_chains", fake_build_chains)
    monkeypatch.setattr(agent_module.Agent, "_embed_text", lambda self, text: [1.0, 0.0], raising=False)

    agent = agent_module.Agent(AgentConfig())
    result = agent.answer_once("Hola?")
    capsys.readouterr()

    assert result is None
    assert agent._last_error and "Search decision failed" in agent._last_error


def test_seed_response_error_is_fatal(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    class SeedErrorChain(DummyChain):
        def invoke(self, inputs: dict[str, Any]):  # noqa: D401, ANN001
            raise agent_module.ResponseError("seed boom")

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "context": DummyChain(outputs=["NEW_TOPIC"]),
            "seed": SeedErrorChain(),
            "planning": DummyChain(outputs=["none"]),
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": DummyChain(outputs=["NO"]),
            "search_decision": DummyChain(outputs=["SEARCH"]),
            "response": DummyChain(stream_tokens=["unused"]),
            "response_no_search": DummyChain(stream_tokens=["unused"]),
        }

    monkeypatch.setattr(agent_module, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module, "build_chains", fake_build_chains)
    monkeypatch.setattr(agent_module.Agent, "_ddg_results", lambda self, q: [], raising=False)
    monkeypatch.setattr(agent_module.Agent, "_embed_text", lambda self, text: [1.0, 0.0], raising=False)

    agent = agent_module.Agent(AgentConfig())
    result = agent.answer_once("Hola?")
    capsys.readouterr()

    assert result is None
    assert agent._last_error and "Seed query generation failed" in agent._last_error


def test_planning_response_error_is_fatal(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
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
    monkeypatch.setattr(agent_module.Agent, "_embed_text", lambda self, text: [1.0, 0.0], raising=False)

    agent = agent_module.Agent(AgentConfig(max_rounds=2))
    result = agent.answer_once("test planning?")
    capsys.readouterr()

    assert result is None
    assert agent._last_error and "Query planning failed" in agent._last_error


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

    monkeypatch.setattr(agent_module, "DDGS", _FakeDDGS)
    monkeypatch.setattr(agent_module.Agent, "_embed_text", lambda self, text: [1.0, 0.0], raising=False)

    agent = agent_module.Agent(AgentConfig(search_timeout=0.5, no_auto_search=True))
    agent._ddg_results("hello")

    assert recorded and all(val == 0.5 for val in recorded)


def test_stream_error_does_not_persist_state(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
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

    monkeypatch.setattr(agent_module, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module, "build_chains", fake_build_chains)
    monkeypatch.setattr(agent_module.Agent, "_embed_text", lambda self, text: [1.0, 0.0], raising=False)

    agent = agent_module.Agent(AgentConfig(no_auto_search=True))
    result = agent.answer_once("Hello?")
    capsys.readouterr()

    assert result is None
    assert agent.topics == []


def test_search_loop_guard_prevents_spin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
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
    monkeypatch.setattr(agent_module.Agent, "_embed_text", lambda self, text: [1.0, 0.0], raising=False)

    agent = agent_module.Agent(AgentConfig(max_rounds=1))
    with caplog.at_level(logging.WARNING):
        result = agent.answer_once("Force spin guard?")
    capsys.readouterr()

    assert result == "ok"
    assert any("Search loop aborted" in msg for msg in caplog.messages)
