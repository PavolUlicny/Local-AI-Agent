"""Tests for query rewriting functionality."""

from __future__ import annotations

from io import StringIO
from typing import Any

import pytest

from src import agent as agent_module
from src.agent import Agent
from src.config import AgentConfig

from tests.agent_test_utils import DummyChain


def test_query_rewrite_resolves_pronouns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that pronouns are resolved using conversation history."""
    cfg = AgentConfig()
    out = StringIO()

    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    mock_chain_output = "Joe Biden vice president"

    class QueryRewriteChain:
        def invoke(self, inputs: dict[str, Any]):
            # Verify conversation history is included
            assert "conversation_history" in inputs
            assert "Joe Biden" in inputs["conversation_history"]
            return mock_chain_output

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "planning": DummyChain(outputs=["none"]),
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": DummyChain(outputs=["NO"]),
            "query_rewrite": QueryRewriteChain(),
            "search_decision": DummyChain(outputs=["NO_SEARCH"]),
            "response": DummyChain(stream_tokens=["test"]),
            "response_no_search": DummyChain(stream_tokens=["test"]),
        }

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(
        agent_module._embedding_client_mod.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False
    )

    agent = Agent(cfg, output_stream=out, is_tty=True)

    # Add conversation context
    agent.conversation.add_turn(
        user_query="Who is Joe Biden?",
        assistant_response="Joe Biden is the 46th President of the United States.",
        search_used=True,
    )

    # Test rewriting
    result = agent._rewrite_user_query("his vp", write_fn=agent._write)

    assert result == "Joe Biden vice president"
    assert "Understood as:" in out.getvalue()


def test_query_rewrite_expands_acronyms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that acronyms are expanded."""
    cfg = AgentConfig()

    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    mock_chain_output = "United States president"

    class QueryRewriteChain:
        def invoke(self, inputs: dict[str, Any]):
            assert inputs["user_question"] == "us president"
            return mock_chain_output

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "planning": DummyChain(outputs=["none"]),
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": DummyChain(outputs=["NO"]),
            "query_rewrite": QueryRewriteChain(),
            "search_decision": DummyChain(outputs=["NO_SEARCH"]),
            "response": DummyChain(stream_tokens=["test"]),
            "response_no_search": DummyChain(stream_tokens=["test"]),
        }

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(
        agent_module._embedding_client_mod.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False
    )

    agent = Agent(cfg, output_stream=StringIO(), is_tty=False)
    result = agent._rewrite_user_query("us president")
    assert result == "United States president"


def test_query_rewrite_adds_temporal_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that temporal context is added for time-sensitive queries."""
    cfg = AgentConfig()

    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    captured_inputs = {}

    class QueryRewriteChain:
        def invoke(self, inputs: dict[str, Any]):
            captured_inputs.update(inputs)
            # Return query with year
            return f"current president {inputs['current_year']}"

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "planning": DummyChain(outputs=["none"]),
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": DummyChain(outputs=["NO"]),
            "query_rewrite": QueryRewriteChain(),
            "search_decision": DummyChain(outputs=["NO_SEARCH"]),
            "response": DummyChain(stream_tokens=["test"]),
            "response_no_search": DummyChain(stream_tokens=["test"]),
        }

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(
        agent_module._embedding_client_mod.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False
    )

    agent = Agent(cfg, output_stream=StringIO(), is_tty=False)
    result = agent._rewrite_user_query("current president")

    # Verify temporal context was provided to chain
    assert "current_year" in captured_inputs
    assert "current_month" in captured_inputs
    assert "current_day" in captured_inputs
    # Verify result contains year
    assert captured_inputs["current_year"] in result


def test_query_rewrite_skips_short_queries() -> None:
    """Test that very short queries are not rewritten."""
    # Note: This test doesn't need monkeypatch since it short-circuits before chain invocation
    cfg = AgentConfig()
    agent = Agent(cfg, output_stream=StringIO(), is_tty=False)

    # These should return unchanged
    assert agent._rewrite_user_query("hi") == "hi"
    assert agent._rewrite_user_query("ok") == "ok"
    assert agent._rewrite_user_query("bye") == "bye"
    assert agent._rewrite_user_query("thx") == "thx"  # < 5 chars


def test_query_rewrite_fallback_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that errors fall back to original query."""
    cfg = AgentConfig()

    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    class QueryRewriteChainError:
        def invoke(self, inputs: dict[str, Any]):
            raise RuntimeError("Ollama connection failed")

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "planning": DummyChain(outputs=["none"]),
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": DummyChain(outputs=["NO"]),
            "query_rewrite": QueryRewriteChainError(),
            "search_decision": DummyChain(outputs=["NO_SEARCH"]),
            "response": DummyChain(stream_tokens=["test"]),
            "response_no_search": DummyChain(stream_tokens=["test"]),
        }

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(
        agent_module._embedding_client_mod.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False
    )

    agent = Agent(cfg, output_stream=StringIO(), is_tty=False)

    # Should return original query despite error
    result = agent._rewrite_user_query("test query")
    assert result == "test query"


def test_query_rewrite_fallback_on_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that empty rewrites fall back to original."""
    cfg = AgentConfig()

    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    class QueryRewriteChainEmpty:
        def invoke(self, inputs: dict[str, Any]):
            return ""

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "planning": DummyChain(outputs=["none"]),
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": DummyChain(outputs=["NO"]),
            "query_rewrite": QueryRewriteChainEmpty(),
            "search_decision": DummyChain(outputs=["NO_SEARCH"]),
            "response": DummyChain(stream_tokens=["test"]),
            "response_no_search": DummyChain(stream_tokens=["test"]),
        }

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(
        agent_module._embedding_client_mod.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False
    )

    agent = Agent(cfg, output_stream=StringIO(), is_tty=False)
    result = agent._rewrite_user_query("test query")
    assert result == "test query"


def test_query_rewrite_fallback_on_too_long(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that overly long rewrites fall back to original."""
    cfg = AgentConfig()

    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    class QueryRewriteChainTooLong:
        def invoke(self, inputs: dict[str, Any]):
            return "x" * 5000  # Exceeds MAX_QUERY_LENGTH

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "planning": DummyChain(outputs=["none"]),
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": DummyChain(outputs=["NO"]),
            "query_rewrite": QueryRewriteChainTooLong(),
            "search_decision": DummyChain(outputs=["NO_SEARCH"]),
            "response": DummyChain(stream_tokens=["test"]),
            "response_no_search": DummyChain(stream_tokens=["test"]),
        }

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(
        agent_module._embedding_client_mod.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False
    )

    agent = Agent(cfg, output_stream=StringIO(), is_tty=False)
    result = agent._rewrite_user_query("test query")
    assert result == "test query"


def test_query_rewrite_no_display_when_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that unchanged queries don't show 'Understood as' message."""
    cfg = AgentConfig()
    out = StringIO()

    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    class QueryRewriteChainUnchanged:
        def invoke(self, inputs: dict[str, Any]):
            return inputs["user_question"]  # Return unchanged

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "planning": DummyChain(outputs=["none"]),
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": DummyChain(outputs=["NO"]),
            "query_rewrite": QueryRewriteChainUnchanged(),
            "search_decision": DummyChain(outputs=["NO_SEARCH"]),
            "response": DummyChain(stream_tokens=["test"]),
            "response_no_search": DummyChain(stream_tokens=["test"]),
        }

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(
        agent_module._embedding_client_mod.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False
    )

    agent = Agent(cfg, output_stream=out, is_tty=True)
    agent._rewrite_user_query("test query", write_fn=agent._write)

    # Should not display "Understood as" since query unchanged
    assert "Understood as:" not in out.getvalue()


def test_query_rewrite_integration_with_conversation_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that conversation history is properly used in rewrite context."""
    cfg = AgentConfig()

    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    captured_inputs = {}

    class QueryRewriteChain:
        def invoke(self, inputs: dict[str, Any]):
            captured_inputs.update(inputs)
            return "Elon Musk age"

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "planning": DummyChain(outputs=["none"]),
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": DummyChain(outputs=["NO"]),
            "query_rewrite": QueryRewriteChain(),
            "search_decision": DummyChain(outputs=["NO_SEARCH"]),
            "response": DummyChain(stream_tokens=["test"]),
            "response_no_search": DummyChain(stream_tokens=["test"]),
        }

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(
        agent_module._embedding_client_mod.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False
    )

    agent = Agent(cfg, output_stream=StringIO(), is_tty=False)

    # Add multiple turns to conversation
    agent.conversation.add_turn(
        user_query="Tell me about Tesla", assistant_response="Tesla is an electric vehicle company...", search_used=True
    )
    agent.conversation.add_turn(
        user_query="Who is their CEO?", assistant_response="Elon Musk is the CEO of Tesla.", search_used=True
    )

    result = agent._rewrite_user_query("how old is he")

    # Verify conversation history was included
    assert "Tesla" in captured_inputs["conversation_history"]
    assert "Elon Musk" in captured_inputs["conversation_history"]
    assert result == "Elon Musk age"


def test_query_rewrite_returns_none_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that None return from chain falls back to original."""
    cfg = AgentConfig()

    def fake_build_llms(cfg: AgentConfig):  # noqa: ANN001
        return "robot", "assistant"

    class QueryRewriteChainNone:
        def invoke(self, inputs: dict[str, Any]):
            return None

    def fake_build_chains(llm_robot: Any, llm_assistant: Any):  # noqa: ANN401
        return {
            "planning": DummyChain(outputs=["none"]),
            "result_filter": DummyChain(outputs=["NO"]),
            "query_filter": DummyChain(outputs=["NO"]),
            "query_rewrite": QueryRewriteChainNone(),
            "search_decision": DummyChain(outputs=["NO_SEARCH"]),
            "response": DummyChain(stream_tokens=["test"]),
            "response_no_search": DummyChain(stream_tokens=["test"]),
        }

    monkeypatch.setattr(agent_module._chains, "build_llms", fake_build_llms)
    monkeypatch.setattr(agent_module._chains, "build_chains", fake_build_chains)
    monkeypatch.setattr(
        agent_module._embedding_client_mod.EmbeddingClient, "embed", lambda self, text: [1.0, 0.0], raising=False
    )

    agent = Agent(cfg, output_stream=StringIO(), is_tty=False)
    result = agent._rewrite_user_query("test query")
    assert result == "test query"
