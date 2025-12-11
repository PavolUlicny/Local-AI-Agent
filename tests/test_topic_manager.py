from __future__ import annotations

import pytest

from src.config import AgentConfig
from src.topic_manager import TopicManager
from src.topic_utils import Topic


class _StubEmbeddingClient:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.0, 1.0]
        self.calls: list[str] = []

    def embed(self, text: str):  # noqa: ANN001 - simple stub
        self.calls.append(text)
        return list(self.vector)


def test_topic_manager_creates_topic_and_updates_embedding() -> None:
    cfg = AgentConfig(max_context_turns=2, embedding_history_decay=0.65)
    manager = TopicManager(cfg, embedding_client=_StubEmbeddingClient(), char_budget=lambda base: base)
    topics: list[Topic] = []

    idx = manager.update_topics(
        topics=topics,
        selected_topic_index=None,
        topic_keywords={"solar"},
        question_keywords={"policy"},
        aggregated_results=["Solar incentives overview"],
        user_query="How are solar policies changing?",
        response_text="New incentives are rolling out in 2025.",
        question_embedding=[1.0, 0.0],
    )

    assert idx == 0
    topic = topics[0]
    assert topic.turns == [("How are solar policies changing?", "New incentives are rolling out in 2025.")]
    assert topic.summary
    assert topic.embedding == pytest.approx([0.65, 0.35])
    assert "solar" in topic.keywords
    assert topic.keywords.intersection({"policy", "policies"})


def test_topic_manager_respects_zero_history_window() -> None:
    cfg = AgentConfig(max_context_turns=0)
    manager = TopicManager(cfg, embedding_client=_StubEmbeddingClient(), char_budget=lambda base: base)
    existing_topic = Topic(keywords={"ai"})
    topics = [existing_topic]

    idx = manager.update_topics(
        topics=topics,
        selected_topic_index=0,
        topic_keywords={"ml"},
        question_keywords={"ai"},
        aggregated_results=[],
        user_query="Any AI updates?",
        response_text="Plenty of updates arrive weekly.",
        question_embedding=None,
    )

    assert idx == 0
    assert topics[0].turns == []  # history cleared when window is zero
    assert topics[0].keywords.issuperset({"ai", "ml"})
