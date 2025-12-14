"""Topic management utilities for Local AI Agent."""

from __future__ import annotations

from typing import Callable, List, Set, TYPE_CHECKING

from src.keywords import extract_keywords
from src.text_utils import summarize_answer, truncate_text
from src.topic_utils import (
    MAX_TOPICS,
    MAX_TURN_KEYWORD_SOURCE_CHARS,
    Topic,
    blend_embeddings,
    prune_keywords,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.config import AgentConfig
    from src.embedding_client import EmbeddingClient


class TopicManager:
    """Encapsulates topic bookkeeping, keyword tracking, and embeddings."""

    def __init__(
        self,
        cfg: "AgentConfig",
        *,
        embedding_client: "EmbeddingClient",
        char_budget: Callable[[int], int],
    ) -> None:
        self.cfg = cfg
        self._embedding_client = embedding_client
        self._char_budget = char_budget

    def update_topics(
        self,
        *,
        topics: List[Topic],
        selected_topic_index: int | None,
        topic_keywords: Set[str],
        question_keywords: Set[str],
        aggregated_results: List[str],
        user_query: str,
        response_text: str,
        question_embedding: List[float] | None,
    ) -> int:
        selected_topic_index = self._ensure_topic_exists(
            topics,
            selected_topic_index,
            topic_keywords,
            question_embedding,
        )
        selected_topic_index = self._move_topic_to_recent(topics, selected_topic_index)
        selected_topic_index = self._enforce_topic_limit(topics, selected_topic_index)
        topic_entry = topics[selected_topic_index]
        self._update_turn_history(topic_entry, user_query, response_text)
        self._update_keywords(
            topic_entry,
            topic_keywords,
            question_keywords,
            aggregated_results,
            user_query,
            response_text,
        )
        self._update_summary_and_embedding(topic_entry, user_query, response_text)
        prune_keywords(topic_entry)
        return selected_topic_index

    def _ensure_topic_exists(
        self,
        topics: List[Topic],
        selected_topic_index: int | None,
        topic_keywords: Set[str],
        question_embedding: List[float] | None,
    ) -> int:
        if selected_topic_index is not None:
            return selected_topic_index
        initial_embedding = list(question_embedding) if question_embedding else None
        topics.append(Topic(keywords=set(topic_keywords), embedding=initial_embedding))
        return len(topics) - 1

    def _move_topic_to_recent(self, topics: List[Topic], selected_topic_index: int) -> int:
        if selected_topic_index == len(topics) - 1:
            return selected_topic_index
        moved_topic = topics.pop(selected_topic_index)
        topics.append(moved_topic)
        return len(topics) - 1

    def _enforce_topic_limit(self, topics: List[Topic], selected_topic_index: int) -> int:
        while len(topics) > MAX_TOPICS:
            topics.pop(0)
            selected_topic_index = max(0, selected_topic_index - 1)
        return selected_topic_index

    def _update_turn_history(self, topic: Topic, user_query: str, response_text: str) -> None:
        topic.turns.append((user_query, response_text))
        history_window = max(0, self.cfg.max_context_turns) * 2
        if history_window == 0:
            topic.turns = []
        elif len(topic.turns) > history_window:
            topic.turns = topic.turns[-history_window:]

    def _update_keywords(
        self,
        topic: Topic,
        topic_keywords: Set[str],
        question_keywords: Set[str],
        aggregated_results: List[str],
        user_query: str,
        response_text: str,
    ) -> None:
        aggregated_keyword_source = truncate_text(
            " ".join(aggregated_results),
            self._char_budget(MAX_TURN_KEYWORD_SOURCE_CHARS),
        )
        turn_keywords = extract_keywords(" ".join([user_query, response_text, aggregated_keyword_source]))
        if not turn_keywords:
            turn_keywords = set(question_keywords)
        topic.keywords.update(turn_keywords)
        topic.keywords.update(topic_keywords)

    def _update_summary_and_embedding(self, topic: Topic, user_query: str, response_text: str) -> None:
        new_summary = summarize_answer(response_text)
        if new_summary:
            topic.summary = new_summary
        turn_embedding = self._embedding_client.embed(f"User: {user_query}\nAssistant: {response_text}")
        if turn_embedding:
            topic.embedding = blend_embeddings(
                topic.embedding,
                turn_embedding,
                self.cfg.embedding_history_decay,
            )


__all__ = ["TopicManager"]
