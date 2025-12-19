"""Topic management wrapper helpers extracted from `Agent`.

Provides `update_topics(agent, ...)` which forwards to the agent's
`topic_manager.update_topics` method and returns the selected topic index.
This thin wrapper makes it easy to unit test topic update logic in isolation.
"""

from __future__ import annotations

from typing import Any, List, Set


def update_topics(
    agent: Any,
    selected_topic_index: int | None,
    topic_keywords: Set[str],
    question_keywords: List[str],
    aggregated_results: List[str],
    user_query: str,
    response_text: str,
    question_embedding: List[float] | None,
) -> int | None:
    """Forward to `agent.topic_manager.update_topics` and return the result.

    This wrapper keeps the `Agent` class small and allows targeted tests
    for argument forwarding and return value handling.
    """
    from typing import cast

    # Accept either a list or set for `question_keywords` and normalize to a
    # set before invoking TopicManager for consistent behavior.
    return cast(
        int | None,
        agent.topic_manager.update_topics(
            topics=agent.topics,
            selected_topic_index=selected_topic_index,
            topic_keywords=topic_keywords,
            question_keywords=set(question_keywords),
            aggregated_results=aggregated_results,
            user_query=user_query,
            response_text=response_text,
            question_embedding=question_embedding,
        ),
    )


__all__ = ["update_topics"]
