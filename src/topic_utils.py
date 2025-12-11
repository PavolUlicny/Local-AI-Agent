"""Topic management primitives and similarity helpers."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Set, Tuple

from .keywords import _extract_keywords, _looks_like_followup
from .text_utils import _PATTERN_CONTEXT, _normalize_context_decision, _regex_validate

TopicTurns = List[Tuple[str, str]]

MAX_TOPICS = 20
MAX_TOPIC_KEYWORDS = 150
MAX_TURN_KEYWORD_SOURCE_CHARS = 17500


@dataclass
class Topic:
    keywords: Set[str] = field(default_factory=set)
    turns: TopicTurns = field(default_factory=list)
    summary: str = ""
    embedding: List[float] | None = None


def _tail_turns(turns: TopicTurns, limit: int) -> TopicTurns:
    if limit is None or limit <= 0:
        return []
    return turns[-limit:]


def _format_turns(turns: TopicTurns, fallback: str) -> str:
    if not turns:
        return fallback
    return "\n\n".join(f"User: {user}\nAssistant: {assistant}" for user, assistant in turns)


def _topic_brief(topic: Topic, max_keywords: int = 10) -> str:
    parts: List[str] = []
    if topic.summary:
        parts.append(f"Summary: {topic.summary}")
    if topic.keywords:
        keywords = ", ".join(sorted(topic.keywords)[:max_keywords])
        parts.append(f"Keywords: {keywords}")
    return "\n".join(parts).strip()


def _cosine_similarity(vec_a: Sequence[float] | None, vec_b: Sequence[float] | None) -> float:
    if not vec_a or not vec_b:
        return 0.0
    length = min(len(vec_a), len(vec_b))
    if length == 0:
        return 0.0
    dot = sum(vec_a[i] * vec_b[i] for i in range(length))
    norm_a = math.sqrt(sum(vec_a[i] ** 2 for i in range(length)))
    norm_b = math.sqrt(sum(vec_b[i] ** 2 for i in range(length)))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    similarity = dot / (norm_a * norm_b)
    return max(-1.0, min(1.0, similarity))


def _blend_embeddings(existing: Optional[List[float]], new: Sequence[float], decay: float) -> List[float]:
    if not new:
        return list(existing or [])
    decay_clamped = min(max(decay, 0.0), 0.9999)
    new_weight = 1.0 - decay_clamped
    prev = list(existing or [])
    size = max(len(prev), len(new))
    blended: List[float] = []
    for idx in range(size):
        prev_val = prev[idx] if idx < len(prev) else 0.0
        new_val = new[idx] if idx < len(new) else 0.0
        blended.append(prev_val * decay_clamped + new_val * new_weight)
    return blended


def _collect_prior_responses(topic: Topic, limit: int = 3, max_chars: int = 1200) -> str:
    if not topic.turns:
        return "No prior answers for this topic."
    snippets: List[str] = []
    budget = max(max_chars // max(1, limit), 200)
    for _, assistant in topic.turns[-limit:]:
        snippet = assistant.strip()
        if len(snippet) > budget:
            truncated = snippet[:budget].rsplit(" ", 1)[0].rstrip(".,;:")
            snippet = f"{truncated}..."
        snippets.append(snippet)
    return "\n\n---\n\n".join(snippets) or "No prior answers for this topic."


def _select_topic(
    context_chain: Any,
    topics: List[Topic],
    question: str,
    base_keywords: Set[str],
    max_context_turns: int,
    current_datetime: str,
    current_year: str,
    current_month: str,
    current_day: str,
    question_embedding: Sequence[float] | None = None,
    embedding_threshold: float = 0.35,
) -> Tuple[Optional[int], TopicTurns, Set[str]]:
    if not topics:
        return None, [], set(base_keywords)
    candidates: List[Tuple[int, float, int]] = []
    for idx, topic in enumerate(topics):
        overlap = len(topic.keywords.intersection(base_keywords))
        similarity = 0.0
        if question_embedding and topic.embedding:
            similarity = _cosine_similarity(question_embedding, topic.embedding)
        if overlap > 0 or similarity >= embedding_threshold:
            candidates.append((overlap, similarity, idx))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    top_candidates = candidates[:3]
    if not top_candidates:
        if _looks_like_followup(question, base_keywords):
            fallback_idx = len(topics) - 1
            fallback_topic = topics[fallback_idx]
            recent_turns = _tail_turns(fallback_topic.turns, max_context_turns)
            return fallback_idx, recent_turns, base_keywords.union(fallback_topic.keywords)
        return None, [], set(base_keywords)
    decisions: List[Tuple[str, int, TopicTurns]] = []
    last_error: Exception | None = None
    for _, _, idx in top_candidates:
        topic = topics[idx]
        recent_turns = _tail_turns(topic.turns, max_context_turns)
        recent_text = _format_turns(recent_turns, "No prior conversation.")
        topic_brief = _topic_brief(topic)
        if topic_brief:
            recent_text = f"Topic brief:\n{topic_brief}\n\nRecent turns:\n{recent_text}"
        try:
            decision_raw = context_chain.invoke(
                {
                    "recent_conversation": recent_text,
                    "new_question": question,
                    "current_datetime": current_datetime,
                    "current_year": current_year,
                    "current_month": current_month,
                    "current_day": current_day,
                }
            )
        except Exception as exc:
            last_error = exc
            logging.warning("Context classification failed for a topic: %s", exc)
            continue
        validated_context = _regex_validate(str(decision_raw), _PATTERN_CONTEXT, "NEW_TOPIC")
        normalized = _normalize_context_decision(validated_context)
        decisions.append((normalized, idx, recent_turns))
        if normalized == "FOLLOW_UP":
            return idx, recent_turns, base_keywords.union(topics[idx].keywords)
    for normalized, idx, recent_turns in decisions:
        if normalized == "EXPAND":
            return idx, recent_turns, base_keywords.union(topics[idx].keywords)
    if last_error is not None:
        raise last_error
    return None, [], set(base_keywords)


def _prune_keywords(topic: Topic, max_keep: int = MAX_TOPIC_KEYWORDS) -> None:
    if len(topic.keywords) <= max_keep:
        return
    freq: dict[str, int] = {}
    for user_text, assistant_text in topic.turns:
        combined = f"{user_text} {assistant_text}"
        for kw in _extract_keywords(combined):
            freq[kw] = freq.get(kw, 0) + 1
    for kw in topic.keywords:
        freq.setdefault(kw, 0)
    ordered = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    keep = {k for k, _ in ordered[:max_keep]}
    topic.keywords.intersection_update(keep)


__all__ = [
    "MAX_TOPICS",
    "MAX_TOPIC_KEYWORDS",
    "MAX_TURN_KEYWORD_SOURCE_CHARS",
    "Topic",
    "TopicTurns",
    "_blend_embeddings",
    "_collect_prior_responses",
    "_cosine_similarity",
    "_format_turns",
    "_prune_keywords",
    "_select_topic",
    "_tail_turns",
    "_topic_brief",
]
