"""Keyword extraction heuristics and relevance helpers."""

from __future__ import annotations

import re
from typing import Set

_TOKEN_PATTERN = re.compile(r"[\w']+", re.UNICODE)
_STOP_WORDS: Set[str] = {
    "and",
    "about",
    "are",
    "ask",
    "buy",
    "can",
    "for",
    "from",
    "give",
    "going",
    "good",
    "gonna",
    "have",
    "here",
    "into",
    "just",
    "like",
    "more",
    "need",
    "please",
    "really",
    "should",
    "that",
    "than",
    "the",
    "then",
    "there",
    "think",
    "tell",
    "this",
    "those",
    "want",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}
_GENERIC_TOKENS: Set[str] = {"question", "questions", "info"}
_FOLLOWUP_PRONOUNS: Set[str] = {
    "it",
    "its",
    "they",
    "them",
    "their",
    "theirs",
    "this",
    "that",
    "those",
    "these",
    "there",
    "thereof",
}
_FOLLOWUP_HINT_TOKENS: Set[str] = {
    "cost",
    "price",
    "one",
    "same",
    "again",
    "more",
    "details",
    "info",
    "about",
}
_FOLLOWUP_PREFIXES = (
    "how much",
    "how many",
    "what about",
    "how about",
    "and what",
    "and how",
    "tell me more",
    "is it",
    "are they",
    "does it",
    "can it",
)


def extract_keywords(text: str) -> Set[str]:
    if not text:
        return set()
    tokens = _TOKEN_PATTERN.findall(text.lower())
    cleaned = {token.strip("_\"'.!?") for token in tokens}
    return {
        token
        for token in cleaned
        if len(token) > 2
        and token not in _STOP_WORDS
        and token not in _GENERIC_TOKENS
        and not token.isdigit()
        and not (sum(ch.isdigit() for ch in token) / len(token) > 0.6)
    }


def is_relevant(text: str, topic_keywords: Set[str]) -> bool:
    if not topic_keywords:
        return True
    return bool(extract_keywords(text).intersection(topic_keywords))


def looks_like_followup(question: str, keywords: Set[str]) -> bool:
    lowered = question.lower().strip()
    if not lowered:
        return False
    tokens = _TOKEN_PATTERN.findall(lowered)
    if not tokens:
        return False
    pronoun_hits = sum(1 for token in tokens if token in _FOLLOWUP_PRONOUNS)
    referential_hits = sum(1 for token in tokens if token in _FOLLOWUP_HINT_TOKENS)
    if pronoun_hits and len(tokens) <= 6:
        return True
    if pronoun_hits and referential_hits:
        return True
    if pronoun_hits and len(keywords) <= 2:
        return True
    for prefix in _FOLLOWUP_PREFIXES:
        if lowered.startswith(prefix):
            return True
    return False


__all__ = [
    "extract_keywords",
    "is_relevant",
    "looks_like_followup",
]
