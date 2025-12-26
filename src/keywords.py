"""Keyword extraction heuristics and relevance helpers."""

from __future__ import annotations

import re
from typing import Set

from .constants import MIN_KEYWORD_LENGTH, MAX_DIGIT_RATIO

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


def extract_keywords(text: str) -> Set[str]:
    """Extract meaningful keywords from text for topic matching.

    Filters out stop words, generic terms, short tokens, and digit-heavy strings.

    Args:
        text: Input text to extract keywords from.

    Returns:
        Set of normalized keyword strings.
    """
    if not text:
        return set()
    tokens = _TOKEN_PATTERN.findall(text.lower())
    cleaned = {token.strip("_\"'.!?") for token in tokens}
    return {
        token
        for token in cleaned
        if len(token) >= MIN_KEYWORD_LENGTH
        and token not in _STOP_WORDS
        and token not in _GENERIC_TOKENS
        and not token.isdigit()
        and not (sum(ch.isdigit() for ch in token) / len(token) > MAX_DIGIT_RATIO)
    }


def is_relevant(text: str, topic_keywords: Set[str]) -> bool:
    """Check if text contains any topic keywords (used by search result filtering).

    Args:
        text: Text to check for relevance.
        topic_keywords: Set of keywords to match against.

    Returns:
        True if any keyword found in text, False otherwise.
    """
    if not topic_keywords:
        return False
    keywords = extract_keywords(text)
    return bool(keywords & topic_keywords)


__all__ = [
    "extract_keywords",
    "is_relevant",
]
