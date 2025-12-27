"""Text normalization, truncation, and validation utilities for Local AI Agent."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Pattern

from .constants import (
    MAX_SINGLE_RESULT_CHARS,
    MAX_SEARCH_RESULTS_CHARS,
    MAX_REBUILD_RETRIES,
)

PATTERN_SEARCH_DECISION = re.compile(r"^(SEARCH|NO_SEARCH)$")
PATTERN_YES_NO = re.compile(r"^(YES|NO)$")


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, breaking at word boundary.

    Args:
        text: Text to truncate
        max_chars: Maximum characters allowed

    Returns:
        Truncated text with "..." suffix if truncated
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:")
    return f"{truncated}..."


def truncate_result(text: str, max_chars: int = MAX_SINGLE_RESULT_CHARS) -> str:
    """Truncate search result text (backward compatibility alias for truncate_text).

    Args:
        text: Search result text to truncate
        max_chars: Maximum characters (default: MAX_SINGLE_RESULT_CHARS)

    Returns:
        Truncated text
    """
    return truncate_text(text, max_chars)


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower()).strip(" .,!?:;\"'()[]{}")


def regex_validate(raw: str, pattern: Pattern[str], default: str) -> str:
    if not isinstance(raw, str):
        return default
    candidate = raw.strip().upper()
    return candidate if pattern.fullmatch(candidate) else default


def current_datetime_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def is_context_length_error(message: str) -> bool:
    msg = message.lower()
    indicators = [
        "context length",
        "context window",
        "max context",
        "maximum context",
        "over maximum",
        "too many tokens",
        "token limit",
        "sequence length",
        "ctx length",
        "prompt too long",
        "input too long",
    ]
    return any(ind in msg for ind in indicators)


__all__ = [
    "MAX_REBUILD_RETRIES",
    "MAX_SEARCH_RESULTS_CHARS",
    "MAX_SINGLE_RESULT_CHARS",
    "PATTERN_SEARCH_DECISION",
    "PATTERN_YES_NO",
    "current_datetime_utc",
    "is_context_length_error",
    "normalize_query",
    "regex_validate",
    "truncate_result",
    "truncate_text",
]
