"""Text normalization, truncation, and validation utilities for Local AI Agent."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Pattern

MAX_SINGLE_RESULT_CHARS = 2000
MAX_CONVERSATION_CHARS = 22500
MAX_PRIOR_RESPONSE_CHARS = 13000
MAX_SEARCH_RESULTS_CHARS = 32500
MAX_TOPIC_SUMMARY_CHARS = 400
MAX_PROMPT_RECENT_TURNS = 4
MAX_REBUILD_RETRIES = 2

_PATTERN_SEARCH_DECISION = re.compile(r"^(SEARCH|NO_SEARCH)$")
_PATTERN_YES_NO = re.compile(r"^(YES|NO)$")
_PATTERN_CONTEXT = re.compile(r"^(FOLLOW[_\-\s]?UP|EXPAND|NEW[_\-\s]?TOPIC)$")


def _truncate_result(text: str, max_chars: int = MAX_SINGLE_RESULT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    cut = cut.rsplit(" ", 1)[0].rstrip(".,;:")
    return f"{cut}..."


def _normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower()).strip(" .,!?:;\"'()[]{}")


def _regex_validate(raw: str, pattern: Pattern[str], default: str) -> str:
    if not isinstance(raw, str):
        return default
    candidate = raw.strip().upper()
    return candidate if pattern.fullmatch(candidate) else default


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:")
    return f"{truncated}..."


def _normalize_context_decision(value: str) -> str:
    normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
    if normalized.startswith("FOLLOW"):
        return "FOLLOW_UP"
    if normalized.startswith("EXPAND"):
        return "EXPAND"
    if normalized.startswith("NEW"):
        return "NEW_TOPIC"
    return normalized or "NEW_TOPIC"


def _current_datetime_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _is_context_length_error(message: str) -> bool:
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


def _pick_seed_query(seed_text: str, fallback: str) -> str:
    banned = {
        "none",
        "n/a",
        "na",
        "no",
        "nothing",
        "no new info",
        "no new information",
        "no additional info",
        "no additional information",
    }
    for line in seed_text.splitlines():
        candidate = line.strip().strip("-*\"'").strip()
        if not candidate:
            continue
        if ":" in candidate:
            prefix, remainder = candidate.split(":", 1)
            if prefix.isupper():
                candidate = remainder.strip()
        cleaned = re.sub(r"\s+", " ", candidate)
        lowered = cleaned.lower()
        if not cleaned or lowered in banned:
            continue
        if not any(ch.isalnum() for ch in cleaned):
            continue
        if len(cleaned) < 3:
            continue
        return cleaned
    return fallback


def _summarize_answer(text: str, max_chars: int = MAX_TOPIC_SUMMARY_CHARS) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    summary = " ".join(sentences[:2]).strip() or cleaned
    return _truncate_text(summary, max_chars)


__all__ = [
    "MAX_CONVERSATION_CHARS",
    "MAX_PRIOR_RESPONSE_CHARS",
    "MAX_PROMPT_RECENT_TURNS",
    "MAX_REBUILD_RETRIES",
    "MAX_SEARCH_RESULTS_CHARS",
    "MAX_SINGLE_RESULT_CHARS",
    "MAX_TOPIC_SUMMARY_CHARS",
    "_PATTERN_CONTEXT",
    "_PATTERN_SEARCH_DECISION",
    "_PATTERN_YES_NO",
    "_current_datetime_utc",
    "_is_context_length_error",
    "_normalize_context_decision",
    "_normalize_query",
    "_pick_seed_query",
    "_regex_validate",
    "_summarize_answer",
    "_truncate_result",
    "_truncate_text",
]
