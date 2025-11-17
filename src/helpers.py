from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse

# Regex/tokenization and stopwords
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
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

# Patterns used for strict validation of LLM outputs
_PATTERN_SEARCH_DECISION = re.compile(r"^(SEARCH|NO_SEARCH)$")
_PATTERN_YES_NO = re.compile(r"^(YES|NO)$")
_PATTERN_CONTEXT = re.compile(r"^(FOLLOW_UP|EXPAND|NEW_TOPIC)$")

# Truncation and size limits
MAX_SINGLE_RESULT_CHARS = 2000
MAX_CONVERSATION_CHARS = 22500
MAX_PRIOR_RESPONSE_CHARS = 13000
MAX_SEARCH_RESULTS_CHARS = 32500
MAX_TURN_KEYWORD_SOURCE_CHARS = 17500
MAX_TOPICS = 20
MAX_TOPIC_KEYWORDS = 150
MAX_REBUILD_RETRIES = 2


def _truncate_result(text: str, max_chars: int = MAX_SINGLE_RESULT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    cut = cut.rsplit(" ", 1)[0].rstrip(".,;:")
    return f"{cut}..."


def _normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower()).strip(" .,!?:;\"'()[]{}")


def _regex_validate(raw: str, pattern: re.Pattern, default: str) -> str:
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


def _canonicalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        raw = url.strip()
        # Handle protocol-relative URLs like //example.com/path
        if raw.startswith("//"):
            raw = "http:" + raw
        parsed = urlparse(raw)
        # Handle schemeless URLs like example.com/path by prefixing http://
        if not parsed.netloc and parsed.path:
            first_segment = parsed.path.split("/", 1)[0]
            looks_like_host_port = bool(re.match(r"^[A-Za-z0-9._-]+:\d+$", first_segment))
            is_localhost = first_segment.startswith("localhost")
            is_ipv4 = bool(re.match(r"^\d+\.\d+\.\d+\.\d+$", first_segment))
            if "://" not in raw and ("." in first_segment or looks_like_host_port or is_localhost or is_ipv4):
                parsed = urlparse("http://" + raw)

        scheme = (parsed.scheme or "http").lower()
        netloc = (parsed.netloc or "").lower()
        path = parsed.path or ""

        # strip www.
        if netloc.startswith("www."):
            netloc = netloc[4:]
        # drop default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        if netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]
        # normalize path: remove trailing slash except root
        if path.endswith("/") and path != "/":
            path = path.rstrip("/")

        # Preserve query for dedup granularity, drop params/fragment
        return urlunparse((scheme, netloc, path, "", parsed.query, ""))
    except Exception:
        # Safer fallback: preserve original case when we cannot parse
        return url.strip().rstrip("/")


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


TopicTurns = List[Tuple[str, str]]


@dataclass
class Topic:
    keywords: Set[str] = field(default_factory=set)
    turns: TopicTurns = field(default_factory=list)


def _extract_keywords(text: str) -> Set[str]:
    if not text:
        return set()
    tokens = _TOKEN_PATTERN.findall(text.lower())
    cleaned = {token.strip("\"'.!?") for token in tokens}
    return {
        token
        for token in cleaned
        if len(token) > 2
        and token not in _STOP_WORDS
        and token not in _GENERIC_TOKENS
        and not token.isdigit()
        and not (sum(ch.isdigit() for ch in token) / len(token) > 0.6)
    }


def _is_relevant(text: str, topic_keywords: Set[str]) -> bool:
    if not topic_keywords:
        return True
    return bool(_extract_keywords(text).intersection(topic_keywords))


def _format_turns(turns: TopicTurns, fallback: str) -> str:
    if not turns:
        return fallback
    return "\n\n".join(f"User: {user}\nAssistant: {assistant}" for user, assistant in turns)


def _collect_prior_responses(
    topic: Topic,
    limit: int = 3,
    max_chars: int = 1200,
) -> str:
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
) -> Tuple[Optional[int], TopicTurns, Set[str]]:
    if not topics:
        return None, [], set(base_keywords)

    scored = sorted(
        ((len(topic.keywords.intersection(base_keywords)), idx) for idx, topic in enumerate(topics)),
        key=lambda x: (-x[0], x[1]),
    )
    top_candidates = [item for item in scored if item[0] > 0][:3]

    if not top_candidates:
        return None, [], set(base_keywords)

    decisions: List[Tuple[str, int, TopicTurns]] = []
    for _, idx in top_candidates:
        topic = topics[idx]
        recent_turns = topic.turns[-max_context_turns:]
        try:
            decision_raw = context_chain.invoke(
                {
                    "recent_conversation": _format_turns(recent_turns, "No prior conversation."),
                    "new_question": question,
                    "current_datetime": current_datetime,
                    "current_year": current_year,
                    "current_month": current_month,
                    "current_day": current_day,
                }
            )
        except Exception:
            # If a single candidate evaluation fails, continue evaluating other
            # candidates instead of aborting the entire selection cycle.
            continue
        validated_context = _regex_validate(str(decision_raw), _PATTERN_CONTEXT, "NEW_TOPIC")
        normalized = _normalize_context_decision(validated_context)
        decisions.append((normalized, idx, recent_turns))
        if normalized == "FOLLOW_UP":
            return idx, recent_turns, base_keywords.union(topics[idx].keywords)

    for normalized, idx, recent_turns in decisions:
        if normalized == "EXPAND":
            return idx, recent_turns, base_keywords.union(topics[idx].keywords)

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
