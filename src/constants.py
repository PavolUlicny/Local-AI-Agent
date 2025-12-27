"""Centralized constants for Local AI Agent.

This module contains all magic numbers and configuration constants
used throughout the codebase, improving maintainability and consistency.
"""

from __future__ import annotations

from enum import Enum


class ChainName(str, Enum):
    """Type-safe LLM chain identifiers.

    Inherits from str to work as dict keys and in string operations.
    """

    PLANNING = "planning"
    RESULT_FILTER = "result_filter"
    QUERY_FILTER = "query_filter"
    QUERY_REWRITE = "query_rewrite"
    SEARCH_DECISION = "search_decision"
    RESPONSE = "response"
    RESPONSE_NO_SEARCH = "response_no_search"

    def __str__(self) -> str:
        """Return the string value."""
        return self.value


class RebuildKey(str, Enum):
    """Type-safe context rebuild tracking keys.

    Inherits from str to work as dict keys and in string operations.
    """

    SEARCH_DECISION = "search_decision"
    PLANNING = "planning"
    RELEVANCE = "relevance"
    QUERY_FILTER = "query_filter"
    QUERY_REWRITE = "query_rewrite"
    ANSWER = "answer"

    def __str__(self) -> str:
        """Return the string value."""
        return self.value


# Character budget calculations
MIN_CHAR_BUDGET = 1024  # Minimum character budget regardless of context size
CHARS_PER_TOKEN_ESTIMATE = 4  # Rough estimate of characters per LLM token
CONTEXT_SAFETY_MARGIN = 0.8  # Use 80% of context to leave safety margin

# Search orchestration
ITERATION_GUARD_MULTIPLIER = 4  # Multiplier for max iterations guard
MIN_ITERATION_GUARD = 20  # Minimum iteration guard value

# Input validation
MAX_QUERY_LENGTH = 4000  # Maximum user query length (~1000 tokens)
MAX_SINGLE_RESULT_CHARS = 2000  # Maximum characters per search result
MAX_SEARCH_RESULTS_CHARS = 32500  # Maximum total search results characters

# Retry behavior
MAX_REBUILD_RETRIES = 2  # Maximum context rebuild attempts
RETRY_JITTER_MAX = 0.2  # Maximum random jitter for retry delay (seconds)
RETRY_BACKOFF_MULTIPLIER = 1.75  # Exponential backoff multiplier
RETRY_MAX_DELAY = 3.0  # Maximum delay between retries (seconds)

# Keyword extraction
MIN_KEYWORD_LENGTH = 3  # Minimum keyword length (excludes short words)
MAX_DIGIT_RATIO = 0.6  # Maximum ratio of digits in a keyword token

# Ollama defaults
DEFAULT_OLLAMA_PORT = 11434  # Default Ollama HTTP API port
DEFAULT_OLLAMA_HOST = "127.0.0.1"  # Default Ollama host

# Conversation management
DEFAULT_MAX_CONVERSATION_CHARS = 64000  # ~16k tokens, ~90 turns with search
DEFAULT_COMPACT_KEEP_TURNS = 10  # Default turns to keep when compacting
DEFAULT_FORMAT_OVERHEAD = 30  # Extra chars per turn for formatting
MAX_CONVERSATION_TURNS = 200  # Maximum turns to prevent DoS

# Timeouts
DEFAULT_SEARCH_TIMEOUT = 10.0  # Default search timeout (seconds)
DEFAULT_OLLAMA_READY_TIMEOUT = 60.0  # Time to wait for Ollama startup
DEFAULT_OLLAMA_POLL_INTERVAL = 1.0  # Polling interval for Ollama readiness
DEFAULT_EMBEDDING_TIMEOUT = 5.0  # Embedding generation timeout

__all__ = [
    # Enums
    "ChainName",
    "RebuildKey",
    # Numeric constants
    "MIN_CHAR_BUDGET",
    "CHARS_PER_TOKEN_ESTIMATE",
    "CONTEXT_SAFETY_MARGIN",
    "ITERATION_GUARD_MULTIPLIER",
    "MIN_ITERATION_GUARD",
    "MAX_QUERY_LENGTH",
    "MAX_SINGLE_RESULT_CHARS",
    "MAX_SEARCH_RESULTS_CHARS",
    "MAX_REBUILD_RETRIES",
    "RETRY_JITTER_MAX",
    "RETRY_BACKOFF_MULTIPLIER",
    "RETRY_MAX_DELAY",
    "MIN_KEYWORD_LENGTH",
    "MAX_DIGIT_RATIO",
    "DEFAULT_OLLAMA_PORT",
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_MAX_CONVERSATION_CHARS",
    "DEFAULT_COMPACT_KEEP_TURNS",
    "DEFAULT_FORMAT_OVERHEAD",
    "MAX_CONVERSATION_TURNS",
    "DEFAULT_SEARCH_TIMEOUT",
    "DEFAULT_OLLAMA_READY_TIMEOUT",
    "DEFAULT_OLLAMA_POLL_INTERVAL",
    "DEFAULT_EMBEDDING_TIMEOUT",
]
