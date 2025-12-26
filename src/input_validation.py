"""Input validation and sanitization utilities for Local AI Agent.

Provides multi-layered defense against prompt injection and malicious input.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from .constants import MAX_QUERY_LENGTH
from .exceptions import InputValidationError


def sanitize_user_input(user_query: str) -> str:
    """Sanitize user input to prevent prompt injection and excessive length.

    Uses multi-layered defense:
    1. Strips control characters
    2. Truncates excessive length
    3. Unicode normalization to prevent obfuscation
    4. Comprehensive pattern matching for injection attempts

    Args:
        user_query: Raw user query

    Returns:
        Sanitized query string

    Raises:
        InputValidationError: If input contains suspicious injection patterns
    """
    # Strip control characters except newlines and tabs
    sanitized = "".join(char for char in user_query if char.isprintable() or char in "\n\t")

    # Truncate to reasonable length
    if len(sanitized) > MAX_QUERY_LENGTH:
        sanitized = sanitized[:MAX_QUERY_LENGTH]
        logging.info("Query truncated to %d characters", MAX_QUERY_LENGTH)

    # Unicode normalization to defeat obfuscation (NFKD decomposes lookalike chars)
    normalized = unicodedata.normalize("NFKD", sanitized).encode("ascii", "ignore").decode("ascii")
    lower_query = normalized.lower()

    # Comprehensive prompt injection patterns with variations
    injection_patterns = [
        # Instruction manipulation
        (
            r"(?:ignore|disregard|forget|skip)\s+(?:all\s+)?(?:previous|prior|above|earlier|past)",
            "instruction override attempt",
        ),
        (
            r"(?:new|different|updated)\s+(?:instructions?|rules?|system|directives?)\s*:?",
            "instruction replacement attempt",
        ),
        # Role manipulation
        (
            r"(?:you\s+are|you're|youre|ur)\s+(?:now\s+)?(?:a|an)?\s*(?:admin|root|system|developer|hacker|god)",
            "identity/role override attempt",
        ),
        (
            r"(?:act\s+as|behave\s+like|pretend\s+to\s+be|roleplay\s+as)\s+(?:a|an)?\s*(?:admin|root|system|developer|hacker|god)",
            "behavior override attempt",
        ),
        (
            r"\b(?:system|admin|root|developer)\s*:(?!\s*$)",
            "privileged role assumption",
        ),
        (
            r"\b(?:developer|god)\s*mode",
            "privileged mode activation",
        ),
        # Meta-prompting
        (
            r"(?:end\s+of|stop)\s+(?:instructions?|prompt|system)",
            "prompt boundary manipulation",
        ),
        (
            r"(?:start|begin)\s+(?:new|fresh)\s+(?:conversation|session)",
            "session reset attempt",
        ),
        # Command injection
        (
            r"(?:execute|run|eval|compile)\s*\(",
            "code execution attempt",
        ),
        # Output manipulation
        (
            r"(?:output|print|write|display)\s+(?:only|just)\s+",
            "output manipulation attempt",
        ),
    ]

    for pattern, description in injection_patterns:
        if re.search(pattern, lower_query, re.IGNORECASE | re.MULTILINE):
            logging.warning(
                "Blocked prompt injection: %s (pattern: %s) in query: %s",
                description,
                pattern,
                normalized[:100],
            )
            raise InputValidationError(
                "Input contains suspicious patterns. Please rephrase naturally without "
                "system directives or meta-instructions."
            )

    return normalized.strip()


__all__ = ["sanitize_user_input"]
