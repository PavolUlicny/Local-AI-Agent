"""Shared retry logic for search clients.

This module consolidates retry handling, backoff logic, and exception handling
that is common between synchronous and asynchronous search clients.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TYPE_CHECKING

from ddgs.exceptions import DDGSException, TimeoutException

if TYPE_CHECKING:  # pragma: no cover
    from src.config import AgentConfig


class SearchRetryContext:
    """Context for retry operations across attempts.

    Tracks attempt number, delay, and configuration for retry logic.
    """

    def __init__(self, cfg: "AgentConfig") -> None:
        self.cfg = cfg
        self.attempt = 0
        self.delay = 0.5
        self.last_exception: Exception | None = None

    def increment_attempt(self) -> None:
        """Move to next retry attempt."""
        self.attempt += 1

    def is_retry_needed(self) -> bool:
        """Check if another retry attempt is available."""
        return self.attempt < self.cfg.search_retries

    def get_delay(self) -> float:
        """Get current delay value."""
        return self.delay

    def update_delay(self, backoff_multiplier: float, max_delay: float) -> None:
        """Update delay for next retry using exponential backoff."""
        self.delay = min(self.delay * backoff_multiplier, max_delay)


def safe_close_client(client: Any) -> None:
    """Safely close DDGS client.

    Args:
        client: DDGS client instance to close
    """
    if client is None:
        return
    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception as exc:
            logging.debug("Client close failed: %s", exc)


def handle_search_exception(
    exc: Exception,
    query: str,
    attempt: int,
    max_retries: int,
    delay: float,
) -> bool:
    """Handle search exceptions and determine if retry should continue.

    Args:
        exc: Exception that occurred
        query: Search query string
        attempt: Current attempt number (1-indexed)
        max_retries: Maximum number of retry attempts
        delay: Current delay before retry

    Returns:
        True if should retry, False if should abort
    """
    if isinstance(exc, TimeoutException):
        logging.warning(
            "Search timeout for '%s' (attempt %s/%s); retrying after %.1fs",
            query,
            attempt,
            max_retries,
            delay,
        )
        logging.debug("Timeout details: %s", exc)
        return True

    elif isinstance(exc, DDGSException):
        logging.warning(
            "DDGS search error for '%s' (attempt %s/%s): %s; retrying after %.1fs",
            query,
            attempt,
            max_retries,
            exc,
            delay,
        )
        return True

    elif isinstance(exc, (ConnectionError, OSError)):
        logging.warning(
            "Network error for '%s' (attempt %s/%s): %s; retrying after %.1fs",
            query,
            attempt,
            max_retries,
            exc,
            delay,
        )
        return True

    else:
        # Unexpected error - don't retry
        logging.error(
            "Unexpected search error for '%s' (attempt %s/%s): %s",
            query,
            attempt,
            max_retries,
            exc,
            exc_info=True,
        )
        return False


def log_final_failure(query: str, max_retries: int) -> None:
    """Log final failure message after all retries exhausted.

    Args:
        query: Search query that failed
        max_retries: Number of retries attempted
    """
    logging.warning("Search failed after %s attempts for '%s'.", max_retries, query)


def should_notify_retry(
    attempt: int,
    max_retries: int,
    notify_fn: Callable[[int, int, float, Exception], None] | None,
    delay: float,
    exception: Exception,
) -> None:
    """Notify about retry attempt if callback is provided.

    Args:
        attempt: Current attempt number (1-indexed)
        max_retries: Maximum number of retries
        notify_fn: Optional callback function
        delay: Delay before retry
        exception: Exception that triggered retry
    """
    if attempt < max_retries and notify_fn is not None:
        notify_fn(attempt, max_retries, delay, exception)


__all__ = [
    "SearchRetryContext",
    "safe_close_client",
    "handle_search_exception",
    "log_final_failure",
    "should_notify_retry",
]
