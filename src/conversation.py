"""Conversation history management for Local AI Agent.

Replaces the complex topic-based system with a simple conversation list
that leverages 128k context windows of modern LLMs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple

from .constants import (
    DEFAULT_MAX_CONVERSATION_CHARS,
    DEFAULT_FORMAT_OVERHEAD,
    MAX_CONVERSATION_TURNS,
)

# Format: (user_query, assistant_response, timestamp, search_used)
ConversationTurn = Tuple[str, str, datetime, bool]


@dataclass
class ConversationStats:
    """Statistics about the current conversation."""

    turns: int
    chars: int
    budget: int
    usage_percent: int
    search_turns: int
    oldest_timestamp: datetime | None
    newest_timestamp: datetime | None


class ConversationManager:
    """Manages conversation history with automatic memory management.

    Features:
    - Automatic sliding window based on character budget
    - Manual compaction via commands
    - Full conversation formatting for prompts
    - Statistics tracking

    Design Philosophy:
    With 128k context models, we can keep 90+ conversation turns in memory.
    This eliminates the need for topic tracking, embedding similarity, and
    context classification. The LLM handles continuity naturally.
    """

    def __init__(
        self,
        max_context_chars: int = DEFAULT_MAX_CONVERSATION_CHARS,
        max_turns: int = MAX_CONVERSATION_TURNS,
        *,
        format_overhead: int = DEFAULT_FORMAT_OVERHEAD,
    ) -> None:
        """Initialize conversation manager.

        Args:
            max_context_chars: Maximum characters to keep in conversation history.
                Default 64k chars ≈ 16k tokens ≈ 90 turns with search results.
            max_turns: Maximum number of conversation turns (DoS prevention).
                Default 200 turns.
            format_overhead: Extra characters per turn for "User: " / "Assistant: " formatting.
        """
        self.turns: List[ConversationTurn] = []
        self.max_context_chars = max_context_chars
        self.max_turns = max_turns
        self.format_overhead = format_overhead

    def add_turn(
        self,
        user_query: str,
        assistant_response: str,
        *,
        search_used: bool = False,
    ) -> None:
        """Add a conversation turn and auto-trim if needed.

        Args:
            user_query: User's question or message
            assistant_response: Agent's response
            search_used: Whether web search was used for this turn
        """
        timestamp = datetime.now(timezone.utc)
        self.turns.append((user_query, assistant_response, timestamp, search_used))

        # Enforce max turns limit (DoS prevention)
        while len(self.turns) > self.max_turns:
            self.turns.pop(0)
            logging.debug("Removed turn exceeding max_turns limit (%d)", self.max_turns)

        self._auto_trim()

    def _auto_trim(self) -> None:
        """Remove oldest turns until within character budget.

        This is silent and automatic - the user doesn't need to know.
        With 64k budget, this rarely triggers in normal conversations.

        Always preserves at least 1 turn (the most recent).
        """
        while len(self.turns) > 1 and self._total_chars() > self.max_context_chars:
            removed = self.turns.pop(0)
            logging.debug(
                "Auto-trimmed conversation turn from %s (%d chars)",
                removed[2].isoformat(),
                len(removed[0]) + len(removed[1]),
            )

    def _total_chars(self) -> int:
        """Calculate total characters in conversation including formatting."""
        return sum(
            len(user_query) + len(assistant_response) + self.format_overhead
            for user_query, assistant_response, _, _ in self.turns
        )

    def clear(self) -> None:
        """Clear all conversation history."""
        self.turns.clear()
        logging.info("Conversation history cleared")

    def compact(self, keep_last_n: int = 10) -> int:
        """Keep only the last N turns, removing older ones.

        Args:
            keep_last_n: Number of recent turns to keep

        Returns:
            Number of turns removed
        """
        if len(self.turns) <= keep_last_n:
            return 0

        removed_count = len(self.turns) - keep_last_n
        self.turns = self.turns[-keep_last_n:]
        logging.info("Compacted conversation: kept last %d turns (removed %d)", keep_last_n, removed_count)
        return removed_count

    def format_for_prompt(self, *, include_timestamps: bool = False) -> str:
        """Format conversation history for LLM prompt.

        Args:
            include_timestamps: Whether to include timestamps (useful for debugging)

        Returns:
            Formatted conversation string ready for prompt injection
        """
        if not self.turns:
            return "No prior conversation."

        formatted = []
        for user_query, assistant_response, timestamp, search_used in self.turns:
            if include_timestamps:
                ts_str = timestamp.strftime("%H:%M:%S")
                search_marker = " [search]" if search_used else ""
                formatted.append(f"[{ts_str}{search_marker}] User: {user_query}")
                formatted.append(f"[{ts_str}] Assistant: {assistant_response}")
            else:
                formatted.append(f"User: {user_query}")
                formatted.append(f"Assistant: {assistant_response}")

        return "\n\n".join(formatted)

    def get_stats(self) -> ConversationStats:
        """Get conversation statistics.

        Returns:
            ConversationStats object with current state
        """
        chars = self._total_chars()
        oldest = self.turns[0][2] if self.turns else None
        newest = self.turns[-1][2] if self.turns else None
        search_count = sum(1 for _, _, _, search_used in self.turns if search_used)

        return ConversationStats(
            turns=len(self.turns),
            chars=chars,
            budget=self.max_context_chars,
            usage_percent=int(chars / self.max_context_chars * 100) if self.max_context_chars > 0 else 0,
            search_turns=search_count,
            oldest_timestamp=oldest,
            newest_timestamp=newest,
        )

    def format_stats(self, stats: ConversationStats | None = None) -> str:
        """Format statistics as a human-readable string.

        Args:
            stats: Optional pre-computed stats, otherwise fetches current

        Returns:
            Formatted statistics string
        """
        if stats is None:
            stats = self.get_stats()

        lines = [
            "\nConversation Statistics:",
            f"  Turns: {stats.turns}",
            f"  Characters: {stats.chars:,} / {stats.budget:,}",
            f"  Usage: {stats.usage_percent}%",
        ]

        if stats.oldest_timestamp and stats.newest_timestamp:
            duration = stats.newest_timestamp - stats.oldest_timestamp
            lines.append(f"  Duration: {duration}")

        return "\n".join(lines)


__all__ = ["ConversationManager", "ConversationStats", "ConversationTurn"]
