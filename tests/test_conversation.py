"""Tests for conversation.ConversationManager."""

from __future__ import annotations

from datetime import datetime


from src.conversation import ConversationManager


def test_conversation_manager_init():
    cm = ConversationManager(max_context_chars=1000)
    assert cm.turns == []
    assert cm.max_context_chars == 1000


def test_add_turn_basic():
    cm = ConversationManager(max_context_chars=10000)
    cm.add_turn("Hello", "Hi there", search_used=False)
    assert len(cm.turns) == 1
    user_query, assistant_response, timestamp, search_used = cm.turns[0]
    assert user_query == "Hello"
    assert assistant_response == "Hi there"
    assert isinstance(timestamp, datetime)
    assert search_used is False


def test_add_turn_with_search():
    cm = ConversationManager(max_context_chars=10000)
    cm.add_turn("What's the weather?", "It's sunny", search_used=True)
    assert len(cm.turns) == 1
    _, _, _, search_used = cm.turns[0]
    assert search_used is True


def test_format_for_prompt_empty():
    cm = ConversationManager()
    result = cm.format_for_prompt()
    assert result == "No prior conversation."


def test_format_for_prompt_single_turn():
    cm = ConversationManager(max_context_chars=10000)
    cm.add_turn("Hello", "Hi there", search_used=False)
    result = cm.format_for_prompt()
    assert "User: Hello" in result
    assert "Assistant: Hi there" in result


def test_format_for_prompt_multiple_turns():
    cm = ConversationManager(max_context_chars=10000)
    cm.add_turn("Hello", "Hi there", search_used=False)
    cm.add_turn("How are you?", "I'm good!", search_used=False)
    result = cm.format_for_prompt()
    assert "User: Hello" in result
    assert "Assistant: Hi there" in result
    assert "User: How are you?" in result
    assert "Assistant: I'm good!" in result


def test_format_for_prompt_with_timestamps():
    cm = ConversationManager(max_context_chars=10000)
    cm.add_turn("Hello", "Hi there", search_used=False)
    result = cm.format_for_prompt(include_timestamps=True)
    assert "User: Hello" in result
    assert "Assistant: Hi there" in result
    # Should contain some timestamp info
    assert "[" in result and "]" in result


def test_auto_trim_removes_oldest():
    # Set very small budget to force trimming
    cm = ConversationManager(max_context_chars=100, format_overhead=10)
    cm.add_turn("First question", "First answer", search_used=False)
    cm.add_turn("Second question", "Second answer", search_used=False)
    cm.add_turn(
        "Third question with a very long string to exceed budget",
        "Third answer with another long string",
        search_used=False,
    )

    # Should have trimmed the oldest turn(s) - verify only 1 turn remains
    assert len(cm.turns) == 1
    # Most recent turn should still be present
    formatted = cm.format_for_prompt()
    assert "Third question" in formatted
    assert "Third answer" in formatted
    # First and second should have been removed
    assert "First" not in formatted
    assert "Second" not in formatted


def test_clear():
    cm = ConversationManager(max_context_chars=10000)
    cm.add_turn("Hello", "Hi", search_used=False)
    cm.add_turn("How are you?", "Good", search_used=False)
    assert len(cm.turns) == 2

    cm.clear()
    assert cm.turns == []
    assert cm.format_for_prompt() == "No prior conversation."


def test_compact():
    cm = ConversationManager(max_context_chars=10000)
    for i in range(20):
        cm.add_turn(f"Question {i}", f"Answer {i}", search_used=False)
    assert len(cm.turns) == 20

    removed = cm.compact(keep_last_n=5)
    assert removed == 15
    assert len(cm.turns) == 5

    # Should keep the most recent 5
    formatted = cm.format_for_prompt()
    assert "Question 19" in formatted
    assert "Question 15" in formatted
    assert "Question 14" not in formatted


def test_compact_with_fewer_turns_than_keep():
    cm = ConversationManager(max_context_chars=10000)
    cm.add_turn("Q1", "A1", search_used=False)
    cm.add_turn("Q2", "A2", search_used=False)

    removed = cm.compact(keep_last_n=10)
    assert removed == 0
    assert len(cm.turns) == 2


def test_get_stats():
    cm = ConversationManager(max_context_chars=1000)
    cm.add_turn("Hello", "Hi", search_used=True)
    cm.add_turn("How are you?", "Good", search_used=False)
    cm.add_turn("What's up?", "Not much", search_used=True)

    stats = cm.get_stats()
    assert stats.turns == 3
    assert stats.chars > 0
    assert stats.budget == 1000
    assert 0 <= stats.usage_percent <= 100
    assert stats.search_turns == 2


def test_get_stats_empty():
    cm = ConversationManager(max_context_chars=1000)
    stats = cm.get_stats()
    assert stats.turns == 0
    assert stats.chars == 0
    assert stats.budget == 1000
    assert stats.usage_percent == 0
    assert stats.search_turns == 0


def test_auto_trim_preserves_at_least_one_turn():
    # Even with tiny budget, should keep at least the most recent turn
    cm = ConversationManager(max_context_chars=1, format_overhead=0)
    cm.add_turn("This is a very long question that exceeds the budget", "And a long answer too", search_used=False)

    assert len(cm.turns) >= 1
    # The most recent turn should be preserved even if over budget
    assert "This is a very long question" in cm.turns[-1][0]


def test_max_turns_enforcement():
    """Test that max_turns limit is enforced (regression test for infinite loop bug)."""
    # Create manager with max_turns limit
    cm = ConversationManager(max_context_chars=100000, max_turns=5)

    # Add more turns than the limit
    for i in range(10):
        cm.add_turn(f"Question {i}", f"Answer {i}", search_used=False)

    # Should never exceed max_turns
    assert len(cm.turns) == 5

    # Should keep the most recent 5 turns
    formatted = cm.format_for_prompt()
    assert "Question 9" in formatted
    assert "Question 5" in formatted
    assert "Question 4" not in formatted
    assert "Question 0" not in formatted
