"""Tests for commands.CommandHandler."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from src.commands import CommandHandler
from src.conversation import ConversationManager


@pytest.fixture
def handler():
    cm = ConversationManager(max_context_chars=10000)

    # Create mock agent with required attributes
    mock_agent = Mock()
    mock_agent._session_start_time = datetime.now(timezone.utc)
    mock_agent._session_search_count = 0
    mock_agent.cfg.robot_model = "test-robot:1b"
    mock_agent.cfg.assistant_model = "test-assistant:3b"

    return CommandHandler(cm, mock_agent)


def test_quit_command(handler):
    is_command, response, should_exit = handler.handle("/quit")
    assert is_command is True
    assert "Goodbye" in response
    assert should_exit is True


def test_exit_command(handler):
    is_command, response, should_exit = handler.handle("/exit")
    assert is_command is True
    assert "Goodbye" in response
    assert should_exit is True


def test_q_command(handler):
    is_command, response, should_exit = handler.handle("/q")
    assert is_command is True
    assert "Goodbye" in response
    assert should_exit is True


def test_clear_command(handler):
    handler.conversation.add_turn("Hello", "Hi", search_used=False)
    assert len(handler.conversation.turns) == 1

    is_command, response, should_exit = handler.handle("/clear")
    assert is_command is True
    assert "cleared" in response.lower()
    assert should_exit is False
    assert len(handler.conversation.turns) == 0


def test_reset_command(handler):
    handler.conversation.add_turn("Hello", "Hi", search_used=False)
    is_command, response, should_exit = handler.handle("/reset")
    assert is_command is True
    assert "cleared" in response.lower()
    assert should_exit is False
    assert len(handler.conversation.turns) == 0


def test_new_command(handler):
    handler.conversation.add_turn("Hello", "Hi", search_used=False)
    is_command, response, should_exit = handler.handle("/new")
    assert is_command is True
    assert should_exit is False
    assert len(handler.conversation.turns) == 0


def test_compact_command(handler):
    for i in range(20):
        handler.conversation.add_turn(f"Q{i}", f"A{i}", search_used=False)
    assert len(handler.conversation.turns) == 20

    is_command, response, should_exit = handler.handle("/compact")
    assert is_command is True
    assert "10" in response  # Default keep_last_n is 10
    assert "removed" in response.lower()
    assert should_exit is False
    assert len(handler.conversation.turns) == 10


def test_compress_command(handler):
    for i in range(15):
        handler.conversation.add_turn(f"Q{i}", f"A{i}", search_used=False)

    is_command, response, should_exit = handler.handle("/compress")
    assert is_command is True
    assert should_exit is False
    assert len(handler.conversation.turns) == 10


def test_stats_command(handler):
    handler.conversation.add_turn("Hello", "Hi", search_used=True)
    handler.conversation.add_turn("How are you?", "Good", search_used=False)

    # Update mock agent's session search count
    handler.agent._session_search_count = 3

    is_command, response, should_exit = handler.handle("/stats")
    assert is_command is True
    assert "2" in response  # 2 turns (questions)
    assert "1" in response  # 1 search in current conversation
    assert "3" in response  # 3 total session searches
    assert "Session Statistics:" in response
    assert "Duration:" in response
    assert "Models:" in response
    assert "test-robot:1b" in response
    assert "test-assistant:3b" in response
    assert should_exit is False


def test_help_command(handler):
    is_command, response, should_exit = handler.handle("/help")
    assert is_command is True
    assert "/quit" in response
    assert "/clear" in response
    assert "/compact" in response
    assert "/stats" in response
    assert should_exit is False


def test_non_command(handler):
    is_command, response, should_exit = handler.handle("What is the weather?")
    assert is_command is False
    assert response is None
    assert should_exit is False


def test_command_case_insensitive(handler):
    is_command, response, should_exit = handler.handle("/QUIT")
    assert is_command is True
    assert should_exit is True


def test_command_with_whitespace(handler):
    is_command, response, should_exit = handler.handle("  /quit  ")
    assert is_command is True
    assert should_exit is True


def test_partial_command_shows_unknown(handler):
    is_command, response, should_exit = handler.handle("/qui")
    assert is_command is True
    assert "Unknown command" in response
    assert "/help" in response
    assert should_exit is False


def test_command_in_middle_not_recognized(handler):
    is_command, response, should_exit = handler.handle("Please /quit now")
    assert is_command is False
    assert response is None
    assert should_exit is False


def test_typo_command_shows_unknown(handler):
    is_command, response, should_exit = handler.handle("/sttas")
    assert is_command is True
    assert "Unknown command: /sttas" in response
    assert "/help" in response
    assert should_exit is False


def test_unknown_command_with_whitespace(handler):
    is_command, response, should_exit = handler.handle("  /notacommand  ")
    assert is_command is True
    assert "Unknown command" in response
    assert should_exit is False
