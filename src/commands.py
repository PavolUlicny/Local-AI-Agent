"""Command handling for interactive agent sessions.

Provides slash commands for conversation control and system interaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .agent import Agent
    from .conversation import ConversationManager


class CommandHandler:
    """Handles slash commands in interactive sessions.

    Commands:
        /quit, /exit, /q       - Exit the agent
        /clear, /reset, /new   - Clear conversation history
        /compact               - Keep only last 10 turns
        /stats, /info          - Show conversation statistics
        /help, /?              - Show available commands
    """

    def __init__(self, conversation_manager: "ConversationManager", agent: "Agent") -> None:
        """Initialize command handler.

        Args:
            conversation_manager: ConversationManager instance to operate on
            agent: Agent instance for session tracking
        """
        self.conversation = conversation_manager
        self.agent = agent

    def handle(self, user_input: str) -> Tuple[bool, str | None, bool]:
        """Handle a potential command.

        Args:
            user_input: Raw user input to check for commands

        Returns:
            Tuple of (is_command, response_message, should_exit)
            - is_command: True if input was a command
            - response_message: Message to show user (None for silent commands)
            - should_exit: True if the agent should exit
        """
        cmd = user_input.strip().lower()

        # If input doesn't start with /, it's not a command
        if not cmd.startswith("/"):
            return False, None, False

        # Exit commands
        if cmd in ("/quit", "/exit", "/q"):
            return True, "\nGoodbye!\n", True

        # Clear conversation
        if cmd in ("/clear", "/reset", "/new"):
            self.conversation.clear()
            return True, "\nConversation cleared. Starting fresh.", False

        # Compact conversation
        if cmd in ("/compact", "/compress"):
            removed = self.conversation.compact(keep_last_n=10)
            if removed == 0:
                return True, "\nConversation already has 10 or fewer turns.", False
            return True, f"\nCompacted: kept last 10 turns (removed {removed}).", False

        # Show statistics
        if cmd in ("/stats", "/info"):
            stats = self.conversation.get_stats(
                session_start_time=self.agent._session_start_time,
                session_searches=self.agent._session_search_count,
                robot_model=self.agent.cfg.robot_model,
                assistant_model=self.agent.cfg.assistant_model,
            )
            response = self.conversation.format_stats(stats)
            return True, response, False

        # Help
        if cmd in ("/help", "/?"):
            help_text = (
                "\nAvailable Commands:\n"
                "  /quit, /exit, /q       - Exit the agent\n"
                "  /clear, /reset, /new   - Clear conversation history\n"
                "  /compact               - Keep only last 10 turns\n"
                "  /stats, /info          - Show conversation statistics\n"
                "  /help, /?              - Show this help message"
            )
            return True, help_text, False

        # Input starts with / but doesn't match any known command
        return True, f"\nUnknown command: {cmd}\nType /help to see available commands.", False


__all__ = ["CommandHandler"]
