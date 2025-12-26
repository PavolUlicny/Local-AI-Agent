"""Local AI Agent - Conversational AI with web search capabilities.

This package provides a self-steering local LLM assistant that can optionally
enrich answers with iterative, relevance-filtered web searches while tracking
multi-turn conversational context.
"""

from .agent import Agent
from .config import AgentConfig
from .conversation import ConversationManager, ConversationStats
from .exceptions import (
    AgentError,
    ConfigurationError,
    InputValidationError,
    SearchError,
    SearchAbort,
)

__version__ = "1.0.0"

__all__ = [
    # Core classes
    "Agent",
    "AgentConfig",
    "ConversationManager",
    "ConversationStats",
    # Exceptions
    "AgentError",
    "ConfigurationError",
    "InputValidationError",
    "SearchError",
    "SearchAbort",
]
