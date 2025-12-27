"""Protocol definitions for dependency injection and type hints.

This module defines Protocol classes that specify the interfaces used throughout
the codebase, enabling better type checking without tight coupling.
"""

from __future__ import annotations

from typing import Protocol, Any, List, Callable, Dict, Iterator, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .config import AgentConfig
    from .conversation import ConversationManager
    from .constants import ChainName
    from .embedding_client import EmbeddingClient


class LLMChain(Protocol):
    """Protocol for LangChain chain objects.

    Represents a configured LLM chain that can be invoked or streamed.
    """

    def invoke(self, inputs: dict[str, Any]) -> str:
        """Invoke the chain with given inputs and return complete response.

        Args:
            inputs: Dictionary of input variables for the chain

        Returns:
            Complete LLM response as a string
        """
        ...

    def stream(self, inputs: dict[str, Any]) -> Iterator[str]:
        """Stream the chain output token by token.

        Args:
            inputs: Dictionary of input variables for the chain

        Returns:
            Iterator yielding response chunks as strings
        """
        ...


class EmbeddingProvider(Protocol):
    """Protocol for embedding generation services."""

    def embed(self, text: str) -> List[float] | None:
        """Generate embeddings for text.

        Args:
            text: Input text to embed

        Returns:
            List of embedding values, or None if embedding failed
        """
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class SearchProvider(Protocol):
    """Protocol for search client services."""

    def fetch(self, query: str) -> List[dict[str, str]]:
        """Fetch search results for a query.

        Args:
            query: Search query string

        Returns:
            List of search result dictionaries
        """
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class AgentProtocol(Protocol):
    """Protocol defining the Agent interface used by helper modules.

    This allows agent_utils and other modules to type-hint Agent
    without circular imports, while providing strong type safety.
    """

    cfg: "AgentConfig"
    chains: Dict["ChainName", LLMChain]
    embedding_client: "EmbeddingClient"
    conversation: "ConversationManager"
    rebuild_counts: Dict[str, int]
    build_inputs: Callable[..., Dict[str, Any]]

    def _reduce_context_and_rebuild(self, rebuild_key: str, rebuild_label: str) -> None:
        """Reduce context and rebuild LLM chains.

        Args:
            rebuild_key: Key for tracking rebuild attempts
            rebuild_label: Human-readable label for logging
        """
        ...

    def _mark_error(self, message: str) -> str:
        """Mark an error message for display to user.

        Args:
            message: Error message to display

        Returns:
            The formatted error message
        """
        ...

    def _char_budget(self, _base_limit: int) -> int:
        """Calculate character budget based on context size.

        Args:
            _base_limit: Base character limit to use

        Returns:
            Adjusted character budget considering context constraints
        """
        ...


__all__ = [
    "LLMChain",
    "EmbeddingProvider",
    "SearchProvider",
    "AgentProtocol",
]
