"""Context objects for search orchestration.

This module provides dataclasses that bundle related parameters for search operations,
reducing parameter counts from 20-35 to 5-7 and improving code clarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Set, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.config import AgentConfig


@dataclass(frozen=True)
class SearchContext:
    """Immutable context for search operations.

    Contains all query-level context that doesn't change during search orchestration.
    """

    # DateTime context
    current_datetime: str
    current_year: str
    current_month: str
    current_day: str

    # Query & conversation context
    user_query: str
    conversation_text: str
    prior_responses_text: str

    # Embeddings (immutable references)
    question_embedding: List[float] | None
    topic_embedding_current: List[float] | None


@dataclass
class SearchState:
    """Mutable state during search orchestration.

    Tracks deduplication sets and keywords that are updated as search progresses.
    """

    seen_urls: Set[str] = field(default_factory=set)
    seen_result_hashes: Set[str] = field(default_factory=set)
    seen_query_norms: Set[str] = field(default_factory=set)
    topic_keywords: Set[str] = field(default_factory=set)


@dataclass(frozen=True)
class SearchServices:
    """Dependencies and callbacks for search operations.

    Bundles all external services, configuration, and agent callbacks to eliminate
    callback parameter proliferation.
    """

    # Configuration
    cfg: "AgentConfig"

    # External services
    chains: dict[str, Any]
    embedding_client: Any
    ddg_results: Callable[[str], List[dict[str, str]] | None]

    # Agent callbacks
    inputs_builder: Callable[..., dict[str, Any]]
    reduce_context_and_rebuild: Callable[[str, str], None]
    mark_error: Callable[[str], str]
    context_similarity: Callable[[List[float] | None, List[float] | None, List[float] | None], float]
    char_budget: Callable[[int], int]

    # Mutable state reference (not frozen)
    rebuild_counts: dict[str, int]
