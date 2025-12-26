"""Context objects for search orchestration.

This module provides dataclasses that bundle related parameters for search operations,
reducing parameter counts from 20-35 to 5-7 and improving code clarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Set, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.config import AgentConfig
    from src.constants import ChainName, RebuildKey
    from src.protocols import LLMChain, EmbeddingProvider


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

    This is a dependency injection container that bundles all external services,
    configuration, and agent callbacks. It eliminates callback parameter proliferation
    by reducing function signatures from 20+ parameters to 3-4.

    The services container enables clean separation between search orchestration logic
    and external dependencies like LLMs, embeddings, and search clients. All services
    are frozen to prevent accidental mutation during search operations.

    Attributes:
        cfg: Agent configuration with LLM parameters and search settings (thresholds,
            retry limits, max rounds, etc.)
        chains: Dictionary of LangChain chains by ChainName enum. Expected keys include
            ChainName.SEED, ChainName.PLANNING, ChainName.QUERY_FILTER,
            ChainName.RESULT_FILTER, ChainName.RESPONSE, and ChainName.RESPONSE_NO_SEARCH
        embedding_client: Client for generating text embeddings for similarity checks,
            used for query/result similarity filtering before LLM validation
        ddg_results: Callback to fetch search results from DuckDuckGo
        inputs_builder: Callback to build prompt inputs with datetime and context
        reduce_context_and_rebuild: Callback to rebuild LLM with smaller context window
            when context-length errors occur
        mark_error: Callback to record error messages for display to user
        context_similarity: Callback to calculate cosine similarity between embeddings
        char_budget: Callback to calculate character budget for text truncation
        rebuild_counts: Mutable dict tracking context rebuild attempts by stage
            (e.g., {"relevance": 1, "planning": 0})
    """

    # Configuration
    cfg: "AgentConfig"

    # External services
    chains: Dict["ChainName" | str, "LLMChain"]  # Allow str for flexibility
    embedding_client: "EmbeddingProvider"
    ddg_results: Callable[[str], List[dict[str, str]] | None]

    # Agent callbacks
    inputs_builder: Callable[..., dict[str, Any]]
    reduce_context_and_rebuild: Callable[["RebuildKey" | str, str], None]
    mark_error: Callable[[str], str]
    context_similarity: Callable[[List[float] | None, List[float] | None, List[float] | None], float]
    char_budget: Callable[[int], int]

    # Mutable state reference (not frozen)
    rebuild_counts: Dict["RebuildKey" | str, int]  # Allow str for flexibility
