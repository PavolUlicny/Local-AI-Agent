"""Parallel search orchestration with backward-compatible synchronous interface."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from src.search_client_async import AsyncSearchClient
from src.search_processing_async import process_search_round_async

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.search_context import SearchContext, SearchServices, SearchState
    from src.config import AgentConfig


def process_queries_parallel(
    queries: list[str],
    context: "SearchContext",
    state: "SearchState",
    services: "SearchServices",
) -> list[str]:
    """Execute multiple search queries in parallel (synchronous interface).

    This function provides a synchronous interface to async parallel search execution.
    It's a drop-in replacement for sequential query processing.

    Args:
        queries: List of search queries to execute in parallel (max determined by config)
        context: Immutable search context
        state: Mutable search state
        services: Services bundle

    Returns:
        List of accepted result texts from all queries
    """
    # Limit concurrent queries based on config
    max_concurrent = services.cfg.max_concurrent_queries
    queries_to_process = queries[:max_concurrent]

    if len(queries_to_process) == 0:
        return []

    logging.debug(f"Processing {len(queries_to_process)} queries in parallel (max: {max_concurrent})")

    # Create async client with same configuration as sync client
    # Access normalizer from Agent instance via __self__
    normalizer_fn = getattr(services.ddg_results, "__self__", None)
    if normalizer_fn and hasattr(normalizer_fn, "_normalize_search_result"):
        normalizer = normalizer_fn._normalize_search_result
    else:
        # Fallback: ddg_results might be a bound method, use identity function
        def identity_normalizer(x: dict) -> dict[str, str] | None:
            return None

        normalizer = identity_normalizer

    async_client = AsyncSearchClient(
        services.cfg,
        normalizer=normalizer,
        notify_retry=None,  # Optional retry notification
    )

    # Run async code in a new event loop
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running (shouldn't happen in our architecture),
            # create a new loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    process_search_round_async(queries_to_process, context, state, services, async_client)
                )
                return future.result()
        else:
            # Loop exists but not running, use it
            return loop.run_until_complete(
                process_search_round_async(queries_to_process, context, state, services, async_client)
            )
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(
            process_search_round_async(queries_to_process, context, state, services, async_client)
        )


def should_use_parallel_search(queries: list[str], cfg: "AgentConfig") -> bool:
    """Determine if parallel search should be used.

    Parallel search is beneficial when:
    1. We have multiple queries to process
    2. max_concurrent_queries > 1

    Args:
        queries: List of queries to process
        cfg: Agent configuration

    Returns:
        True if parallel search should be used
    """
    return len(queries) > 1 and cfg.max_concurrent_queries > 1


__all__ = ["process_queries_parallel", "should_use_parallel_search"]
