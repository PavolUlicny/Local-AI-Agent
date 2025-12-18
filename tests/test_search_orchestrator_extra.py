from __future__ import annotations

import pytest

from src.search_orchestrator import SearchOrchestrator, SearchAbort
from src.exceptions import ResponseError
from src.config import AgentConfig


class _StubEmbeddingClient:
    def embed(self, text: str):  # noqa: ANN001 - simple stub
        return [0.1, 0.2]


def _inputs_builder(*args, **kwargs):
    return {}


def test_search_orchestrator_raises_on_result_filter_model_missing() -> None:
    cfg = AgentConfig()

    def ddg_results(q: str):
        return [{"title": "T", "link": "http://x", "snippet": "S"}]

    # result_filter raises ResponseError with not found
    class BadChain:
        def invoke(self, inputs):  # noqa: ANN001 - failure
            raise ResponseError("Model Not Found: Robot model not found")

    chains = {"result_filter": BadChain()}

    orch = SearchOrchestrator(
        cfg,
        ddg_results=ddg_results,
        embedding_client=_StubEmbeddingClient(),
        context_similarity=lambda a, b, c: 0.0,
        inputs_builder=_inputs_builder,
        reduce_context_and_rebuild=lambda key, label: None,
        rebuild_counts={"relevance": 0, "planning": 0, "query_filter": 0},
        char_budget=lambda x: x,
        mark_error=lambda m: m,
    )

    with pytest.raises(SearchAbort):
        orch.run(
            chains=chains,
            should_search=True,
            user_query="q",
            current_datetime="d",
            current_year="y",
            current_month="m",
            current_day="dd",
            conversation_text="c",
            prior_responses_text="p",
            question_embedding=None,
            topic_embedding_current=None,
            # provide a non-empty topics set so relevance LLM runs
            topic_keywords={"x"},
            primary_search_query="q",
        )
