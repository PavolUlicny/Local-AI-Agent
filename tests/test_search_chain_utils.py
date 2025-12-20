from __future__ import annotations

import pytest

from src.config import AgentConfig
from src.exceptions import ResponseError
from src.search_chain_utils import invoke_chain_with_retry, SearchAbort


def test_invoke_chain_with_retry_model_missing_raises():
    """Test that invoke_chain_with_retry raises SearchAbort on model not found error."""

    class BadChain:
        def invoke(self, inputs):
            raise ResponseError("Model Not Found: Robot model not found")

    cfg = AgentConfig()
    rebuild_counts = {"planning": 0}

    with pytest.raises(SearchAbort):
        invoke_chain_with_retry(
            chain=BadChain(),
            inputs={},
            rebuild_key="planning",
            rebuild_label="planning",
            rebuild_counts=rebuild_counts,
            reduce_context_and_rebuild=lambda k, label: None,
            mark_error=lambda m: m,
            raise_on_non_context_error=True,
            cfg=cfg,
        )


def test_invoke_chain_with_retry_context_retry_then_success():
    """Test that invoke_chain_with_retry retries on context length error and succeeds."""
    calls = {"reduced": 0}

    def reduce(key, label):
        calls["reduced"] += 1

    class Chain:
        def __init__(self):
            self.behavior = [ResponseError("Context length exceeded"), "OK"]

        def invoke(self, inputs):
            val = self.behavior.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

    cfg = AgentConfig()
    rebuild_counts = {"planning": 0}

    out, success = invoke_chain_with_retry(
        chain=Chain(),
        inputs={},
        rebuild_key="planning",
        rebuild_label="planning",
        rebuild_counts=rebuild_counts,
        reduce_context_and_rebuild=reduce,
        mark_error=lambda m: m,
        cfg=cfg,
    )

    assert success is True
    assert out == "OK"
    assert calls["reduced"] == 1


def test_invoke_chain_with_retry_non_context_error_behavior():
    """Test invoke_chain_with_retry behavior with non-context errors."""

    class BadChain:
        def invoke(self, inputs):
            raise ResponseError("Some other error")

    cfg = AgentConfig()
    rebuild_counts = {"planning": 0}

    # When raise_on_non_context_error=True should raise SearchAbort
    with pytest.raises(SearchAbort):
        invoke_chain_with_retry(
            chain=BadChain(),
            inputs={},
            rebuild_key="planning",
            rebuild_label="planning",
            rebuild_counts=rebuild_counts,
            reduce_context_and_rebuild=lambda k, label: None,
            mark_error=lambda m: m,
            raise_on_non_context_error=True,
            cfg=cfg,
        )

    # When raise_on_non_context_error=False should return fallback and False
    out, success = invoke_chain_with_retry(
        chain=BadChain(),
        inputs={},
        rebuild_key="planning",
        rebuild_label="planning",
        rebuild_counts=rebuild_counts,
        reduce_context_and_rebuild=lambda k, label: None,
        mark_error=lambda m: m,
        raise_on_non_context_error=False,
        cfg=cfg,
    )
    assert success is False
    assert out == "NO"


def test_invoke_chain_with_retry_returns_success_false_on_generic_exception():
    """Test invoke_chain_with_retry handles generic exceptions gracefully."""

    class BadChain:
        def invoke(self, inputs):
            raise ValueError("Unexpected error")

    cfg = AgentConfig()
    rebuild_counts = {"test": 0}

    out, success = invoke_chain_with_retry(
        chain=BadChain(),
        inputs={},
        rebuild_key="test",
        rebuild_label="test",
        rebuild_counts=rebuild_counts,
        reduce_context_and_rebuild=lambda k, label: None,
        mark_error=lambda m: m,
        fallback_value="FALLBACK",
        cfg=cfg,
    )

    assert out == "FALLBACK"
    assert success is False


def test_invoke_chain_with_retry_max_rebuild_retries():
    """Test invoke_chain_with_retry respects MAX_REBUILD_RETRIES."""
    from src.text_utils import MAX_REBUILD_RETRIES

    calls = {"reduced": 0}

    def reduce(key, label):
        calls["reduced"] += 1

    cfg = AgentConfig()
    # Start at max-1 so we can see one more rebuild
    rebuild_counts = {"test": MAX_REBUILD_RETRIES - 1}

    class FailChain:
        def __init__(self):
            self.call_count = 0

        def invoke(self, inputs):
            self.call_count += 1
            if self.call_count == 1:
                raise ResponseError("Context length exceeded")
            # Second call succeeds
            return "SUCCESS"

    out, success = invoke_chain_with_retry(
        chain=FailChain(),
        inputs={},
        rebuild_key="test",
        rebuild_label="test",
        rebuild_counts=rebuild_counts,
        reduce_context_and_rebuild=reduce,
        mark_error=lambda m: m,
        fallback_value="FALLBACK",
        cfg=cfg,
    )

    assert out == "SUCCESS"
    assert success is True
    assert calls["reduced"] == 1


def test_invoke_chain_with_retry_exceed_max_rebuilds():
    """Test invoke_chain_with_retry returns fallback when max rebuilds exceeded."""
    from src.text_utils import MAX_REBUILD_RETRIES

    calls = {"reduced": 0}

    def reduce(key, label):
        calls["reduced"] += 1

    cfg = AgentConfig()
    # Already at max
    rebuild_counts = {"test": MAX_REBUILD_RETRIES}

    class FailChain:
        def invoke(self, inputs):
            raise ResponseError("Context length exceeded")

    out, success = invoke_chain_with_retry(
        chain=FailChain(),
        inputs={},
        rebuild_key="test",
        rebuild_label="test",
        rebuild_counts=rebuild_counts,
        reduce_context_and_rebuild=reduce,
        mark_error=lambda m: m,
        fallback_value="FALLBACK",
        cfg=cfg,
    )

    assert out == "FALLBACK"
    assert success is False
    assert calls["reduced"] == 0  # No rebuild attempted
