"""Tests for constants module to ensure values are sane and documented."""

from __future__ import annotations

from src import constants


class TestConstants:
    """Test that constants have reasonable values."""

    def test_char_budget_constants(self):
        """Character budget constants should be reasonable."""
        assert constants.MIN_CHAR_BUDGET == 1024
        assert constants.CHARS_PER_TOKEN_ESTIMATE == 4
        assert constants.CONTEXT_SAFETY_MARGIN == 0.8
        assert 0 < constants.CONTEXT_SAFETY_MARGIN < 1

    def test_search_orchestration_constants(self):
        """Search orchestration constants should be reasonable."""
        assert constants.ITERATION_GUARD_MULTIPLIER == 4
        assert constants.MIN_ITERATION_GUARD == 20
        assert constants.ITERATION_GUARD_MULTIPLIER > 0
        assert constants.MIN_ITERATION_GUARD > 0

    def test_input_validation_constants(self):
        """Input validation constants should be reasonable."""
        assert constants.MAX_QUERY_LENGTH == 4000
        assert constants.MAX_SINGLE_RESULT_CHARS == 2000
        assert constants.MAX_SEARCH_RESULTS_CHARS == 32500
        assert constants.MAX_QUERY_LENGTH > 0
        assert constants.MAX_SINGLE_RESULT_CHARS > 0
        assert constants.MAX_SEARCH_RESULTS_CHARS > constants.MAX_SINGLE_RESULT_CHARS

    def test_retry_behavior_constants(self):
        """Retry behavior constants should be reasonable."""
        assert constants.MAX_REBUILD_RETRIES == 2
        assert constants.RETRY_JITTER_MAX == 0.2
        assert constants.RETRY_BACKOFF_MULTIPLIER == 1.75
        assert constants.RETRY_MAX_DELAY == 3.0
        assert constants.MAX_REBUILD_RETRIES >= 0
        assert constants.RETRY_JITTER_MAX >= 0
        assert constants.RETRY_BACKOFF_MULTIPLIER > 1
        assert constants.RETRY_MAX_DELAY > 0

    def test_keyword_extraction_constants(self):
        """Keyword extraction constants should be reasonable."""
        assert constants.MIN_KEYWORD_LENGTH == 3
        assert constants.MAX_DIGIT_RATIO == 0.6
        assert constants.MIN_KEYWORD_LENGTH > 0
        assert 0 < constants.MAX_DIGIT_RATIO < 1

    def test_ollama_defaults(self):
        """Ollama defaults should be reasonable."""
        assert constants.DEFAULT_OLLAMA_PORT == 11434
        assert constants.DEFAULT_OLLAMA_HOST == "127.0.0.1"
        assert 1024 <= constants.DEFAULT_OLLAMA_PORT <= 65535

    def test_conversation_management_constants(self):
        """Conversation management constants should be reasonable."""
        assert constants.DEFAULT_MAX_CONVERSATION_CHARS == 64000
        assert constants.DEFAULT_COMPACT_KEEP_TURNS == 10
        assert constants.DEFAULT_FORMAT_OVERHEAD == 30
        assert constants.MAX_CONVERSATION_TURNS == 200
        assert constants.DEFAULT_MAX_CONVERSATION_CHARS >= 1024
        assert constants.DEFAULT_COMPACT_KEEP_TURNS > 0
        assert constants.DEFAULT_FORMAT_OVERHEAD > 0
        assert constants.MAX_CONVERSATION_TURNS > 0

    def test_timeout_constants(self):
        """Timeout constants should be reasonable."""
        assert constants.DEFAULT_SEARCH_TIMEOUT == 10.0
        assert constants.DEFAULT_OLLAMA_READY_TIMEOUT == 60.0
        assert constants.DEFAULT_OLLAMA_POLL_INTERVAL == 1.0
        assert constants.DEFAULT_EMBEDDING_TIMEOUT == 5.0
        assert constants.DEFAULT_SEARCH_TIMEOUT > 0
        assert constants.DEFAULT_OLLAMA_READY_TIMEOUT > 0
        assert constants.DEFAULT_OLLAMA_POLL_INTERVAL > 0
        assert constants.DEFAULT_EMBEDDING_TIMEOUT > 0

    def test_all_exports_exist(self):
        """All exported constants should exist in __all__."""
        for name in constants.__all__:
            assert hasattr(constants, name), f"Exported constant {name} not found"

    def test_no_unexpected_exports(self):
        """Only constants should be exported, not private vars."""
        for name in constants.__all__:
            assert not name.startswith("_"), f"Private variable {name} should not be exported"


class TestChainNameEnum:
    """Tests for ChainName type-safe enum."""

    def test_chain_name_has_all_expected_values(self):
        """ChainName should define all chain types."""
        from src.constants import ChainName

        assert ChainName.PLANNING.value == "planning"
        assert ChainName.RESULT_FILTER.value == "result_filter"
        assert ChainName.QUERY_FILTER.value == "query_filter"
        assert ChainName.QUERY_REWRITE.value == "query_rewrite"
        assert ChainName.SEARCH_DECISION.value == "search_decision"
        assert ChainName.RESPONSE.value == "response"
        assert ChainName.RESPONSE_NO_SEARCH.value == "response_no_search"

    def test_chain_name_str_compatible(self):
        """ChainName should be str-compatible for dict keys."""
        from src.constants import ChainName

        # Can be used as dict keys
        chains = {ChainName.PLANNING: "test"}
        assert chains[ChainName.PLANNING] == "test"

        # Can be compared to strings
        assert str(ChainName.PLANNING) == "planning"

    def test_chain_name_backward_compatible(self):
        """ChainName should work with existing string-based code."""
        from src.constants import ChainName

        # Can access dict with string if key is string
        chains = {"planning": "value"}
        assert chains["planning"] == "value"

        # Can create from string
        assert ChainName("planning") == ChainName.PLANNING


class TestRebuildKeyEnum:
    """Tests for RebuildKey type-safe enum."""

    def test_rebuild_key_has_all_expected_values(self):
        """RebuildKey should define all rebuild tracking keys."""
        from src.constants import RebuildKey

        assert RebuildKey.SEARCH_DECISION.value == "search_decision"
        assert RebuildKey.PLANNING.value == "planning"
        assert RebuildKey.RELEVANCE.value == "relevance"
        assert RebuildKey.QUERY_FILTER.value == "query_filter"
        assert RebuildKey.QUERY_REWRITE.value == "query_rewrite"
        assert RebuildKey.ANSWER.value == "answer"

    def test_rebuild_key_str_compatible(self):
        """RebuildKey should be str-compatible for dict keys."""
        from src.constants import RebuildKey

        # Can be used as dict keys
        counts = {RebuildKey.ANSWER: 0}
        assert counts[RebuildKey.ANSWER] == 0

        # Can be compared to strings
        assert str(RebuildKey.ANSWER) == "answer"

    def test_rebuild_key_backward_compatible(self):
        """RebuildKey should work with existing string-based code."""
        from src.constants import RebuildKey

        # Can access dict with string if key is string
        counts = {"answer": 1}
        assert counts["answer"] == 1

        # Can create from string
        assert RebuildKey("answer") == RebuildKey.ANSWER
