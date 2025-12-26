"""Comprehensive tests for AgentConfig validation.

Tests all validation branches in config.__post_init__ to achieve 100% coverage.
"""

from __future__ import annotations

import pytest
from src.config import AgentConfig
from src.exceptions import ConfigurationError


class TestConfigValidation:
    """Test all validation branches in AgentConfig.__post_init__."""

    def test_valid_config_passes(self):
        """Valid configuration should not raise errors."""
        cfg = AgentConfig()
        assert cfg is not None

    def test_max_rounds_too_low(self):
        """max_rounds < 1 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="max_rounds must be >= 1"):
            AgentConfig(max_rounds=0)

        with pytest.raises(ConfigurationError, match="max_rounds must be >= 1"):
            AgentConfig(max_rounds=-1)

    def test_max_conversation_chars_too_low(self):
        """max_conversation_chars < 1024 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="max_conversation_chars must be >= 1024"):
            AgentConfig(max_conversation_chars=1023)

        with pytest.raises(ConfigurationError, match="max_conversation_chars must be >= 1024"):
            AgentConfig(max_conversation_chars=0)

    def test_compact_keep_turns_too_low(self):
        """compact_keep_turns < 1 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="compact_keep_turns must be >= 1"):
            AgentConfig(compact_keep_turns=0)

    def test_max_followup_suggestions_negative(self):
        """max_followup_suggestions < 0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="max_followup_suggestions must be >= 0"):
            AgentConfig(max_followup_suggestions=-1)

    def test_max_fill_attempts_negative(self):
        """max_fill_attempts < 0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="max_fill_attempts must be >= 0"):
            AgentConfig(max_fill_attempts=-1)

    def test_max_relevance_llm_checks_negative(self):
        """max_relevance_llm_checks < 0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="max_relevance_llm_checks must be >= 0"):
            AgentConfig(max_relevance_llm_checks=-1)

    def test_assistant_num_ctx_too_low(self):
        """assistant_num_ctx < 512 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="assistant_num_ctx must be >= 512"):
            AgentConfig(assistant_num_ctx=511)

    def test_robot_num_ctx_too_low(self):
        """robot_num_ctx < 512 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="robot_num_ctx must be >= 512"):
            AgentConfig(robot_num_ctx=511)

    def test_assistant_num_predict_too_low(self):
        """assistant_num_predict < 1 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="assistant_num_predict must be >= 1"):
            AgentConfig(assistant_num_predict=0)

    def test_robot_num_predict_too_low(self):
        """robot_num_predict < 1 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="robot_num_predict must be >= 1"):
            AgentConfig(robot_num_predict=0)

    def test_robot_temp_too_low(self):
        """robot_temp < 0.0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="robot_temp must be in"):
            AgentConfig(robot_temp=-0.1)

    def test_robot_temp_too_high(self):
        """robot_temp > 2.0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="robot_temp must be in"):
            AgentConfig(robot_temp=2.1)

    def test_assistant_temp_too_low(self):
        """assistant_temp < 0.0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="assistant_temp must be in"):
            AgentConfig(assistant_temp=-0.1)

    def test_assistant_temp_too_high(self):
        """assistant_temp > 2.0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="assistant_temp must be in"):
            AgentConfig(assistant_temp=2.1)

    def test_robot_top_p_too_low(self):
        """robot_top_p < 0.0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="robot_top_p must be in"):
            AgentConfig(robot_top_p=-0.1)

    def test_robot_top_p_too_high(self):
        """robot_top_p > 1.0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="robot_top_p must be in"):
            AgentConfig(robot_top_p=1.1)

    def test_assistant_top_p_too_low(self):
        """assistant_top_p < 0.0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="assistant_top_p must be in"):
            AgentConfig(assistant_top_p=-0.1)

    def test_assistant_top_p_too_high(self):
        """assistant_top_p > 1.0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="assistant_top_p must be in"):
            AgentConfig(assistant_top_p=1.1)

    def test_embedding_similarity_threshold_too_low(self):
        """embedding_similarity_threshold < 0.0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="embedding_similarity_threshold must be in"):
            AgentConfig(embedding_similarity_threshold=-0.1)

    def test_embedding_similarity_threshold_too_high(self):
        """embedding_similarity_threshold > 1.0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="embedding_similarity_threshold must be in"):
            AgentConfig(embedding_similarity_threshold=1.1)

    def test_embedding_result_similarity_threshold_too_low(self):
        """embedding_result_similarity_threshold < 0.0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="embedding_result_similarity_threshold must be in"):
            AgentConfig(embedding_result_similarity_threshold=-0.1)

    def test_embedding_result_similarity_threshold_too_high(self):
        """embedding_result_similarity_threshold > 1.0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="embedding_result_similarity_threshold must be in"):
            AgentConfig(embedding_result_similarity_threshold=1.1)

    def test_embedding_query_similarity_threshold_too_low(self):
        """embedding_query_similarity_threshold < 0.0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="embedding_query_similarity_threshold must be in"):
            AgentConfig(embedding_query_similarity_threshold=-0.1)

    def test_embedding_query_similarity_threshold_too_high(self):
        """embedding_query_similarity_threshold > 1.0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="embedding_query_similarity_threshold must be in"):
            AgentConfig(embedding_query_similarity_threshold=1.1)

    def test_embedding_history_decay_too_low(self):
        """embedding_history_decay < 0.0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="embedding_history_decay must be in"):
            AgentConfig(embedding_history_decay=-0.1)

    def test_embedding_history_decay_at_one(self):
        """embedding_history_decay = 1.0 should raise ConfigurationError (must be < 1.0)."""
        with pytest.raises(ConfigurationError, match="embedding_history_decay must be in"):
            AgentConfig(embedding_history_decay=1.0)

    def test_search_max_results_too_low(self):
        """search_max_results < 1 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="search_max_results must be >= 1"):
            AgentConfig(search_max_results=0)

    def test_search_retries_negative(self):
        """search_retries < 0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="search_retries must be >= 0"):
            AgentConfig(search_retries=-1)

    def test_search_timeout_zero(self):
        """search_timeout <= 0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="search_timeout must be > 0"):
            AgentConfig(search_timeout=0.0)

    def test_search_timeout_negative(self):
        """search_timeout < 0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="search_timeout must be > 0"):
            AgentConfig(search_timeout=-1.0)

    def test_max_concurrent_queries_too_low(self):
        """max_concurrent_queries < 1 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="max_concurrent_queries must be >= 1"):
            AgentConfig(max_concurrent_queries=0)

    def test_ddg_safesearch_invalid(self):
        """Invalid ddg_safesearch should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="ddg_safesearch must be"):
            AgentConfig(ddg_safesearch="invalid")

    def test_ddg_safesearch_valid_values(self):
        """Valid ddg_safesearch values should pass."""
        for value in ("off", "moderate", "strict"):
            cfg = AgentConfig(ddg_safesearch=value)
            assert cfg.ddg_safesearch == value

    def test_conflicting_search_flags_both_true(self):
        """Both no_auto_search=True and force_search=True should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Conflicting flags"):
            AgentConfig(no_auto_search=True, force_search=True)

    def test_no_auto_search_alone_is_valid(self):
        """no_auto_search=True alone should be valid."""
        cfg = AgentConfig(no_auto_search=True, force_search=False)
        assert cfg.no_auto_search is True
        assert cfg.force_search is False

    def test_force_search_alone_is_valid(self):
        """force_search=True alone should be valid."""
        cfg = AgentConfig(no_auto_search=False, force_search=True)
        assert cfg.no_auto_search is False
        assert cfg.force_search is True

    def test_both_search_flags_false_is_valid(self):
        """Both flags False should be valid (default behavior)."""
        cfg = AgentConfig(no_auto_search=False, force_search=False)
        assert cfg.no_auto_search is False
        assert cfg.force_search is False

    def test_boundary_values_pass(self):
        """Boundary values should pass validation."""
        # Test minimum valid values
        cfg = AgentConfig(
            max_rounds=1,
            max_conversation_chars=1024,
            compact_keep_turns=1,
            max_followup_suggestions=0,
            max_fill_attempts=0,
            max_relevance_llm_checks=0,
            assistant_num_ctx=512,
            robot_num_ctx=512,
            assistant_num_predict=1,
            robot_num_predict=1,
            robot_temp=0.0,
            assistant_temp=0.0,
            robot_top_p=0.0,
            assistant_top_p=0.0,
            embedding_similarity_threshold=0.0,
            embedding_result_similarity_threshold=0.0,
            embedding_query_similarity_threshold=0.0,
            embedding_history_decay=0.0,
            search_max_results=1,
            search_retries=0,
            search_timeout=0.001,
            max_concurrent_queries=1,
        )
        assert cfg is not None

        # Test maximum valid values
        cfg = AgentConfig(
            robot_temp=2.0,
            assistant_temp=2.0,
            robot_top_p=1.0,
            assistant_top_p=1.0,
            embedding_similarity_threshold=1.0,
            embedding_result_similarity_threshold=1.0,
            embedding_query_similarity_threshold=1.0,
            embedding_history_decay=0.99,
        )
        assert cfg is not None


# Note: Environment variable tests are integration tests and complex to test in unit tests
# since they're evaluated at module load time. These are better tested manually or in E2E tests.
