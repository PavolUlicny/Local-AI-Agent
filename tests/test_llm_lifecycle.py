"""Tests for LLM lifecycle management (building, rebuilding, restoring)."""

from __future__ import annotations

from unittest.mock import Mock, patch

from src.config import AgentConfig
from src.llm_lifecycle import LLMManager


class TestLLMManagerInit:
    """Tests for LLMManager initialization."""

    @patch("src.llm_lifecycle._chains.build_llms")
    @patch("src.llm_lifecycle._chains.build_chains")
    def test_init_builds_llms_and_chains(self, mock_build_chains, mock_build_llms):
        """LLMManager should build LLMs and chains on initialization."""
        cfg = AgentConfig()
        mock_llm_robot = Mock()
        mock_llm_assistant = Mock()
        mock_build_llms.return_value = (mock_llm_robot, mock_llm_assistant)
        mock_chains_dict = {"planning": Mock(), "response": Mock()}
        mock_build_chains.return_value = mock_chains_dict

        manager = LLMManager(cfg)

        # Should build LLMs
        mock_build_llms.assert_called_once_with(cfg)

        # Should build chains
        mock_build_chains.assert_called_once_with(mock_llm_robot, mock_llm_assistant)

        # Should store LLMs and chains
        assert manager.llm_robot is mock_llm_robot
        assert manager.llm_assistant is mock_llm_assistant
        assert manager.chains is mock_chains_dict

    @patch("src.llm_lifecycle._chains.build_llms")
    @patch("src.llm_lifecycle._chains.build_chains")
    def test_init_stores_base_params(self, mock_build_chains, mock_build_llms):
        """LLMManager should store original configuration parameters."""
        cfg = AgentConfig(
            assistant_num_ctx=32000,
            robot_num_ctx=8000,
            assistant_num_predict=4000,
            robot_num_predict=2000,
        )
        mock_build_llms.return_value = (Mock(), Mock())
        mock_build_chains.return_value = {}

        manager = LLMManager(cfg)

        # Should store original parameters
        assert manager._base_params["assistant_num_ctx"] == 32000
        assert manager._base_params["robot_num_ctx"] == 8000
        assert manager._base_params["assistant_num_predict"] == 4000
        assert manager._base_params["robot_num_predict"] == 2000


class TestLLMManagerRebuild:
    """Tests for LLM rebuilding with reduced context."""

    @patch("src.llm_lifecycle._chains.build_llms")
    @patch("src.llm_lifecycle._chains.build_chains")
    def test_rebuild_with_reduced_context_halves_context(self, mock_build_chains, mock_build_llms):
        """rebuild_with_reduced_context should halve the context window."""
        cfg = AgentConfig(assistant_num_ctx=8000, robot_num_ctx=8000, assistant_num_predict=4000)
        mock_build_llms.return_value = (Mock(), Mock())
        mock_build_chains.return_value = {}

        manager = LLMManager(cfg)
        rebuild_counts = {}

        # Reset mock to only count rebuild calls
        mock_build_llms.reset_mock()
        mock_build_chains.reset_mock()

        # Rebuild with reduced context
        manager.rebuild_with_reduced_context("planning", "test stage", rebuild_counts)

        # Should halve context: 8000 -> 4000
        assert cfg.assistant_num_ctx == 4000
        assert cfg.robot_num_ctx == 4000
        # Should reduce predict to half of context
        assert cfg.assistant_num_predict == 2000

        # Should rebuild LLMs
        mock_build_llms.assert_called_once()
        mock_build_chains.assert_called_once()

        # Should increment rebuild count
        assert rebuild_counts["planning"] == 1

    @patch("src.llm_lifecycle._chains.build_llms")
    @patch("src.llm_lifecycle._chains.build_chains")
    def test_rebuild_respects_minimum_context(self, mock_build_chains, mock_build_llms):
        """rebuild_with_reduced_context should not go below 1024 context."""
        cfg = AgentConfig(assistant_num_ctx=1500, robot_num_ctx=1500, assistant_num_predict=750)
        mock_build_llms.return_value = (Mock(), Mock())
        mock_build_chains.return_value = {}

        manager = LLMManager(cfg)
        rebuild_counts = {}

        mock_build_llms.reset_mock()

        # Rebuild - should use 1024 minimum instead of 750 (1500 // 2)
        manager.rebuild_with_reduced_context("planning", "test", rebuild_counts)

        # Should use 1024 (minimum) instead of 750 (half of 1500)
        assert cfg.assistant_num_ctx == 1024
        assert cfg.robot_num_ctx == 1024
        # Predict should be at least 512
        assert cfg.assistant_num_predict == 512

    @patch("src.llm_lifecycle._chains.build_llms")
    @patch("src.llm_lifecycle._chains.build_chains")
    def test_rebuild_never_increases_context(self, mock_build_chains, mock_build_llms):
        """rebuild_with_reduced_context should never increase context size."""
        cfg = AgentConfig(assistant_num_ctx=512, robot_num_ctx=512, assistant_num_predict=256)
        mock_build_llms.return_value = (Mock(), Mock())
        mock_build_chains.return_value = {}

        manager = LLMManager(cfg)
        rebuild_counts = {}

        mock_build_llms.reset_mock()

        # Rebuild - minimum is 1024, but current is 512, so should stay at 512
        # Calculation: max(1024, 512 // 2) = max(1024, 256) = 1024
        # Then: min(512, 1024) = 512 (never grow!)
        manager.rebuild_with_reduced_context("planning", "test", rebuild_counts)

        # Should not grow (512 < 1024 minimum, but we never grow)
        assert cfg.assistant_num_ctx == 512
        assert cfg.robot_num_ctx == 512


class TestLLMManagerRestore:
    """Tests for restoring original LLM parameters."""

    @patch("src.llm_lifecycle._chains.build_llms")
    @patch("src.llm_lifecycle._chains.build_chains")
    def test_restore_rebuilds_with_original_params(self, mock_build_chains, mock_build_llms):
        """restore_original_params should rebuild LLMs with original configuration."""
        cfg = AgentConfig(assistant_num_ctx=8000, robot_num_ctx=8000, assistant_num_predict=4000)
        mock_build_llms.return_value = (Mock(), Mock())
        mock_build_chains.return_value = {}

        manager = LLMManager(cfg)
        rebuild_counts = {}

        # Reduce context
        manager.rebuild_with_reduced_context("planning", "test", rebuild_counts)
        assert cfg.assistant_num_ctx == 4000  # Reduced

        # Reset mocks
        mock_build_llms.reset_mock()
        mock_build_chains.reset_mock()

        # Restore
        manager.restore_original_params()

        # Should restore to original values
        assert cfg.assistant_num_ctx == 8000
        assert cfg.robot_num_ctx == 8000
        assert cfg.assistant_num_predict == 4000

        # Should rebuild
        mock_build_llms.assert_called_once()
        mock_build_chains.assert_called_once()

    @patch("src.llm_lifecycle._chains.build_llms")
    @patch("src.llm_lifecycle._chains.build_chains")
    def test_restore_skips_rebuild_if_unchanged(self, mock_build_chains, mock_build_llms):
        """restore_original_params should skip rebuild if parameters unchanged."""
        cfg = AgentConfig(assistant_num_ctx=8000, robot_num_ctx=8000)
        mock_build_llms.return_value = (Mock(), Mock())
        mock_build_chains.return_value = {}

        manager = LLMManager(cfg)

        # Reset mocks
        mock_build_llms.reset_mock()
        mock_build_chains.reset_mock()

        # Restore without any changes
        manager.restore_original_params()

        # Should NOT rebuild since params unchanged
        mock_build_llms.assert_not_called()
        mock_build_chains.assert_not_called()


class TestLLMManagerGetters:
    """Tests for getter methods."""

    @patch("src.llm_lifecycle._chains.build_llms")
    @patch("src.llm_lifecycle._chains.build_chains")
    def test_get_llms_returns_current_llms(self, mock_build_chains, mock_build_llms):
        """get_llms should return current LLM instances."""
        cfg = AgentConfig()
        mock_robot = Mock()
        mock_assistant = Mock()
        mock_build_llms.return_value = (mock_robot, mock_assistant)
        mock_build_chains.return_value = {}

        manager = LLMManager(cfg)
        robot, assistant = manager.get_llms()

        assert robot is mock_robot
        assert assistant is mock_assistant

    @patch("src.llm_lifecycle._chains.build_llms")
    @patch("src.llm_lifecycle._chains.build_chains")
    def test_get_chains_returns_current_chains(self, mock_build_chains, mock_build_llms):
        """get_chains should return current chains dictionary."""
        cfg = AgentConfig()
        mock_chains_dict = {"planning": Mock(), "response": Mock()}
        mock_build_llms.return_value = (Mock(), Mock())
        mock_build_chains.return_value = mock_chains_dict

        manager = LLMManager(cfg)
        chains = manager.get_chains()

        assert chains is mock_chains_dict
        assert "planning" in chains
        assert "response" in chains


class TestLLMManagerIntegration:
    """Integration tests for full rebuild cycle."""

    @patch("src.llm_lifecycle._chains.build_llms")
    @patch("src.llm_lifecycle._chains.build_chains")
    def test_full_rebuild_cycle(self, mock_build_chains, mock_build_llms):
        """Test complete cycle: build -> reduce -> restore."""
        cfg = AgentConfig(assistant_num_ctx=8000, robot_num_ctx=8000, assistant_num_predict=4000)
        mock_build_llms.return_value = (Mock(), Mock())
        mock_build_chains.return_value = {}

        manager = LLMManager(cfg)
        rebuild_counts = {}

        # Initial state
        assert cfg.assistant_num_ctx == 8000

        # Reduce once
        manager.rebuild_with_reduced_context("planning", "test", rebuild_counts)
        assert cfg.assistant_num_ctx == 4000
        assert rebuild_counts["planning"] == 1

        # Reduce again
        manager.rebuild_with_reduced_context("planning", "test", rebuild_counts)
        assert cfg.assistant_num_ctx == 2000
        assert rebuild_counts["planning"] == 2

        # Restore
        manager.restore_original_params()
        assert cfg.assistant_num_ctx == 8000
        assert rebuild_counts["planning"] == 2  # Count unchanged by restore
