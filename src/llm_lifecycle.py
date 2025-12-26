"""LLM lifecycle management utilities.

This module handles building, rebuilding, and restoring LLM instances
when context windows need adjustment or configuration changes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Tuple

from . import chains as _chains
from .constants import ChainName, MAX_REBUILD_RETRIES

if TYPE_CHECKING:  # pragma: no cover
    from .config import AgentConfig
    from .protocols import LLMChain
    from langchain_ollama import OllamaLLM


class LLMManager:
    """Manages LLM lifecycle: building, rebuilding with reduced context, and restoration.

    This class encapsulates the complexity of managing LLM instances that need
    to be rebuilt when context windows are exceeded.
    """

    def __init__(self, cfg: "AgentConfig") -> None:
        """Initialize LLM manager with configuration.

        Args:
            cfg: Agent configuration with LLM parameters
        """
        self.cfg = cfg

        # Store original parameters for restoration
        self._base_params = {
            "assistant_num_ctx": cfg.assistant_num_ctx,
            "robot_num_ctx": cfg.robot_num_ctx,
            "assistant_num_predict": cfg.assistant_num_predict,
            "robot_num_predict": cfg.robot_num_predict,
        }

        # Build initial LLMs
        self.llm_robot: "OllamaLLM"
        self.llm_assistant: "OllamaLLM"
        self.chains: Dict[ChainName, "LLMChain"]
        self._build_llms()

    def _build_llms(self) -> None:
        """Build LLMs and chains from current configuration."""
        self.llm_robot, self.llm_assistant = _chains.build_llms(self.cfg)
        self.chains = _chains.build_chains(self.llm_robot, self.llm_assistant)

    def rebuild_with_reduced_context(
        self,
        stage_key: str,
        label: str,
        rebuild_counts: Dict[str, int],
    ) -> None:
        """Rebuild LLMs with reduced context window.

        This is called when a context-length error occurs, typically during LLM
        invocation. The context is halved (but never grows) and LLMs are rebuilt.

        Args:
            stage_key: Key for tracking rebuild count (e.g., "planning", "answer")
            label: Human-readable label for logging
            rebuild_counts: Mutable dict tracking rebuild attempts by stage
        """
        rebuild_counts[stage_key] = rebuild_counts.get(stage_key, 0) + 1

        current_ctx = min(self.cfg.assistant_num_ctx, self.cfg.robot_num_ctx)
        current_predict = self.cfg.assistant_num_predict

        # Compute reduced context: halve it but never grow it, with 1024 minimum
        reduced_ctx_candidate = max(1024, current_ctx // 2)
        reduced_ctx = min(current_ctx, reduced_ctx_candidate)
        reduced_predict = max(512, min(current_predict, reduced_ctx // 2))

        logging.info(
            "Context too large (%s); rebuild %s/%s with num_ctx=%s, num_predict=%s.",
            label,
            rebuild_counts[stage_key],
            MAX_REBUILD_RETRIES,
            reduced_ctx,
            reduced_predict,
        )

        self._rebuild_with_params(reduced_ctx, reduced_predict)

    def _rebuild_with_params(self, new_ctx: int, new_predict: int) -> None:
        """Rebuild LLMs with specific context and prediction parameters.

        Args:
            new_ctx: New context window size
            new_predict: New max prediction tokens
        """
        self.cfg.assistant_num_ctx = new_ctx
        self.cfg.robot_num_ctx = new_ctx
        self.cfg.assistant_num_predict = new_predict
        # Robot uses smaller prediction window
        self.cfg.robot_num_predict = min(self.cfg.robot_num_predict, new_predict)

        self._build_llms()

    def restore_original_params(self) -> None:
        """Restore LLMs to original configuration parameters.

        This is typically called at the end of a query to reset any context
        reductions that occurred during processing.
        """
        base = self._base_params
        cfg = self.cfg

        # Check if restoration is needed
        needs_restore = any(
            [
                cfg.assistant_num_ctx != base["assistant_num_ctx"],
                cfg.robot_num_ctx != base["robot_num_ctx"],
                cfg.assistant_num_predict != base["assistant_num_predict"],
                cfg.robot_num_predict != base["robot_num_predict"],
            ]
        )

        if not needs_restore:
            return

        # Restore configuration
        cfg.assistant_num_ctx = base["assistant_num_ctx"]
        cfg.robot_num_ctx = base["robot_num_ctx"]
        cfg.assistant_num_predict = base["assistant_num_predict"]
        cfg.robot_num_predict = base["robot_num_predict"]

        # Rebuild with original params
        self._build_llms()

    def get_llms(self) -> Tuple["OllamaLLM", "OllamaLLM"]:
        """Get current LLM instances.

        Returns:
            Tuple of (robot_llm, assistant_llm)
        """
        return self.llm_robot, self.llm_assistant

    def get_chains(self) -> Dict[ChainName, "LLMChain"]:
        """Get current chain dictionary.

        Returns:
            Dictionary mapping chain names to configured chains
        """
        return self.chains


__all__ = ["LLMManager"]
