"""Configuration using Pydantic for improved validation and error messages."""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from .exceptions import ConfigurationError


class AgentConfig(BaseModel):
    """Configuration parameters for Agent behavior and LLM settings.

    Uses Pydantic for robust validation with clear error messages.
    All parameters have sensible defaults and can be overridden via CLI arguments
    or environment variables.
    """

    # Behavioral flags
    no_auto_search: bool = False
    force_search: bool = False

    # Search and round limits
    max_rounds: int = Field(default=8, ge=1, description="Maximum search rounds per query")
    max_conversation_chars: int = Field(default=24000, ge=1024, description="Max conversation history size")
    compact_keep_turns: int = Field(default=10, ge=1, description="Turns to keep when compacting history")
    max_followup_suggestions: int = Field(default=8, ge=0, description="Max followup suggestions to generate")
    max_fill_attempts: int = Field(default=2, ge=0, description="Max attempts to fill missing results")
    max_relevance_llm_checks: int = Field(default=2, ge=0, description="Max LLM checks for result relevance")

    # LLM context and prediction parameters
    assistant_num_ctx: int = Field(default=16384, ge=512, description="Assistant context window size")
    robot_num_ctx: int = Field(default=16384, ge=512, description="Robot context window size")
    assistant_num_predict: int = Field(default=4096, ge=1, description="Assistant max prediction tokens")
    robot_num_predict: int = Field(default=512, ge=1, description="Robot max prediction tokens")

    # LLM sampling parameters
    robot_temp: float = Field(default=0.0, ge=0.0, le=2.0, description="Robot temperature")
    assistant_temp: float = Field(default=0.6, ge=0.0, le=2.0, description="Assistant temperature")
    robot_top_p: float = Field(default=0.4, ge=0.0, le=1.0, description="Robot nucleus sampling threshold")
    assistant_top_p: float = Field(default=0.8, ge=0.0, le=1.0, description="Assistant nucleus sampling threshold")
    robot_top_k: int = Field(default=20, description="Robot top-k sampling")
    assistant_top_k: int = Field(default=80, description="Assistant top-k sampling")
    robot_repeat_penalty: float = Field(default=1.1, description="Robot repetition penalty")
    assistant_repeat_penalty: float = Field(default=1.2, description="Assistant repetition penalty")

    # Search engine configuration
    ddg_region: str = Field(default="us-en", description="DuckDuckGo region")
    ddg_safesearch: Literal["off", "moderate", "strict"] = Field(
        default="moderate", description="DuckDuckGo SafeSearch level"
    )
    ddg_backend: str = Field(default="auto", description="DuckDuckGo backend")

    # Search behavior
    search_max_results: int = Field(default=5, ge=1, description="Max results per search query")
    search_retries: int = Field(default=3, ge=0, description="Max search retry attempts")
    search_timeout: float = Field(default=10.0, gt=0, description="Search timeout in seconds")
    max_concurrent_queries: int = Field(default=10, ge=1, description="Max parallel queries per round")

    # Logging configuration
    log_level: str = Field(default="WARNING", description="Logging level")
    log_file: str | None = Field(default=None, description="Log file path")
    log_console: bool = Field(default=True, description="Enable console logging")

    # CLI question
    question: str | None = Field(default=None, description="Initial question from CLI")

    # Model names (support environment variables)
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("AGENT_EMBEDDING_MODEL", "embeddinggemma:300m"),
        description="Embedding model name",
    )
    robot_model: str = Field(
        default_factory=lambda: os.getenv("AGENT_ROBOT_MODEL", "phi4-mini:3.8b"),
        description="Robot (planning) model name",
    )
    assistant_model: str = Field(
        default_factory=lambda: os.getenv("AGENT_ASSISTANT_MODEL", "llama3.1:8b"),
        description="Assistant (response) model name",
    )

    # Embedding thresholds
    embedding_similarity_threshold: float = Field(
        default=0.35, ge=0.0, le=1.0, description="Minimum similarity for embedding deduplication"
    )
    embedding_history_decay: float = Field(
        default=0.65, ge=0.0, lt=1.0, description="History decay factor for embeddings"
    )
    embedding_result_similarity_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum similarity for result filtering"
    )
    embedding_query_similarity_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum similarity for query filtering"
    )

    @model_validator(mode="after")
    def validate_search_flags(self) -> "AgentConfig":
        """Ensure search flags are not contradictory.

        Raises:
            ValueError: If both no_auto_search and force_search are True
        """
        if self.no_auto_search and self.force_search:
            raise ValueError(
                "Conflicting flags: no_auto_search and force_search cannot both be True. "
                "Use force_search=True alone to always search, or no_auto_search=True to disable auto-decision."
            )
        return self

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "extra": "forbid",  # Raise error on unknown fields
        "use_enum_values": True,  # Use enum values instead of enum objects
    }

    def _format_validation_error(self, e: ValidationError, data: dict) -> str:
        """Format Pydantic ValidationError into user-friendly message.

        Args:
            e: Pydantic ValidationError
            data: Input data dict

        Returns:
            Formatted error message string
        """
        errors = e.errors()
        if not errors:
            return f"Configuration validation failed: {e}"

        first_error = errors[0]
        field_name_raw = first_error["loc"][0] if first_error["loc"] else "unknown"
        field_name = str(field_name_raw)  # Convert to str for dict key access
        error_type = first_error["type"]
        value = data.get(field_name)
        ctx = first_error.get("ctx", {})

        # Check if this is a range validation (has both min and max constraints)
        # Extract min and max from field constraints
        field_info = self.model_fields.get(field_name)
        min_val = None
        max_val = None
        has_gt_constraint = False
        has_lt_constraint = False

        if field_info and hasattr(field_info, "metadata"):
            for constraint in field_info.metadata:
                if hasattr(constraint, "ge"):
                    min_val = constraint.ge
                elif hasattr(constraint, "gt"):
                    min_val = constraint.gt
                    has_gt_constraint = True
                if hasattr(constraint, "le"):
                    max_val = constraint.le
                elif hasattr(constraint, "lt"):
                    max_val = constraint.lt
                    has_lt_constraint = True

        # For fields with both min and max constraints, format as range
        if min_val is not None and max_val is not None:
            # Format range based on whether it's inclusive or exclusive
            left_bracket = "(" if has_gt_constraint else "["
            right_bracket = ")" if has_lt_constraint else "]"
            return f"{field_name} must be in {left_bracket}{min_val}, {max_val}{right_bracket}, got {value}"

        # Map Pydantic error types to clear messages
        if "greater_than_equal" in error_type:
            constraint = ctx.get("ge")
            return f"{field_name} must be >= {constraint}, got {value}"
        elif "less_than_equal" in error_type:
            constraint = ctx.get("le")
            return f"{field_name} must be <= {constraint}, got {value}"
        elif "greater_than" in error_type:
            constraint = ctx.get("gt")
            return f"{field_name} must be > {constraint}, got {value}"
        elif "less_than" in error_type:
            constraint = ctx.get("lt")
            return f"{field_name} must be < {constraint}, got {value}"
        elif "literal_error" in error_type:
            expected = ctx.get("expected")
            return f"{field_name} must be {expected}, got {value}"
        else:
            # Generic error message
            msg = first_error.get("msg", str(e))
            return f"Invalid configuration: {msg}"

    def __init__(self, **data: Any) -> None:
        """Initialize config with validation error conversion."""
        try:
            super().__init__(**data)
        except ValidationError as e:
            # Convert Pydantic ValidationError to ConfigurationError with clearer messages
            msg = self._format_validation_error(e, data)
            raise ConfigurationError(msg) from e

    @property
    def auto_search_decision(self) -> bool:
        """Whether to automatically decide when to search."""
        return not self.no_auto_search


__all__ = ["AgentConfig"]
