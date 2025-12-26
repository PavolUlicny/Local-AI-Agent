"""Custom exceptions for Local AI Agent.

This module defines a hierarchy of exceptions for different error categories,
making error handling more specific and maintainable.
"""

from __future__ import annotations

import importlib
from typing import Type


# ============================================================================
# Base Exception
# ============================================================================


class AgentError(Exception):
    """Base exception for all Local AI Agent errors.

    All custom exceptions in this project should inherit from this base class.
    This allows catching all project-specific errors with a single except clause.
    """


# ============================================================================
# Configuration & Validation Errors
# ============================================================================


class ConfigurationError(AgentError):
    """Raised when configuration values are invalid.

    Examples:
        - Context window size below minimum
        - Invalid parameter ranges
        - Incompatible configuration combinations
    """


class InputValidationError(AgentError):
    """Raised when user input fails validation.

    Examples:
        - Prompt injection attempts detected
        - Input exceeds maximum length
        - Input contains prohibited patterns
    """


# ============================================================================
# Search & Orchestration Errors
# ============================================================================


class SearchError(AgentError):
    """Base class for search-related errors."""


class SearchAbort(SearchError):
    """Raised when search orchestration must halt due to a fatal error.

    This is used in search orchestration when a non-recoverable error occurs
    that prevents continuing the search process (e.g., model not found,
    repeated LLM failures, resource exhaustion).
    """


class SearchTimeoutError(SearchError):
    """Raised when search operations exceed time limits."""


# ============================================================================
# LLM & Model Errors
# ============================================================================


class _DefaultResponseError(Exception):
    """Fallback ResponseError when ollama module not available."""


# Import ResponseError from ollama if available, otherwise use default
_resp: Type[Exception] = _DefaultResponseError
for modname in ("ollama", "ollama._types"):
    try:
        mod = importlib.import_module(modname)
    except Exception:
        continue
    if hasattr(mod, "ResponseError"):
        _resp = mod.ResponseError
        break

ResponseError: Type[Exception] = _resp
"""LLM response error from ollama library.

Note: This is imported from the ollama library when available, otherwise
a default implementation is used. Used for LLM-specific errors like
context length exceeded, model not found, etc.
"""


class ContextLengthError(AgentError):
    """Raised when LLM context window is exceeded.

    This is a more specific wrapper around ResponseError for context length
    issues, allowing targeted handling of this common error case.
    """


class ModelNotFoundError(AgentError):
    """Raised when a required model is not available.

    Examples:
        - Configured model not installed
        - Embedding model not found
        - Assistant/robot model unavailable
    """


# ============================================================================
# Resource & Network Errors
# ============================================================================


class ResourceError(AgentError):
    """Base class for resource-related errors."""


class NetworkError(ResourceError):
    """Raised when network operations fail.

    Examples:
        - Search API unavailable
        - Connection timeouts
        - DNS resolution failures
    """


class ResourceExhaustedError(ResourceError):
    """Raised when resources are exhausted.

    Examples:
        - Maximum retry attempts exceeded
        - Memory limits reached
        - File handle limits exceeded
    """


# ============================================================================
# Exports
# ============================================================================


__all__ = [
    # Base
    "AgentError",
    # Configuration & Validation
    "ConfigurationError",
    "InputValidationError",
    # Search
    "SearchError",
    "SearchAbort",
    "SearchTimeoutError",
    # LLM & Models
    "ResponseError",
    "ContextLengthError",
    "ModelNotFoundError",
    # Resources
    "ResourceError",
    "NetworkError",
    "ResourceExhaustedError",
]
