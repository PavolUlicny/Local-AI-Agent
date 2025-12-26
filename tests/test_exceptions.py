"""Comprehensive tests for exception hierarchy.

Tests all custom exceptions to ensure proper inheritance and behavior.
"""

from __future__ import annotations

import pytest

from src.exceptions import (
    # Base
    AgentError,
    # Configuration & Validation
    ConfigurationError,
    InputValidationError,
    # Search
    SearchError,
    SearchAbort,
    SearchTimeoutError,
    # LLM & Models
    ResponseError,
    ContextLengthError,
    ModelNotFoundError,
    # Resources
    ResourceError,
    NetworkError,
    ResourceExhaustedError,
)


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy."""

    def test_agent_error_is_base_exception(self):
        """AgentError should inherit from Exception."""
        assert issubclass(AgentError, Exception)

    def test_configuration_error_inherits_from_agent_error(self):
        """ConfigurationError should inherit from AgentError."""
        assert issubclass(ConfigurationError, AgentError)
        assert issubclass(ConfigurationError, Exception)

    def test_input_validation_error_inherits_from_agent_error(self):
        """InputValidationError should inherit from AgentError."""
        assert issubclass(InputValidationError, AgentError)
        assert issubclass(InputValidationError, Exception)

    def test_search_error_inherits_from_agent_error(self):
        """SearchError should inherit from AgentError."""
        assert issubclass(SearchError, AgentError)
        assert issubclass(SearchError, Exception)

    def test_search_abort_inherits_from_search_error(self):
        """SearchAbort should inherit from SearchError."""
        assert issubclass(SearchAbort, SearchError)
        assert issubclass(SearchAbort, AgentError)
        assert issubclass(SearchAbort, Exception)

    def test_search_timeout_error_inherits_from_search_error(self):
        """SearchTimeoutError should inherit from SearchError."""
        assert issubclass(SearchTimeoutError, SearchError)
        assert issubclass(SearchTimeoutError, AgentError)
        assert issubclass(SearchTimeoutError, Exception)

    def test_context_length_error_inherits_from_agent_error(self):
        """ContextLengthError should inherit from AgentError."""
        assert issubclass(ContextLengthError, AgentError)
        assert issubclass(ContextLengthError, Exception)

    def test_model_not_found_error_inherits_from_agent_error(self):
        """ModelNotFoundError should inherit from AgentError."""
        assert issubclass(ModelNotFoundError, AgentError)
        assert issubclass(ModelNotFoundError, Exception)

    def test_resource_error_inherits_from_agent_error(self):
        """ResourceError should inherit from AgentError."""
        assert issubclass(ResourceError, AgentError)
        assert issubclass(ResourceError, Exception)

    def test_network_error_inherits_from_resource_error(self):
        """NetworkError should inherit from ResourceError."""
        assert issubclass(NetworkError, ResourceError)
        assert issubclass(NetworkError, AgentError)
        assert issubclass(NetworkError, Exception)

    def test_resource_exhausted_error_inherits_from_resource_error(self):
        """ResourceExhaustedError should inherit from ResourceError."""
        assert issubclass(ResourceExhaustedError, ResourceError)
        assert issubclass(ResourceExhaustedError, AgentError)
        assert issubclass(ResourceExhaustedError, Exception)


class TestExceptionRaising:
    """Test that exceptions can be raised and caught."""

    def test_agent_error_can_be_raised(self):
        """AgentError can be raised and caught."""
        with pytest.raises(AgentError, match="test"):
            raise AgentError("test")

    def test_configuration_error_can_be_raised(self):
        """ConfigurationError can be raised and caught."""
        with pytest.raises(ConfigurationError, match="invalid config"):
            raise ConfigurationError("invalid config")

    def test_input_validation_error_can_be_raised(self):
        """InputValidationError can be raised and caught."""
        with pytest.raises(InputValidationError, match="bad input"):
            raise InputValidationError("bad input")

    def test_search_abort_can_be_raised(self):
        """SearchAbort can be raised and caught."""
        with pytest.raises(SearchAbort, match="fatal error"):
            raise SearchAbort("fatal error")

    def test_search_timeout_error_can_be_raised(self):
        """SearchTimeoutError can be raised and caught."""
        with pytest.raises(SearchTimeoutError, match="timeout"):
            raise SearchTimeoutError("timeout")

    def test_context_length_error_can_be_raised(self):
        """ContextLengthError can be raised and caught."""
        with pytest.raises(ContextLengthError, match="context exceeded"):
            raise ContextLengthError("context exceeded")

    def test_model_not_found_error_can_be_raised(self):
        """ModelNotFoundError can be raised and caught."""
        with pytest.raises(ModelNotFoundError, match="model missing"):
            raise ModelNotFoundError("model missing")

    def test_network_error_can_be_raised(self):
        """NetworkError can be raised and caught."""
        with pytest.raises(NetworkError, match="connection failed"):
            raise NetworkError("connection failed")

    def test_resource_exhausted_error_can_be_raised(self):
        """ResourceExhaustedError can be raised and caught."""
        with pytest.raises(ResourceExhaustedError, match="out of retries"):
            raise ResourceExhaustedError("out of retries")


class TestExceptionCatching:
    """Test that exceptions can be caught at different levels of hierarchy."""

    def test_catch_configuration_error_as_agent_error(self):
        """ConfigurationError can be caught as AgentError."""
        with pytest.raises(AgentError):
            raise ConfigurationError("test")

    def test_catch_search_abort_as_search_error(self):
        """SearchAbort can be caught as SearchError."""
        with pytest.raises(SearchError):
            raise SearchAbort("test")

    def test_catch_search_abort_as_agent_error(self):
        """SearchAbort can be caught as AgentError."""
        with pytest.raises(AgentError):
            raise SearchAbort("test")

    def test_catch_network_error_as_resource_error(self):
        """NetworkError can be caught as ResourceError."""
        with pytest.raises(ResourceError):
            raise NetworkError("test")

    def test_catch_network_error_as_agent_error(self):
        """NetworkError can be caught as AgentError."""
        with pytest.raises(AgentError):
            raise NetworkError("test")

    def test_catch_all_custom_exceptions_as_agent_error(self):
        """All custom exceptions can be caught as AgentError."""
        exceptions = [
            ConfigurationError("test"),
            InputValidationError("test"),
            SearchAbort("test"),
            SearchTimeoutError("test"),
            ContextLengthError("test"),
            ModelNotFoundError("test"),
            NetworkError("test"),
            ResourceExhaustedError("test"),
        ]

        for exc in exceptions:
            with pytest.raises(AgentError):
                raise exc


class TestResponseError:
    """Test ResponseError import mechanism."""

    def test_response_error_is_exception(self):
        """ResponseError should be an Exception type."""
        assert issubclass(ResponseError, Exception)

    def test_response_error_can_be_raised(self):
        """ResponseError can be raised and caught."""
        with pytest.raises(ResponseError):
            raise ResponseError("test")

    def test_response_error_with_message(self):
        """ResponseError preserves error messages."""
        try:
            raise ResponseError("custom message")
        except ResponseError as exc:
            assert "custom message" in str(exc)


class TestExceptionMessages:
    """Test that exception messages are preserved."""

    def test_configuration_error_preserves_message(self):
        """ConfigurationError should preserve error message."""
        try:
            raise ConfigurationError("context too small")
        except ConfigurationError as exc:
            assert "context too small" in str(exc)

    def test_input_validation_error_preserves_message(self):
        """InputValidationError should preserve error message."""
        try:
            raise InputValidationError("suspicious patterns detected")
        except InputValidationError as exc:
            assert "suspicious patterns detected" in str(exc)

    def test_search_abort_preserves_message(self):
        """SearchAbort should preserve error message."""
        try:
            raise SearchAbort("model not found")
        except SearchAbort as exc:
            assert "model not found" in str(exc)

    def test_network_error_preserves_message(self):
        """NetworkError should preserve error message."""
        try:
            raise NetworkError("connection timeout")
        except NetworkError as exc:
            assert "connection timeout" in str(exc)


class TestExceptionDocstrings:
    """Test that all exceptions have proper documentation."""

    def test_agent_error_has_docstring(self):
        """AgentError should have a docstring."""
        assert AgentError.__doc__ is not None
        assert len(AgentError.__doc__) > 0

    def test_configuration_error_has_docstring(self):
        """ConfigurationError should have a docstring."""
        assert ConfigurationError.__doc__ is not None
        assert len(ConfigurationError.__doc__) > 0

    def test_input_validation_error_has_docstring(self):
        """InputValidationError should have a docstring."""
        assert InputValidationError.__doc__ is not None
        assert len(InputValidationError.__doc__) > 0

    def test_search_error_has_docstring(self):
        """SearchError should have a docstring."""
        assert SearchError.__doc__ is not None
        assert len(SearchError.__doc__) > 0

    def test_search_abort_has_docstring(self):
        """SearchAbort should have a docstring."""
        assert SearchAbort.__doc__ is not None
        assert len(SearchAbort.__doc__) > 0

    def test_all_exceptions_have_docstrings(self):
        """All exception classes should have docstrings."""
        exceptions = [
            AgentError,
            ConfigurationError,
            InputValidationError,
            SearchError,
            SearchAbort,
            SearchTimeoutError,
            ContextLengthError,
            ModelNotFoundError,
            ResourceError,
            NetworkError,
            ResourceExhaustedError,
        ]

        for exc_class in exceptions:
            assert exc_class.__doc__ is not None, f"{exc_class.__name__} missing docstring"
            assert len(exc_class.__doc__) > 0, f"{exc_class.__name__} has empty docstring"
