"""Tests for search retry utilities."""

from __future__ import annotations

from unittest.mock import Mock

from src.config import AgentConfig
from src.search_retry_utils import (
    SearchRetryContext,
    safe_close_client,
    handle_search_exception,
    log_final_failure,
    should_notify_retry,
)
from ddgs.exceptions import DDGSException, TimeoutException


class TestSearchRetryContext:
    """Tests for SearchRetryContext class."""

    def test_init_sets_initial_values(self):
        """Should initialize with correct default values."""
        cfg = AgentConfig()
        ctx = SearchRetryContext(cfg)

        assert ctx.cfg is cfg
        assert ctx.attempt == 0
        assert ctx.delay == 0.5
        assert ctx.last_exception is None

    def test_increment_attempt(self):
        """Should increment attempt counter."""
        ctx = SearchRetryContext(AgentConfig())
        assert ctx.attempt == 0

        ctx.increment_attempt()
        assert ctx.attempt == 1

        ctx.increment_attempt()
        assert ctx.attempt == 2

    def test_is_retry_needed_returns_true_when_retries_available(self):
        """Should return True when retries are available."""
        ctx = SearchRetryContext(AgentConfig(search_retries=3))
        ctx.attempt = 2

        assert ctx.is_retry_needed() is True

    def test_is_retry_needed_returns_false_when_max_retries_reached(self):
        """Should return False when max retries reached."""
        ctx = SearchRetryContext(AgentConfig(search_retries=3))
        ctx.attempt = 3

        assert ctx.is_retry_needed() is False

    def test_get_delay_returns_current_delay(self):
        """Should return current delay value."""
        ctx = SearchRetryContext(AgentConfig())
        assert ctx.get_delay() == 0.5

        ctx.delay = 1.0
        assert ctx.get_delay() == 1.0

    def test_update_delay_applies_backoff(self):
        """Should apply exponential backoff to delay."""
        ctx = SearchRetryContext(AgentConfig())
        assert ctx.delay == 0.5

        ctx.update_delay(backoff_multiplier=1.8, max_delay=3.0)
        assert ctx.delay == 0.9  # 0.5 * 1.8

        ctx.update_delay(backoff_multiplier=1.8, max_delay=3.0)
        assert ctx.delay == 1.62  # 0.9 * 1.8

    def test_update_delay_respects_max_delay(self):
        """Should cap delay at max_delay."""
        ctx = SearchRetryContext(AgentConfig())
        ctx.delay = 2.0

        ctx.update_delay(backoff_multiplier=1.8, max_delay=3.0)
        assert ctx.delay == 3.0  # Capped at max

        ctx.update_delay(backoff_multiplier=1.8, max_delay=3.0)
        assert ctx.delay == 3.0  # Still capped


class TestSafeCloseClient:
    """Tests for safe_close_client function."""

    def test_safe_close_calls_close_method(self):
        """Should call close method if it exists."""
        mock_client = Mock()
        mock_client.close = Mock()

        safe_close_client(mock_client)

        mock_client.close.assert_called_once()

    def test_safe_close_handles_none_client(self):
        """Should handle None client gracefully."""
        safe_close_client(None)
        # Should not raise exception

    def test_safe_close_handles_client_without_close_method(self):
        """Should handle client without close method."""
        mock_client = Mock(spec=[])  # No close method

        safe_close_client(mock_client)
        # Should not raise exception

    def test_safe_close_handles_close_exception(self):
        """Should handle exception during close."""
        mock_client = Mock()
        mock_client.close.side_effect = Exception("Close failed")

        safe_close_client(mock_client)
        # Should not raise exception - error is logged


class TestHandleSearchException:
    """Tests for handle_search_exception function."""

    def test_timeout_exception_returns_true(self):
        """TimeoutException should indicate retry."""
        exc = TimeoutException("timeout")
        should_retry = handle_search_exception(exc, "query", 1, 3, 0.5)

        assert should_retry is True

    def test_ddgs_exception_returns_true(self):
        """DDGSException should indicate retry."""
        exc = DDGSException("search failed")
        should_retry = handle_search_exception(exc, "query", 1, 3, 0.5)

        assert should_retry is True

    def test_connection_error_returns_true(self):
        """ConnectionError should indicate retry."""
        exc = ConnectionError("connection failed")
        should_retry = handle_search_exception(exc, "query", 1, 3, 0.5)

        assert should_retry is True

    def test_os_error_returns_true(self):
        """OSError should indicate retry."""
        exc = OSError("os error")
        should_retry = handle_search_exception(exc, "query", 1, 3, 0.5)

        assert should_retry is True

    def test_unexpected_exception_returns_false(self):
        """Unexpected exceptions should indicate no retry."""
        exc = ValueError("unexpected error")
        should_retry = handle_search_exception(exc, "query", 1, 3, 0.5)

        assert should_retry is False

    def test_runtime_error_returns_false(self):
        """RuntimeError should indicate no retry."""
        exc = RuntimeError("runtime error")
        should_retry = handle_search_exception(exc, "query", 1, 3, 0.5)

        assert should_retry is False


class TestLogFinalFailure:
    """Tests for log_final_failure function."""

    def test_logs_failure_message(self, caplog):
        """Should log failure message."""
        log_final_failure("test query", 3)

        assert "Search failed after 3 attempts for 'test query'." in caplog.text

    def test_logs_with_different_retry_counts(self, caplog):
        """Should log correct retry count."""
        log_final_failure("query", 5)

        assert "Search failed after 5 attempts" in caplog.text


class TestShouldNotifyRetry:
    """Tests for should_notify_retry function."""

    def test_calls_notify_fn_when_provided(self):
        """Should call notify function when provided."""
        mock_notify = Mock()
        exc = Exception("test error")

        should_notify_retry(1, 3, mock_notify, 0.5, exc)

        mock_notify.assert_called_once_with(1, 3, 0.5, exc)

    def test_does_not_call_when_notify_fn_is_none(self):
        """Should not raise when notify function is None."""
        exc = Exception("test error")

        should_notify_retry(1, 3, None, 0.5, exc)
        # Should not raise exception

    def test_does_not_call_when_at_max_retries(self):
        """Should not notify when at max retries."""
        mock_notify = Mock()
        exc = Exception("test error")

        should_notify_retry(3, 3, mock_notify, 0.5, exc)

        mock_notify.assert_not_called()

    def test_calls_with_correct_parameters(self):
        """Should pass correct parameters to notify function."""
        mock_notify = Mock()
        exc = TimeoutException("timeout")

        should_notify_retry(2, 5, mock_notify, 1.2, exc)

        mock_notify.assert_called_once_with(2, 5, 1.2, exc)
