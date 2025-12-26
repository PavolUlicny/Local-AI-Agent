from __future__ import annotations

import pytest

from src import url_utils as U


def test_canonicalize_basic_cases() -> None:
    assert U.canonicalize_url("") == ""
    assert U.canonicalize_url("//example.com/path") == "http://example.com/path"
    assert U.canonicalize_url("http://www.Example.COM/") == "http://example.com"
    assert U.canonicalize_url("example.com/dir/") == "http://example.com/dir"


@pytest.mark.parametrize(
    "input_url,expected",
    [
        ("HTTPS://WWW.Example.com/path/?q=1#frag", "https://example.com/path?q=1"),
        ("example.com/foo/bar/", "http://example.com/foo/bar"),
    ],
)
def test_canonicalize_url_variants(input_url: str, expected: str) -> None:
    assert U.canonicalize_url(input_url) == expected


def test_disallowed_schemes_blocked(caplog) -> None:
    """Test that disallowed URL schemes are blocked with warning."""
    caplog.clear()
    caplog.set_level("WARNING")

    # Test various disallowed schemes
    assert U.canonicalize_url("ftp://example.com/file.txt") == ""
    assert U.canonicalize_url("file:///etc/passwd") == ""
    assert U.canonicalize_url("javascript:alert(1)") == ""
    assert U.canonicalize_url("data:text/html,<script>alert(1)</script>") == ""

    # Should have logged warnings
    assert "Blocked URL with disallowed scheme" in caplog.text


def test_port_normalization() -> None:
    """Test that default ports are removed from URLs."""
    # HTTP default port :80 should be removed
    assert U.canonicalize_url("http://example.com:80/path") == "http://example.com/path"

    # HTTPS default port :443 should be removed
    assert U.canonicalize_url("https://example.com:443/path") == "https://example.com/path"

    # Non-default ports should be kept
    assert U.canonicalize_url("http://example.com:8080/path") == "http://example.com:8080/path"
    assert U.canonicalize_url("https://example.com:8443/path") == "https://example.com:8443/path"


def test_invalid_url_handling(caplog, monkeypatch) -> None:
    """Test that invalid URLs are handled gracefully."""
    caplog.clear()
    caplog.set_level("DEBUG")

    # Mock urlparse to raise TypeError to test exception handling
    def raise_type_error(*args, **kwargs):
        raise TypeError("Invalid URL type")

    monkeypatch.setattr("src.url_utils.urlparse", raise_type_error)

    # Should return empty string without crashing and log the error
    result = U.canonicalize_url("http://example.com")
    assert result == ""

    # Should have logged debug message about failure
    assert "URL canonicalization failed" in caplog.text


class TestSSRFProtection:
    """Test SSRF (Server-Side Request Forgery) attack protection."""

    def test_blocks_localhost_variations(self, caplog) -> None:
        """Test that localhost variations are blocked."""
        caplog.clear()
        caplog.set_level("WARNING")

        # Direct localhost references
        assert U.canonicalize_url("http://localhost/admin") == ""
        assert U.canonicalize_url("http://localhost:8080/api") == ""

        # Loopback IPs (127.0.0.0/8)
        assert U.canonicalize_url("http://127.0.0.1/") == ""
        assert U.canonicalize_url("http://127.1.1.1/") == ""
        assert U.canonicalize_url("http://127.255.255.255/") == ""

        # Should have logged warnings
        assert "Blocked SSRF attempt" in caplog.text

    def test_blocks_private_ip_ranges(self, caplog) -> None:
        """Test that private IP ranges (RFC 1918) are blocked."""
        caplog.clear()
        caplog.set_level("WARNING")

        # 10.0.0.0/8
        assert U.canonicalize_url("http://10.0.0.1/") == ""
        assert U.canonicalize_url("http://10.255.255.255/internal") == ""

        # 172.16.0.0/12
        assert U.canonicalize_url("http://172.16.0.1/") == ""
        assert U.canonicalize_url("http://172.31.255.255/") == ""

        # 192.168.0.0/16
        assert U.canonicalize_url("http://192.168.0.1/") == ""
        assert U.canonicalize_url("http://192.168.255.255/router") == ""

        assert "Blocked SSRF attempt" in caplog.text

    def test_blocks_link_local(self, caplog) -> None:
        """Test that link-local addresses (169.254.0.0/16) are blocked."""
        caplog.clear()
        caplog.set_level("WARNING")

        assert U.canonicalize_url("http://169.254.0.1/") == ""
        assert U.canonicalize_url("http://169.254.169.254/latest/meta-data") == ""

        assert "Blocked SSRF attempt" in caplog.text

    def test_blocks_cloud_metadata_endpoints(self, caplog) -> None:
        """Test that cloud metadata endpoints are blocked."""
        caplog.clear()
        caplog.set_level("WARNING")

        # AWS/GCP/Azure metadata IP
        assert U.canonicalize_url("http://169.254.169.254/latest/meta-data/") == ""

        # GCP metadata hostname
        assert U.canonicalize_url("http://metadata.google.internal/computeMetadata/") == ""
        assert U.canonicalize_url("http://METADATA.GOOGLE.INTERNAL/v1/") == ""  # Case insensitive

        # Generic metadata hostname
        assert U.canonicalize_url("http://metadata/") == ""

        # AWS instance-data alternative
        assert U.canonicalize_url("http://instance-data/latest/") == ""

        assert caplog.text.count("Blocked SSRF attempt") >= 4

    def test_blocks_ipv6_loopback_and_private(self, caplog) -> None:
        """Test that IPv6 loopback and private addresses are blocked."""
        caplog.clear()
        caplog.set_level("WARNING")

        # IPv6 loopback (::1)
        assert U.canonicalize_url("http://[::1]/") == ""
        assert U.canonicalize_url("http://[::1]:8080/admin") == ""

        # IPv6 unique local addresses (fc00::/7)
        assert U.canonicalize_url("http://[fc00::1]/") == ""
        assert U.canonicalize_url("http://[fd00::1]/") == ""

        # IPv6 link-local (fe80::/10)
        assert U.canonicalize_url("http://[fe80::1]/") == ""

        assert "Blocked SSRF attempt" in caplog.text

    def test_allows_public_urls(self) -> None:
        """Test that legitimate public URLs are allowed."""
        # Should allow normal public domains
        assert U.canonicalize_url("http://example.com") == "http://example.com"
        assert U.canonicalize_url("https://google.com/search") == "https://google.com/search"
        assert U.canonicalize_url("http://api.github.com/repos") == "http://api.github.com/repos"

        # Should allow public IPs (non-RFC 1918)
        assert U.canonicalize_url("http://8.8.8.8/") == "http://8.8.8.8"
        assert U.canonicalize_url("http://1.1.1.1/") == "http://1.1.1.1"

    def test_is_safe_url_function_directly(self) -> None:
        """Test the is_safe_url function directly."""
        # Safe URLs
        assert U.is_safe_url("http://example.com") is True
        assert U.is_safe_url("https://google.com/search") is True
        assert U.is_safe_url("http://8.8.8.8/") is True

        # Unsafe URLs
        assert U.is_safe_url("http://localhost/") is False
        assert U.is_safe_url("http://127.0.0.1/") is False
        assert U.is_safe_url("http://10.0.0.1/") is False
        assert U.is_safe_url("http://192.168.1.1/") is False
        assert U.is_safe_url("http://169.254.169.254/") is False
        assert U.is_safe_url("http://metadata.google.internal/") is False
        assert U.is_safe_url("http://[::1]/") is False
        assert U.is_safe_url("http://[fc00::1]/") is False

    @pytest.mark.parametrize(
        "dangerous_url",
        [
            "http://127.0.0.1/admin",
            "http://localhost:8080/internal",
            "http://10.0.0.1/secret",
            "http://172.16.0.1/config",
            "http://192.168.1.1/router",
            "http://169.254.169.254/metadata",
            "http://metadata.google.internal/computeMetadata",
            "http://metadata/",
            "http://[::1]/",
            "http://[fc00::1]/",
            "http://[fe80::1]/",
        ],
    )
    def test_parametrized_dangerous_urls_blocked(self, dangerous_url: str, caplog) -> None:
        """Test that various dangerous URLs are blocked."""
        caplog.clear()
        caplog.set_level("WARNING")

        result = U.canonicalize_url(dangerous_url)
        assert result == "", f"Expected {dangerous_url} to be blocked, but got {result}"

        # Should have logged a warning
        assert "Blocked SSRF attempt" in caplog.text or "Blocked URL" in caplog.text
