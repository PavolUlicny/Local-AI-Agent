"""URL normalization helpers with SSRF protection."""

from __future__ import annotations

import ipaddress
import logging
import re
from urllib.parse import urlparse, urlunparse

# Whitelist of allowed URL schemes to prevent SSRF and other attacks
ALLOWED_SCHEMES = {"http", "https"}

# Blocked IP ranges to prevent SSRF attacks on private networks
BLOCKED_IP_RANGES = [
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback
    ipaddress.ip_network("10.0.0.0/8"),  # Private (RFC 1918)
    ipaddress.ip_network("172.16.0.0/12"),  # Private (RFC 1918)
    ipaddress.ip_network("192.168.0.0/16"),  # Private (RFC 1918)
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local (RFC 3927)
    ipaddress.ip_network("::1/128"),  # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),  # IPv6 unique local addresses (RFC 4193)
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
]

# Blocked hostnames to prevent access to cloud metadata endpoints and localhost
BLOCKED_HOSTNAMES = {
    "localhost",  # Localhost hostname
    "metadata.google.internal",  # GCP metadata
    "169.254.169.254",  # AWS/GCP/Azure metadata IP
    "metadata",  # Common metadata hostname
    "instance-data",  # AWS instance metadata alternative
}


def is_safe_url(parsed_url: str) -> bool:
    """Check if URL is safe from SSRF attacks.

    Args:
        parsed_url: URL string to check

    Returns:
        True if URL is safe (public), False if it targets private/blocked resources

    Checks:
        1. Blocked hostnames (cloud metadata endpoints)
        2. Private IP ranges (RFC 1918, link-local, loopback)
        3. IPv6 private/link-local addresses
    """
    try:
        parsed = urlparse(parsed_url)
        hostname = parsed.hostname

        if not hostname:
            return True  # No hostname, can't be SSRF

        # Check blocked hostnames (case-insensitive)
        hostname_lower = hostname.lower()
        if hostname_lower in BLOCKED_HOSTNAMES:
            logging.warning("Blocked SSRF attempt to blocked hostname: %s", hostname)
            return False

        # Try to parse as IP address and check against blocked ranges
        try:
            ip = ipaddress.ip_address(hostname)
            for blocked_range in BLOCKED_IP_RANGES:
                if ip in blocked_range:
                    logging.warning("Blocked SSRF attempt to private IP range: %s in %s", ip, blocked_range)
                    return False
        except ValueError:
            # Not an IP address, it's a domain name
            # Domain names are allowed unless explicitly blocked above
            pass

        return True

    except Exception as e:
        # If we can't parse it, reject it to be safe
        logging.debug("URL safety check failed for '%s': %s", parsed_url[:100], e)
        return False


def canonicalize_url(url: str) -> str:
    """Canonicalize URL with security validation.

    Args:
        url: URL to canonicalize

    Returns:
        Canonicalized URL string, or empty string if invalid/dangerous
    """
    if not url:
        return ""
    try:
        raw = url.strip()
        if raw.startswith("//"):
            raw = "http:" + raw
        parsed = urlparse(raw)
        if not parsed.netloc and parsed.path:
            first_segment = parsed.path.split("/", 1)[0]
            looks_like_host_port = bool(re.match(r"^[A-Za-z0-9._-]+:\d+$", first_segment))
            is_localhost = first_segment.startswith("localhost")
            is_ipv4 = bool(re.match(r"^\d+\.\d+\.\d+\.\d+$", first_segment))
            if "://" not in raw and ("." in first_segment or looks_like_host_port or is_localhost or is_ipv4):
                parsed = urlparse("http://" + raw)

        # Security: validate scheme before processing
        scheme = (parsed.scheme or "http").lower()
        if scheme not in ALLOWED_SCHEMES:
            logging.warning("Blocked URL with disallowed scheme '%s': %s", scheme, url[:100])
            return ""

        # Security: check for SSRF attacks (private IPs, cloud metadata endpoints)
        if not is_safe_url(urlunparse((scheme, parsed.netloc or "", parsed.path or "", "", "", ""))):
            logging.warning("Blocked SSRF attempt: %s", url[:100])
            return ""

        netloc = (parsed.netloc or "").lower()
        path = parsed.path or ""
        # Normalize root path to empty string so canonical form has no trailing slash
        if path == "/":
            path = ""
        if netloc.startswith("www."):
            netloc = netloc[4:]
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        if netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]
        if path.endswith("/"):
            path = path.rstrip("/")
        return urlunparse((scheme, netloc, path, "", parsed.query, ""))
    except (ValueError, TypeError, AttributeError) as e:
        # URL parsing/processing failed
        logging.debug("URL canonicalization failed for '%s': %s", url[:100], e)
        return ""


__all__ = ["canonicalize_url", "is_safe_url"]
