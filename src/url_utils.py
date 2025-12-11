"""URL normalization helpers."""

from __future__ import annotations

import re
from urllib.parse import urlparse, urlunparse


def _canonicalize_url(url: str) -> str:
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
        scheme = (parsed.scheme or "http").lower()
        netloc = (parsed.netloc or "").lower()
        path = parsed.path or ""
        if netloc.startswith("www."):
            netloc = netloc[4:]
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        if netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]
        if path.endswith("/") and path != "/":
            path = path.rstrip("/")
        return urlunparse((scheme, netloc, path, "", parsed.query, ""))
    except Exception:
        return url.strip().rstrip("/")


__all__ = ["_canonicalize_url"]
