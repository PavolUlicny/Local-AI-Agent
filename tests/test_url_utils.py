from __future__ import annotations

from src import url_utils as U


def test_canonicalize_url_normalizes_scheme_and_query() -> None:
    url = "HTTPS://WWW.Example.com/path/?q=1#frag"
    assert U._canonicalize_url(url) == "https://example.com/path?q=1"


def test_canonicalize_url_handles_schemeless_input() -> None:
    url = "example.com/foo/bar/"
    assert U._canonicalize_url(url) == "http://example.com/foo/bar"
