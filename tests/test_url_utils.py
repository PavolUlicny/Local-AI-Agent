from __future__ import annotations

from src import url_utils as U


import pytest


@pytest.mark.parametrize(
    "input_url,expected",
    [
        ("HTTPS://WWW.Example.com/path/?q=1#frag", "https://example.com/path?q=1"),
        ("example.com/foo/bar/", "http://example.com/foo/bar"),
    ],
)
def test_canonicalize_url_variants(input_url: str, expected: str) -> None:
    assert U.canonicalize_url(input_url) == expected
