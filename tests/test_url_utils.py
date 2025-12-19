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
