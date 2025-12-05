from __future__ import annotations

from src import helpers as H


def test_canonicalize_url_normalizes_scheme_and_query() -> None:
    url = "HTTPS://WWW.Example.com/path/?q=1#frag"
    assert H._canonicalize_url(url) == "https://example.com/path?q=1"


def test_canonicalize_url_handles_schemeless_input() -> None:
    url = "example.com/foo/bar/"
    assert H._canonicalize_url(url) == "http://example.com/foo/bar"


def test_extract_keywords_keeps_accented_english_tokens() -> None:
    text = "Résumé-style reports accentuate naïve café décor."
    keywords = H._extract_keywords(text)
    assert {"résumé", "accentuate", "naïve", "café", "décor"}.issubset(keywords)


def test_tail_turns_limits_history() -> None:
    turns = [(f"user{i}", f"assistant{i}") for i in range(4)]
    assert H._tail_turns(turns, 2) == turns[-2:]
    assert H._tail_turns(turns, 0) == []


def test_pick_seed_query_skips_banned_entries() -> None:
    fallback = "original"
    seed_text = "none\nQUERY: Detailed emissions roadmap for 2030 goals"
    picked = H._pick_seed_query(seed_text, fallback)
    assert picked == "Detailed emissions roadmap for 2030 goals"

    empty_seed = "none"
    assert H._pick_seed_query(empty_seed, fallback) == fallback
