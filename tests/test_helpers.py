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


def test_summarize_answer_uses_first_sentences() -> None:
    text = "Sentence one explains the topic. Sentence two adds nuance. Sentence three is extra."
    summary = H._summarize_answer(text, max_chars=120)
    assert "Sentence one" in summary and "Sentence two" in summary


def test_topic_brief_includes_summary_and_keywords() -> None:
    topic = H.Topic(keywords={"renewable", "policy", "tax"})
    topic.summary = "Key findings on renewable incentives in 2024."
    brief = H._topic_brief(topic, max_keywords=2)
    assert "Summary" in brief and "Keywords" in brief
    assert "renewable" in brief
