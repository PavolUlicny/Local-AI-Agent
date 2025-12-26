from __future__ import annotations

from src import keywords as K


def test_extract_keywords_filters_stopwords_and_digits() -> None:
    s = "What is the price of 2025 model?"
    ks = K.extract_keywords(s)
    assert "price" in ks
    assert "what" not in ks
    assert not any(token.isdigit() for token in ks)


def test_extract_keywords_keeps_accented_english_tokens() -> None:
    text = "Résumé-style reports accentuate naïve café décor."
    keywords = K.extract_keywords(text)
    assert {"résumé", "accentuate", "naïve", "café", "décor"}.issubset(keywords)


def test_is_relevant_returns_false_when_no_topic_keywords() -> None:
    """Test is_relevant returns False when topic_keywords is empty."""
    assert K.is_relevant("any text here", set()) is False
    assert K.is_relevant("", set()) is False


def test_is_relevant_matches_keyword_overlap() -> None:
    """Test is_relevant detects keyword overlap."""
    topic_keywords = {"python", "programming", "code"}

    # Should match - has 'python'
    assert K.is_relevant("Python tutorial for beginners", topic_keywords) is True

    # Should match - has 'programming'
    assert K.is_relevant("Learn programming concepts", topic_keywords) is True

    # Should not match - no overlap
    assert K.is_relevant("JavaScript fundamentals", topic_keywords) is False

    # Should not match - stopwords filtered out
    assert K.is_relevant("the and for", {"the", "and"}) is False
