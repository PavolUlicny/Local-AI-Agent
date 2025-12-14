from __future__ import annotations

from src import keywords as K


def test_extract_keywords_keeps_accented_english_tokens() -> None:
    text = "Résumé-style reports accentuate naïve café décor."
    keywords = K.extract_keywords(text)
    assert {"résumé", "accentuate", "naïve", "café", "décor"}.issubset(keywords)


def test_looks_like_followup_detects_short_pronoun_question() -> None:
    assert K.looks_like_followup("how much does it cost?", {"gpu"}) is True
    assert K.looks_like_followup("explain gpu memory bandwidth", {"gpu"}) is False
