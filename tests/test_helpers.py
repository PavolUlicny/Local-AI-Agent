from __future__ import annotations

import pytest

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


def test_cosine_similarity_handles_matching_and_orthogonal_vectors() -> None:
    assert H._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert H._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_blend_embeddings_respects_decay() -> None:
    blended = H._blend_embeddings([1.0, 0.0], [0.0, 1.0], decay=0.5)
    assert blended == pytest.approx([0.5, 0.5])


class _StubContextChain:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs

    def invoke(self, inputs):  # noqa: ANN001 - helper for tests
        return self.outputs.pop(0) if self.outputs else "FOLLOW_UP"


def test_select_topic_uses_embeddings_when_keywords_absent() -> None:
    chain = _StubContextChain(["FOLLOW_UP"])
    topic = H.Topic(
        keywords=set(),
        turns=[("prev question", "prev answer")],
        summary="Energy storage roadmap",
        embedding=[1.0, 0.0],
    )

    idx, turns, keywords = H._select_topic(
        chain,
        [topic],
        question="Follow-up question",
        base_keywords=set(),
        max_context_turns=2,
        current_datetime="2024-01-01 00:00:00 UTC",
        current_year="2024",
        current_month="01",
        current_day="01",
        question_embedding=[1.0, 0.0],
        embedding_threshold=0.1,
    )

    assert idx == 0
    assert turns  # recent history returned
    assert keywords == set()


def test_looks_like_followup_detects_short_pronoun_question() -> None:
    assert H._looks_like_followup("how much does it cost?", {"gpu"}) is True
    assert H._looks_like_followup("explain gpu memory bandwidth", {"gpu"}) is False


def test_select_topic_fallbacks_to_latest_topic_on_followup() -> None:
    chain = _StubContextChain(["NEW_TOPIC"])  # unused but required parameter
    first_topic = H.Topic(keywords={"workstation"}, turns=[("what's new?", "answer1")])
    latest_topic = H.Topic(keywords={"gpu"}, turns=[("what is the best gpu?", "answer2")])

    idx, turns, keywords = H._select_topic(
        chain,
        [first_topic, latest_topic],
        question="how much does it cost?",
        base_keywords={"cost"},
        max_context_turns=2,
        current_datetime="2025-01-01 00:00:00 UTC",
        current_year="2025",
        current_month="01",
        current_day="01",
        question_embedding=None,
        embedding_threshold=0.5,
    )

    assert idx == 1  # picks most recent topic
    assert turns == latest_topic.turns[-2:]
    assert keywords.issuperset({"cost", "gpu"})
