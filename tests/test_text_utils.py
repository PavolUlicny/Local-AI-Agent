from __future__ import annotations

from src import text_utils as T


def test_pick_seed_query_skips_banned_entries() -> None:
    fallback = "original"
    seed_text = "none\nQUERY: Detailed emissions roadmap for 2030 goals"
    picked = T.pick_seed_query(seed_text, fallback)
    assert picked == "Detailed emissions roadmap for 2030 goals"

    empty_seed = "none"
    assert T.pick_seed_query(empty_seed, fallback) == fallback


def test_summarize_answer_uses_first_sentences() -> None:
    text = "Sentence one explains the topic. Sentence two adds nuance. Sentence three is extra."
    summary = T.summarize_answer(text, max_chars=120)
    assert "Sentence one" in summary and "Sentence two" in summary


def test_context_regex_accepts_spaced_follow_up() -> None:
    validated = T.regex_validate("follow up", T.PATTERN_CONTEXT, "NEW_TOPIC")
    assert validated == "FOLLOW UP"
    assert T.normalize_context_decision(validated) == "FOLLOW_UP"


def test_truncate_result_word_boundary_and_punctuation() -> None:
    text = "This is a long sentence that will be cut off. End." * 10
    truncated = T.truncate_result(text, max_chars=50)
    assert truncated.endswith("...")
    # ensure we don't end with punctuation before ellipsis
    assert not truncated[:-3].endswith(".,;:")
