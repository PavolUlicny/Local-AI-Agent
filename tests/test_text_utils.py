from __future__ import annotations

from src import text_utils as T


def test_pick_seed_query_skips_banned_entries() -> None:
    fallback = "original"
    seed_text = "none\nQUERY: Detailed emissions roadmap for 2030 goals"
    picked = T.pick_seed_query(seed_text, fallback)
    assert picked == "Detailed emissions roadmap for 2030 goals"

    empty_seed = "none"
    assert T.pick_seed_query(empty_seed, fallback) == fallback


def test_truncate_result_word_boundary_and_punctuation() -> None:
    text = "This is a long sentence that will be cut off. End." * 10
    truncated = T.truncate_result(text, max_chars=50)
    assert truncated.endswith("...")
    # ensure we don't end with punctuation before ellipsis
    assert not truncated[:-3].endswith(".,;:")


def test_is_context_length_error_indicators() -> None:
    phrases = [
        "context length exceeded",
        "token limit reached",
        "prompt too long",
        "sequence length exceeded",
        "input too long",
    ]
    for p in phrases:
        assert T.is_context_length_error(p)
