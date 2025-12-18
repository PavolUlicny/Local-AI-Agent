from __future__ import annotations

import pytest

from src import topic_utils as TU


def test_tail_turns_limits_history() -> None:
    turns = [(f"user{i}", f"assistant{i}") for i in range(4)]
    assert TU.tail_turns(turns, 2) == turns[-2:]
    assert TU.tail_turns(turns, 0) == []


def test_topic_brief_includes_summary_and_keywords() -> None:
    topic = TU.Topic(keywords={"renewable", "policy", "tax"})
    topic.summary = "Key findings on renewable incentives in 2024."
    brief = TU.topic_brief(topic, max_keywords=2)
    assert "Summary" in brief and "Keywords" in brief
    assert "renewable" in brief


def test_cosine_similarity_handles_matching_and_orthogonal_vectors() -> None:
    assert TU.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert TU.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "a,b,decay,expected",
    [
        ([1.0, 0.0], [0.0, 1.0], 0.5, [0.5, 0.5]),
        ([1.0], [0.0, 1.0, 2.0], 0.2, [0.2, 0.8, 1.6]),
    ],
)
def test_blend_embeddings_variants(a, b, decay, expected) -> None:
    blended = TU.blend_embeddings(a, b, decay=decay)
    assert blended == pytest.approx(expected)


def test_blend_embeddings_boundary_cases() -> None:
    # decay=0.0 -> new fully replaces existing
    prev = [1.0, 2.0]
    new = [3.0, 4.0]
    out = TU.blend_embeddings(prev, new, decay=0.0)
    assert out == pytest.approx(new)

    # decay near 0.9999 -> new barely affects
    out2 = TU.blend_embeddings(prev, new, decay=0.9999)
    assert out2[0] == pytest.approx(prev[0] * 0.9999 + new[0] * 0.0001)

    # new empty returns a copy of existing
    out3 = TU.blend_embeddings(prev, [], decay=0.5)
    assert out3 == prev
    assert out3 is not prev


def test_cosine_similarity_zero_length_and_zero_norm() -> None:
    # empty vectors
    assert TU.cosine_similarity([], [1.0]) == 0.0
    assert TU.cosine_similarity([1.0], []) == 0.0

    # zero-norm vectors
    assert TU.cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert TU.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_prune_keywords_keeps_most_frequent_terms() -> None:
    topic = TU.Topic(
        keywords={"alpha", "beta", "gamma"},
        turns=[("alpha beta", "alpha"), ("gamma", "beta beta")],
    )
    TU.prune_keywords(topic, max_keep=2)
    assert topic.keywords <= {"alpha", "beta"}

    def test_prune_keywords_tie_breaking_keeps_lexicographic() -> None:
        topic = TU.Topic(keywords={"b", "a", "c"}, turns=[])
        # no terms in turns -> all freqs are 0, so keep lexicographically smallest two: a,b
        TU.prune_keywords(topic, max_keep=2)
        assert topic.keywords == {"a", "b"}


class _StubContextChain:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs

    def invoke(self, inputs):  # noqa: ANN001 - simple stub for tests
        return self.outputs.pop(0) if self.outputs else "FOLLOW_UP"


def test_select_topic_uses_embeddings_when_keywords_absent() -> None:
    chain = _StubContextChain(["FOLLOW_UP"])
    topic = TU.Topic(
        keywords=set(),
        turns=[("prev question", "prev answer")],
        summary="Energy storage roadmap",
        embedding=[1.0, 0.0],
    )

    idx, turns, keywords = TU.select_topic(
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


def test_select_topic_fallbacks_to_latest_topic_on_followup() -> None:
    chain = _StubContextChain(["NEW_TOPIC"])
    first_topic = TU.Topic(keywords={"workstation"}, turns=[("what's new?", "answer1")])
    latest_topic = TU.Topic(keywords={"gpu"}, turns=[("what is the best gpu?", "answer2")])

    idx, turns, keywords = TU.select_topic(
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

    assert idx == 1
    assert turns == latest_topic.turns[-2:]
    assert keywords.issuperset({"cost", "gpu"})


def test_select_topic_raises_on_all_invoke_failures() -> None:
    class BadChain:
        def invoke(self, inputs):  # noqa: ANN001 - deliberate failure
            raise RuntimeError("chain-fail")

    topic = TU.Topic(keywords={"alpha"}, turns=[("u", "r")])
    with pytest.raises(RuntimeError):
        TU.select_topic(
            BadChain(),
            [topic],
            question="why",
            base_keywords={"alpha"},
            max_context_turns=2,
            current_datetime="2025-01-01 00:00:00 UTC",
            current_year="2025",
            current_month="01",
            current_day="01",
            question_embedding=None,
            embedding_threshold=0.1,
        )


def test_collect_prior_responses_truncates_and_joins() -> None:
    long_text = ("word " * 500).strip()
    topic = TU.Topic(turns=[("q1", long_text), ("q2", long_text)])

    # Small max_chars forces a small per-answer budget and truncation
    output = TU.collect_prior_responses(topic, limit=2, max_chars=400)
    assert "..." in output
    assert "---" in output
    assert "No prior answers" not in output
