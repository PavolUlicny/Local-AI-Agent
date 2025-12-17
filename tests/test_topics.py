from types import SimpleNamespace


def test_update_topics_forwards_arguments():
    calls = {}

    class FakeTopicManager:
        def update_topics(self, **kwargs):
            calls.update(kwargs)
            return 3

    fake_agent = SimpleNamespace(
        topic_manager=FakeTopicManager(),
        topics=["t1", "t2"],
    )

    import src.topics as topics_mod

    result = topics_mod.update_topics(
        fake_agent,
        selected_topic_index=1,
        topic_keywords={"a", "b"},
        question_keywords=["q"],
        aggregated_results=["r1"],
        user_query="u",
        response_text="resp",
        question_embedding=None,
    )

    assert result == 3
    # Basic forwarding checks
    assert calls["topics"] == fake_agent.topics
    assert calls["selected_topic_index"] == 1
    assert calls["user_query"] == "u"
