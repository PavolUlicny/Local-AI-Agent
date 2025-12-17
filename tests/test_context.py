from types import SimpleNamespace

import src.context as context_mod
from src.exceptions import ResponseError


class DummyEmbedding:
    def __init__(self, vec=None):
        self._vec = vec or [0.1, 0.2, 0.3]

    def embed(self, text: str):
        return self._vec


def test_build_query_context_happy_path(monkeypatch):
    # Patch select_topic to return a normal topic selection
    def fake_select_topic(*args, **kwargs):
        return 0, [("user1", "resp1")], {"kw1", "kw2"}

    monkeypatch.setattr(context_mod, "_topic_utils_mod", context_mod._topic_utils_mod)
    monkeypatch.setattr(context_mod._topic_utils_mod, "select_topic", fake_select_topic)

    cfg = SimpleNamespace(max_context_turns=2, embedding_similarity_threshold=0.5)
    from src.topic_utils import Topic

    agent = SimpleNamespace(
        cfg=cfg,
        embedding_client=DummyEmbedding(),
        topics=[Topic(keywords={"kw1"}, embedding=[0.1], turns=[("u", "r")], summary="s")],
        chains={"context": "fakechain"},
    )
    # minimal char budget helper used by build_query_context
    agent._char_budget = lambda x: 10000

    ctx = context_mod.build_query_context(agent, "hello world")
    assert ctx.selected_topic_index == 0
    assert isinstance(ctx.conversation_text, str)
    assert isinstance(ctx.prior_responses_text, str)


def test_build_query_context_model_missing(monkeypatch):
    # Simulate select_topic raising ResponseError with 'not found' to trigger missing-model handling
    def raise_not_found(*args, **kwargs):
        raise ResponseError("Model Not Found: Robot model not found")

    monkeypatch.setattr(context_mod, "_topic_utils_mod", context_mod._topic_utils_mod)
    monkeypatch.setattr(context_mod._topic_utils_mod, "select_topic", raise_not_found)

    cfg = SimpleNamespace(max_context_turns=2, embedding_similarity_threshold=0.5, robot_model="robot:1")

    class A:
        def __init__(self):
            self._last_error = None

        def _mark_error(self, msg: str):
            self._last_error = msg
            return msg

    agent = A()
    agent.cfg = cfg
    agent.embedding_client = DummyEmbedding()
    agent.topics = []
    agent.chains = {"context": "fakechain"}
    agent._char_budget = lambda x: 10000

    ctx = context_mod.build_query_context(agent, "q")
    assert ctx.selected_topic_index is None
    assert ctx.conversation_text == "No prior relevant conversation."
    assert agent._last_error is not None
