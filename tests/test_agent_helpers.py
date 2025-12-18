import pytest

from src.config import AgentConfig
from src.agent import Agent
from src import exceptions as _exceptions


class SimpleChain:
    def __init__(self, invoke_side_effect=None, stream_iterable=None):
        self._invoke_side_effect = invoke_side_effect
        self._invoke_called = 0
        self._stream_iterable = stream_iterable

    def invoke(self, inputs):
        if isinstance(self._invoke_side_effect, Exception):
            raise self._invoke_side_effect
        if isinstance(self._invoke_side_effect, list):
            val = self._invoke_side_effect[self._invoke_called]
            self._invoke_called += 1
            if isinstance(val, Exception):
                raise val
            return val
        return self._invoke_side_effect

    def stream(self, inputs):
        if isinstance(self._stream_iterable, Exception):
            raise self._stream_iterable
        for ch in self._stream_iterable or []:
            yield ch


def make_agent(tmp_path):
    cfg = AgentConfig()
    agent = Agent(cfg, output_stream=None, is_tty=False)
    return agent


def test_agent_wrapper_invoke_chain_safe_success():
    agent = make_agent(None)
    agent.chains = {"ok": SimpleChain(invoke_side_effect="RESULT")}
    res = agent._invoke_chain_safe("ok", {})
    assert res == "RESULT"


def test_invoke_chain_safe_not_found_propagates():
    agent = make_agent(None)
    err = _exceptions.ResponseError("Model not found: foo")
    agent.chains = {"bad": SimpleChain(invoke_side_effect=err)}
    with pytest.raises(_exceptions.ResponseError):
        agent._invoke_chain_safe("bad", {})


def test_invoke_chain_safe_context_rebuild_then_retry(monkeypatch):
    agent = make_agent(None)
    # first call raises context-length error, second returns value
    err = _exceptions.ResponseError("context length exceeded")
    agent.chains = {"retry": SimpleChain(invoke_side_effect=[err, "RECOVERED"])}

    # stub out rebuild so it doesn't perform heavy operations
    def fake_reduce(stage_key, label):
        agent.rebuild_counts["retry"] = agent.rebuild_counts.get("retry", 0) + 1

    monkeypatch.setattr(agent, "_reduce_context_and_rebuild", fake_reduce)
    val = agent._invoke_chain_safe("retry", {}, rebuild_key="retry")
    assert val == "RECOVERED"


def test_decide_should_search_and_generate_seed(monkeypatch):
    agent = make_agent(None)
    # stub _inputs to not rely on build_inputs
    monkeypatch.setattr(agent, "_inputs", lambda *a, **k: {"q": "x"})
    agent.chains = {
        "search_decision": SimpleChain(invoke_side_effect="SEARCH"),
        "seed": SimpleChain(invoke_side_effect="Some seed\n"),
    }

    # create a minimal QueryContext via _build_query_context path
    # use a simple user query; _build_query_context needs embedding_client, topics etc.
    # for this test we only validate the helper wiring so build a fake ctx object
    class Ctx:
        current_datetime = "now"
        current_year = "2025"
        current_month = "12"
        current_day = "17"
        conversation_text = "conv"
        prior_responses_text = "prior"

    ctx = Ctx()
    should = agent._decide_should_search(ctx, "hi", "prior")
    assert should is True
    seed = agent._generate_search_seed(ctx, "hi", "prior")
    assert isinstance(seed, str) and seed


def test_run_search_rounds_and_update_topics(monkeypatch):
    agent = make_agent(None)

    # stub build_search_orchestrator to return object with run method
    class Orchestrator:
        def run(self, **kwargs):
            return ["r1", "r2"], set("k1")

    monkeypatch.setattr(agent, "_build_search_orchestrator", lambda: Orchestrator())

    class Ctx:
        current_datetime = "now"
        current_year = "2025"
        current_month = "12"
        current_day = "17"
        conversation_text = "conv"
        prior_responses_text = "prior"

    ctx = Ctx()
    aggregated, kws = agent._run_search_rounds(ctx, "q", True, "q", None, None, set())
    assert aggregated == ["r1", "r2"]
    assert kws == set("k1")
    # topic manager update wrapper
    agent.topic_manager = type("T", (), {"update_topics": lambda self, **kwargs: 5})()
    idx = agent._update_topics(None, set(), [], [], "q", "resp", None)
    assert idx == 5


def test_update_topics_normalizes_question_keywords_in_agent(monkeypatch):
    agent = make_agent(None)

    captured = {}

    class TM:
        def update_topics(self, **kwargs):
            captured.update(kwargs)
            return 7

    agent.topic_manager = TM()
    # pass a list with duplicate values to ensure conversion to set
    idx = agent._update_topics(None, set(), ["a", "b", "a"], [], "q", "resp", None)
    assert idx == 7
    assert isinstance(captured.get("question_keywords"), set)
    assert captured.get("question_keywords") == {"a", "b"}


def test_generate_and_stream_response_success_and_stream_error(monkeypatch):
    agent = make_agent(None)
    # chain that streams chunks
    agent.chains = {"response": SimpleChain(stream_iterable=["a", "b\n"])}
    out = []
    text = agent._generate_and_stream_response({}, "response", one_shot=True, write_fn=lambda s: out.append(s))
    assert text == "ab\n"
    assert out == ["a", "b\n"]

    # chain whose stream raises ResponseError with 'not found' => returns None
    err = _exceptions.ResponseError("Model not found: assistant")
    agent.chains = {"response": SimpleChain(stream_iterable=err)}
    res = agent._generate_and_stream_response({}, "response", one_shot=True, write_fn=lambda s: None)
    assert res is None


def test_build_resp_inputs(monkeypatch):
    agent = make_agent(None)

    # stub _inputs to observe parameters
    def fake_inputs(
        current_datetime, current_year, current_month, current_day, conversation_text, user_query, **kwargs
    ):
        d = {
            current_datetime: current_datetime,
            current_year: current_year,
            conversation_text: conversation_text,
            user_query: user_query,
        }
        d.update(kwargs)
        return d

    monkeypatch.setattr(agent, "_inputs", fake_inputs)
    resp_inputs, chain = agent._build_resp_inputs("now", "2025", "12", "17", "c", "q", True, "prior", "search res")
    assert chain == "response"
    assert resp_inputs["search_results"] == "search res"

    resp_inputs2, chain2 = agent._build_resp_inputs("now", "2025", "12", "17", "c", "q", False, "prior", None)
    assert chain2 == "response_no_search"
    assert "search_results" not in resp_inputs2
