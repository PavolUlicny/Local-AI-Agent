from __future__ import annotations

from types import SimpleNamespace

from src.agent import Agent
from src.config import AgentConfig
from src.exceptions import ResponseError
from src import text_utils as T


class _FakeCtx:
    def __init__(self):
        self.current_datetime = "d"
        self.current_year = "y"
        self.current_month = "m"
        self.current_day = "dd"
        self.question_keywords = []
        self.question_embedding = None
        self.selected_topic_index = None
        self.recent_history = []
        self.topic_keywords = set()
        self.topic_brief_text = ""
        self.topic_embedding_current = None
        self.recent_for_prompt = []
        self.conversation_text = "c"
        self.prior_responses_text = "p"


def test_search_decision_rebuild_then_retry(monkeypatch):
    cfg = AgentConfig()
    agent = Agent(cfg)

    # stub query context builder to avoid heavy logic (patch agent-bound reference)
    monkeypatch.setattr("src.agent.build_query_context", lambda self, q: _FakeCtx())

    calls = {"reduced": 0, "run_search": 0}

    def reduce(key, label):
        calls["reduced"] += 1

    # Simulate _decide_should_search raising a context-length ResponseError
    def decide_raise(ctx, q, p):
        raise ResponseError("Context length exceeded")

    agent._decide_should_search = decide_raise
    # the direct chain invoked during retry should return SEARCH
    agent.chains["search_decision"] = SimpleNamespace(invoke=lambda inputs: "SEARCH")
    monkeypatch.setattr(agent, "_reduce_context_and_rebuild", reduce)

    # patch out downstream work: generate_search_seed, run_search_rounds, build_resp, response
    monkeypatch.setattr(agent, "_generate_search_seed", lambda ctx, q, p: q)

    def run_search(ctx, user_query, should_search, primary, qemb, temb, kws):
        calls["run_search"] += 1
        return (["r1"], set())

    monkeypatch.setattr(agent, "_run_search_rounds", run_search)
    monkeypatch.setattr(agent, "_build_resp_inputs", lambda *a, **k: ({}, "response"))
    monkeypatch.setattr(agent, "_generate_and_stream_response", lambda *a, **k: "ans")

    res = agent._handle_query_core("q", one_shot=True)
    assert res == "ans"
    assert calls["reduced"] == 1
    assert calls["run_search"] == 1


def test_search_decision_rebuild_cap_defaults_to_no_search(monkeypatch):
    cfg = AgentConfig()
    agent = Agent(cfg)

    monkeypatch.setattr("src.agent.build_query_context", lambda self, q: _FakeCtx())

    # Prevent the initial reset from clearing our prepared counts
    monkeypatch.setattr(agent, "_reset_rebuild_counts", lambda: None)
    # set rebuild count at cap so it shouldn't attempt to rebuild
    agent.rebuild_counts["search_decision"] = T.MAX_REBUILD_RETRIES

    # Simulate _decide_should_search raising a context-length ResponseError
    def decide_raise2(ctx, q, p):
        raise ResponseError("Context length exceeded")

    agent._decide_should_search = decide_raise2

    # ensure _run_search_rounds is not called
    def bad_run(*a, **k):
        raise AssertionError("run_search_rounds should not be called when search decision caps out")

    monkeypatch.setattr(agent, "_run_search_rounds", bad_run)
    monkeypatch.setattr(agent, "_build_resp_inputs", lambda *a, **k: ({}, "response_no_search"))
    monkeypatch.setattr(agent, "_generate_and_stream_response", lambda *a, **k: "ans_no_search")

    res = agent._handle_query_core("q", one_shot=True)
    assert res == "ans_no_search"


def test_seed_generation_retries_and_fallback(monkeypatch):
    cfg = AgentConfig()
    agent = Agent(cfg)

    monkeypatch.setattr("src.agent.build_query_context", lambda self, q: _FakeCtx())

    # make _generate_search_seed raise, then have chains['seed'].invoke return a seed text
    def raise_seed(ctx, q, p):
        raise ResponseError("Context length exceeded")

    agent._generate_search_seed = raise_seed

    class SeedChain:
        def invoke(self, inputs):
            return "SEED: found candidate"

    agent.chains["seed"] = SeedChain()

    calls = {"reduced": 0, "run_search": None}

    monkeypatch.setattr(
        agent, "_reduce_context_and_rebuild", lambda k, label: calls.update({"reduced": calls.get("reduced", 0) + 1})
    )

    def run_search(ctx, user_query, should_search, primary, qemb, temb, kws):
        calls["run_search"] = primary
        return ([], set())

    monkeypatch.setattr(agent, "_run_search_rounds", run_search)
    monkeypatch.setattr(agent, "_build_resp_inputs", lambda *a, **k: ({}, "response"))
    monkeypatch.setattr(agent, "_generate_and_stream_response", lambda *a, **k: "ans")

    res = agent._handle_query_core("q", one_shot=True)
    assert res == "ans"
    assert calls["reduced"] == 1
    assert calls["run_search"] == "found candidate"


def test_seed_generation_rebuild_cap_uses_original_query(monkeypatch):
    cfg = AgentConfig()
    agent = Agent(cfg)

    monkeypatch.setattr("src.agent.build_query_context", lambda self, q: _FakeCtx())

    # Prevent reset and set seed rebuild at cap
    monkeypatch.setattr(agent, "_reset_rebuild_counts", lambda: None)
    agent._generate_search_seed = lambda ctx, q, p: (_ for _ in ()).throw(ResponseError("Context length exceeded"))
    agent.rebuild_counts["seed"] = T.MAX_REBUILD_RETRIES

    # ensure run_search receives original query
    called = {}

    def run_search(ctx, user_query, should_search, primary, qemb, temb, kws):
        called["primary"] = primary
        return ([], set())

    monkeypatch.setattr(agent, "_run_search_rounds", run_search)
    monkeypatch.setattr(agent, "_build_resp_inputs", lambda *a, **k: ({}, "response"))
    monkeypatch.setattr(agent, "_generate_and_stream_response", lambda *a, **k: "ans")

    res = agent._handle_query_core("orig", one_shot=True)
    assert res == "ans"
    assert called["primary"] == "orig"
