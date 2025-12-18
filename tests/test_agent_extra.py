import pytest

from src.agent import Agent
from src.config import AgentConfig
from src.exceptions import ResponseError


def test_normalize_search_result_image_only() -> None:
    agent = Agent(AgentConfig())
    raw = {"image": "http://img"}
    res = agent._normalize_search_result(raw)
    assert res == {"title": "", "link": "http://img", "snippet": ""}


def test_normalize_search_result_empty_returns_none() -> None:
    agent = Agent(AgentConfig())
    raw = {"title": "", "body": "", "href": ""}
    assert agent._normalize_search_result(raw) is None


class Ctx:
    current_datetime = "now"
    current_year = "2025"
    current_month = "01"
    current_day = "01"
    conversation_text = "conv"
    prior_responses_text = "prior"


def test_decide_should_search_propagates_not_found(monkeypatch):
    cfg = AgentConfig()
    agent = Agent(cfg)

    def invoke_raise(*args, **kwargs):
        raise ResponseError("Model not found: Robot")

    monkeypatch.setattr(agent, "_invoke_chain_safe", invoke_raise)
    with pytest.raises(ResponseError):
        agent._decide_should_search(Ctx(), "q", "p")


def test_decide_should_search_propagates_context_length(monkeypatch):
    cfg = AgentConfig()
    agent = Agent(cfg)

    # first call raises context-length, reduce_context increments count, second chain returns SEARCH
    called = {"reduced": 0}

    def raise_context(*args, **kwargs):
        raise ResponseError("Context length exceeded")

    class GoodChain:
        def invoke(self, inputs):
            return "SEARCH"

    monkeypatch.setattr(agent, "_invoke_chain_safe", raise_context)
    monkeypatch.setattr(
        agent, "_reduce_context_and_rebuild", lambda k, label: called.update({"reduced": called.get("reduced", 0) + 1})
    )
    agent.chains["search_decision"] = GoodChain()
    # _decide_should_search does not swallow ResponseError; it should propagate
    with pytest.raises(ResponseError):
        agent._decide_should_search(Ctx(), "q", "p")
