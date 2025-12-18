import pytest

from types import SimpleNamespace

import src.agent_utils as au_mod
from src.exceptions import ResponseError
import logging


class FakeChain:
    def __init__(self, behavior):
        # behavior: list of outcomes; Exception instances to raise, or values to return
        self.behavior = list(behavior)

    def invoke(self, inputs):
        if not self.behavior:
            return None
        item = self.behavior.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_invoke_chain_safe_success():
    agent = SimpleNamespace()
    agent.chains = {"c": FakeChain(["ok"])}
    agent.rebuild_counts = {}
    res = au_mod.invoke_chain_safe(agent, "c", {})
    assert res == "ok"


def test_invoke_chain_safe_model_missing_propagates():
    agent = SimpleNamespace()
    agent.chains = {"c": FakeChain([ResponseError("not found")])}
    agent.rebuild_counts = {}
    with pytest.raises(ResponseError):
        au_mod.invoke_chain_safe(agent, "c", {})


def test_invoke_chain_safe_context_length_retry(monkeypatch):
    # first call raises context-length ResponseError, second call returns ok
    exc = ResponseError("Context length exceeded")
    agent = SimpleNamespace()
    agent.chains = {"c": FakeChain([exc, "ok"])}
    agent.rebuild_counts = {"rk": 0}

    def reduce_context_and_rebuild(key, label):
        agent.rebuild_counts[key] = agent.rebuild_counts.get(key, 0) + 1

    agent._reduce_context_and_rebuild = reduce_context_and_rebuild

    res = au_mod.invoke_chain_safe(agent, "c", {}, rebuild_key="rk")
    assert res == "ok"
    assert agent.rebuild_counts["rk"] == 1


def test_inputs_and_build_resp_inputs_delegate():
    agent = SimpleNamespace()
    agent.build_inputs = lambda *a, **k: {"x": 1}
    res = au_mod.inputs(agent, "d", "y", "m", "dd", "conv", "q")
    assert res == {"x": 1}

    agent.build_inputs = lambda *a, **k: {"prior_responses": "p"}
    resp_inputs, chain_name = au_mod.build_resp_inputs(agent, "d", "y", "m", "dd", "conv", "q", True, "p", "sr")
    assert chain_name == "response"

    resp_inputs2, chain_name2 = au_mod.build_resp_inputs(agent, "d", "y", "m", "dd", "conv", "q", False, "p", None)
    assert chain_name2 == "response_no_search"


def test_safe_close_variants(caplog):
    from src.agent import Agent

    class NoClose:
        pass

    class GoodClose:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class BadClose:
        def close(self):
            raise RuntimeError("boom")

    # None should be handled silently
    Agent._safe_close(None)

    # object without close attribute should be fine
    Agent._safe_close(NoClose())

    # close that succeeds
    good = GoodClose()
    Agent._safe_close(good)
    assert good.closed is True

    # close that raises should not propagate and should log at debug
    caplog.set_level(logging.DEBUG)
    Agent._safe_close(BadClose())
    assert any("Client close failed" in rec.message for rec in caplog.records)


def test_mark_and_clear_error() -> None:
    from src.agent import Agent
    from src.config import AgentConfig

    agent = Agent(AgentConfig(no_auto_search=True))
    msg = agent._mark_error("problem")
    assert msg == "problem"
    assert agent._last_error == "problem"
    agent._clear_error()
    assert agent._last_error is None
