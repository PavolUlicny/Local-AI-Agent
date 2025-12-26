from src.agent import Agent
from src.config import AgentConfig
from src.exceptions import ResponseError
from io import StringIO
from src import response as response_mod


def make_agent():
    return Agent(AgentConfig(), output_stream=StringIO(), is_tty=True)


def test_notify_search_retry_prints_when_tty():
    agent = make_agent()
    out = agent._out
    agent._notify_search_retry(1, 3, 0.5, Exception("boom"))
    s = out.getvalue()
    assert "[search retry 1/3]" in s


def test_generate_and_stream_response_wrapper_handles_exception(monkeypatch):
    agent = make_agent()

    def raise_exc(*args, **kwargs):
        raise Exception("boom")

    monkeypatch.setattr(response_mod, "generate_and_stream_response", raise_exc)
    res = agent._generate_and_stream_response({}, "response", one_shot=True, write_fn=lambda s: None)
    assert res is None
    assert agent._last_error is not None
    assert "Answer generation failed" in agent._last_error


def test_handle_query_search_decision_model_missing(monkeypatch):
    agent = make_agent()

    # Simulate search_decision chain raising "model not found" error
    class SearchDecisionChain:
        def invoke(self, inputs):
            raise ResponseError("Model not found: Robot")

    agent.chains["search_decision"] = SearchDecisionChain()
    res = agent.answer_once("question")
    assert res is None
    assert agent._last_error is not None
    assert "not found" in agent._last_error.lower()


def test_build_search_orchestrator_delegates(monkeypatch):
    agent = make_agent()
    monkeypatch.setattr("src.search.build_search_orchestrator", lambda self: "orch")
    assert agent._build_search_orchestrator() == "orch"


def test_restore_llm_params_restores_base_values():
    cfg = AgentConfig()
    agent = Agent(cfg)
    # Access base params through LLMManager
    base = agent._llm_manager._base_params
    # mutate cfg values to different numbers
    cfg.assistant_num_ctx = base["assistant_num_ctx"] + 10
    cfg.robot_num_ctx = base["robot_num_ctx"] + 20
    cfg.assistant_num_predict = base["assistant_num_predict"] + 5
    cfg.robot_num_predict = base["robot_num_predict"] + 5

    agent._restore_llm_params()
    assert cfg.assistant_num_ctx == base["assistant_num_ctx"]
    assert cfg.robot_num_ctx == base["robot_num_ctx"]
    assert cfg.assistant_num_predict == base["assistant_num_predict"]
    assert cfg.robot_num_predict == base["robot_num_predict"]


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
