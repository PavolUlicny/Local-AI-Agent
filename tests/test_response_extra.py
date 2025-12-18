from types import SimpleNamespace

from src import response as response_mod
from src.exceptions import ResponseError


class ChainRaiseNotFound:
    def stream(self, inputs):
        raise ResponseError("Model not found: assistant")


class ChainContextThenYield:
    def __init__(self):
        self._called = 0

    def stream(self, inputs):
        if self._called == 0:
            self._called += 1
            raise ResponseError("Context length exceeded")
        return iter(["ok"])  # return an iterator for the retry


def test_generate_and_stream_response_not_found(monkeypatch):
    class AgentFake(SimpleNamespace):
        def __init__(self):
            super().__init__()
            self.cfg = SimpleNamespace(assistant_model="m")
            self.chains = {"response": ChainRaiseNotFound()}
            self.rebuild_counts = {"answer": 0}
            self._is_tty = False

        def _mark_error(self, msg):
            self._last = msg

        def _writeln(self, text=""):
            pass

        def _write(self, text):
            pass

    agent = AgentFake()
    called = {}
    monkeypatch.setattr(
        response_mod,
        "_model_utils_mod",
        SimpleNamespace(handle_missing_model=lambda cb, kind, name: called.setdefault("m", name)),
    )
    out = response_mod.generate_and_stream_response(agent, {}, "response", one_shot=True, write_fn=lambda x: None)
    assert out is None
    assert called.get("m") == "m"


def test_generate_and_stream_response_context_length_retry_success(monkeypatch):
    class AgentFake(SimpleNamespace):
        def __init__(self):
            super().__init__()
            self.cfg = SimpleNamespace(assistant_model="m")
            self.chains = {"response": ChainContextThenYield()}
            self.rebuild_counts = {"answer": 0}
            self._is_tty = False

        def _mark_error(self, msg):
            self._last = msg

        def _writeln(self, text=""):
            pass

        def _write(self, text):
            pass

        def _reduce_context_and_rebuild(self, k, label):
            self.rebuild_counts["answer"] = self.rebuild_counts.get("answer", 0) + 1

    agent = AgentFake()
    out = response_mod.generate_and_stream_response(agent, {}, "response", one_shot=True, write_fn=lambda x: None)
    assert out == "ok"


def test_generate_and_stream_response_context_length_retry_exceeded(monkeypatch):
    # set rebuild_counts high so it will not retry
    class ChainCtx:
        def stream(self, inputs):
            raise ResponseError("Context length exceeded")

    class AgentFake(SimpleNamespace):
        def __init__(self):
            super().__init__()
            self.cfg = SimpleNamespace(assistant_model="m")
            self.chains = {"response": ChainCtx()}
            self.rebuild_counts = {"answer": 99}
            self._is_tty = False

        def _mark_error(self, msg):
            self._last = msg

        def _writeln(self, text=""):
            pass

        def _write(self, text):
            pass

    agent = AgentFake()
    out = response_mod.generate_and_stream_response(agent, {}, "response", one_shot=True, write_fn=lambda x: None)
    assert out is None


def test_generate_and_stream_response_generic_exception(monkeypatch):
    class ChainErr:
        def stream(self, inputs):
            raise RuntimeError("boom")

    class AgentFake(SimpleNamespace):
        def __init__(self):
            super().__init__()
            self.cfg = SimpleNamespace(assistant_model="m")
            self.chains = {"response": ChainErr()}
            self.rebuild_counts = {"answer": 0}
            self._is_tty = False

        def _mark_error(self, msg):
            self._last = msg

        def _writeln(self, text=""):
            pass

        def _write(self, text):
            pass

    agent = AgentFake()
    out = response_mod.generate_and_stream_response(agent, {}, "response", one_shot=True, write_fn=lambda x: None)
    assert out is None
