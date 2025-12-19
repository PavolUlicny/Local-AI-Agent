from types import SimpleNamespace

import src.response as response_mod


class FakeChain:
    def __init__(self, seq=None, exc=None):
        self.seq = seq or []
        self.exc = exc

    def stream(self, inputs):
        # generator that yields items, then optionally raises an exception
        for item in self.seq:
            yield item
        if self.exc:
            raise self.exc


class FakeAgent(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_error = None

    def _write(self, text: str) -> None:
        # no-op for tests
        pass

    def _writeln(self, text: str = "") -> None:
        # no-op for tests
        pass

    def _mark_error(self, message: str) -> str:
        self._last_error = message
        return message


def test_generate_and_stream_response_happy_path():
    chunks = ["hello ", "world\n"]
    fake_chain = FakeChain(seq=chunks)
    agent = FakeAgent(
        cfg=SimpleNamespace(assistant_model="m"),
        chains={"response": fake_chain},
        rebuild_counts={"answer": 0},
        _is_tty=False,
    )

    collected = []

    def write_fn(s: str):
        collected.append(s)

    out = response_mod.generate_and_stream_response(agent, {}, "response", one_shot=True, write_fn=write_fn)
    assert out == "".join(chunks)
    assert collected == chunks


def test_generate_and_stream_response_keyboard_interrupt():
    # Make a stream that raises KeyboardInterrupt during iteration (so the
    # exception is raised inside the streaming loop and can be handled by the
    # code under test, instead of at generator creation time which would
    # propagate to pytest as a real interrupt).
    class ChainKI:
        def stream(self, inputs):
            def gen():
                raise KeyboardInterrupt()
                yield  # pragma: no cover - unreachable

            return gen()

    agent = FakeAgent(
        cfg=SimpleNamespace(assistant_model="m"),
        chains={"response": ChainKI()},
        rebuild_counts={"answer": 0},
        _is_tty=False,
    )

    out = response_mod.generate_and_stream_response(agent, {}, "response", one_shot=False, write_fn=lambda x: None)
    assert out == ""


def test_generate_and_stream_response_stream_error_sets_mark_error():
    fake_exc = Exception("stream failed")
    fake_chain = FakeChain(seq=["part"], exc=fake_exc)
    agent = FakeAgent(
        cfg=SimpleNamespace(assistant_model="m"),
        chains={"response": fake_chain},
        rebuild_counts={"answer": 0},
        _is_tty=False,
    )
    out = response_mod.generate_and_stream_response(agent, {}, "response", one_shot=False, write_fn=lambda x: None)
    assert out is None
    assert agent._last_error is not None
