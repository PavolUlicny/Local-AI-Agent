from __future__ import annotations


def test_read_user_query_uses_injected_input_fn():
    from src.input_handler import InputHandler

    ih = InputHandler(is_tty=False, prompt_session=None, input_fn=lambda prompt: "injected")
    res = ih.read_user_query(None)
    assert res == "injected"


def test_read_user_query_uses_prompt_session(monkeypatch):
    from src.input_handler import InputHandler

    class DummySession:
        def prompt(self, prompt_text):
            return "from_session"

    ih = InputHandler(is_tty=False, prompt_session=None, input_fn=None)
    res = ih.read_user_query(DummySession())
    assert res == "from_session"
