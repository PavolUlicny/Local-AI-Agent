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


def test_prompt_messages_with_ansi_and_build_session(monkeypatch):
    import src.input_handler as ih

    # Simulate ANSI present and is_tty True
    monkeypatch.setattr(ih, "ANSI", lambda s: f"FMT:{s}")
    h = ih.InputHandler(is_tty=True)
    formatted, plain = h.prompt_messages()
    assert isinstance(formatted, str) and formatted.startswith("FMT:")
    assert plain == "> "

    # Simulate presence of PromptSession and InMemoryHistory
    class DummyHistory:
        pass

    class DummySession:
        def __init__(self, history=None, multiline=False, wrap_lines=False):
            self.history = history

    monkeypatch.setattr(ih, "PromptSession", DummySession)
    monkeypatch.setattr(ih, "InMemoryHistory", DummyHistory)
    sess = h.build_prompt_session()
    assert isinstance(sess, DummySession)
