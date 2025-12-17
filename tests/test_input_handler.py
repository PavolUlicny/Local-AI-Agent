from src.input_handler import InputHandler, build_inputs


def test_build_inputs_basic():
    data = build_inputs("now", "2025", "12", "17", "conv", "what?")
    assert data["current_datetime"] == "now"
    assert data["current_year"] == "2025"
    assert data["conversation_history"] == "conv"
    assert data["user_question"] == "what?"


def test_prompt_messages_plain():
    handler = InputHandler(is_tty=False)
    formatted, plain = handler.prompt_messages()
    # When not a TTY we expect a plain prompt string
    assert plain == "> "
    assert isinstance(formatted, str)


def test_read_user_query_with_injected_input_fn():
    # Provide an input_fn that returns a known value without blocking.
    handler = InputHandler(is_tty=False, input_fn=lambda prompt: "injected input")
    result = handler.read_user_query(session=None)
    assert result == "injected input"


def test_read_user_query_with_fake_session():
    class FakeSession:
        def prompt(self, prompt_text):
            return "session input"

    handler = InputHandler(is_tty=True, prompt_session=None)
    # ensure_prompt_session will create one only if prompt_toolkit present; pass fake directly
    result = handler.read_user_query(session=FakeSession())
    assert result == "session input"
