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
