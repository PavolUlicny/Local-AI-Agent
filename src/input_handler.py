from __future__ import annotations

from typing import Any, Tuple, Callable
import logging

# Runtime variables for optional prompt_toolkit components. Annotate as
# optional to avoid mypy treating them as immutable types when set to None.
PromptSession: Any | None
ANSI: Any | None
InMemoryHistory: Any | None

try:
    from prompt_toolkit import PromptSession as _PromptSession
    from prompt_toolkit.formatted_text import ANSI as _ANSI
    from prompt_toolkit.history import InMemoryHistory as _InMemoryHistory
except Exception:  # pragma: no cover - optional dependency
    PromptSession = None
    ANSI = None
    InMemoryHistory = None
else:
    PromptSession = _PromptSession
    ANSI = _ANSI
    InMemoryHistory = _InMemoryHistory


class InputHandler:
    """Encapsulates interactive prompt/session handling.

    This module mirrors the minimal prompt-related helpers from `Agent` so
    they can be tested and reasoned about separately. The implementation
    intentionally avoids side-effects; reading from stdin falls back to
    built-in `input()` when a `prompt_toolkit` session is unavailable.
    """

    def __init__(self, is_tty: bool, prompt_session: Any | None = None, input_fn: Callable[[str], str] | None = None):
        self._is_tty = bool(is_tty)
        self._prompt_session = prompt_session
        # input_fn allows injecting a non-blocking input function for tests.
        self._input_fn = input_fn

    def prompt_messages(self) -> Tuple[Any, str]:
        """Return (formatted_prompt, plain_prompt) tuple for user input.

        If `prompt_toolkit.ANSI` is available and `is_tty` is True, return a
        formatted ANSI prompt; otherwise return a plain text prompt string.
        """
        if ANSI is not None and self._is_tty:
            try:
                return ANSI("\n\033[92m> \033[0m"), "> "
            except Exception:
                logging.debug("ANSI formatting failed; falling back to plain prompt")
        return "> ", "> "

    def build_prompt_session(self) -> Any | None:
        if PromptSession is None or InMemoryHistory is None:
            return None
        return PromptSession(history=InMemoryHistory(), multiline=False, wrap_lines=True)

    def ensure_prompt_session(self, existing: Any | None = None) -> Any | None:
        if existing is not None:
            return existing
        if self._prompt_session is None:
            self._prompt_session = self.build_prompt_session()
        return self._prompt_session

    def read_user_query(self, session: Any | None = None) -> str:
        formatted_prompt, _ = self.prompt_messages()
        # If an injected input function is provided prefer it and avoid
        # creating a prompt_toolkit session (which would attempt to read
        # from stdin and block in tests).
        if session is None and self._input_fn is not None:
            return self._input_fn(formatted_prompt)
        session = self.ensure_prompt_session(session)
        if session is None:
            # fallback to builtin input() when no injected input function
            # and no prompt session is available.
            return input(formatted_prompt)  # noqa: A001 - acceptable here
        # prompt_toolkit's prompt() can return Any; ensure we return a str for typing
        return str(session.prompt(formatted_prompt))


def build_inputs(
    current_datetime: str,
    current_year: str,
    current_month: str,
    current_day: str,
    conversation_text: str,
    user_query: str,
    **overrides: Any,
) -> dict[str, Any]:
    base = {
        "current_datetime": current_datetime,
        "current_year": current_year,
        "current_month": current_month,
        "current_day": current_day,
        "conversation_history": conversation_text,
        "user_question": user_query,
    }
    base.update(overrides)
    return base


__all__ = ["InputHandler", "build_inputs"]
