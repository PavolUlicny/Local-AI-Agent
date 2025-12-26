from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import pytest

from src import agent as agent_mod
from src.agent import Agent
from src.config import AgentConfig


def test_print_welcome_banner_colored_and_plain(monkeypatch):
    out = StringIO()
    cfg = AgentConfig()
    a = Agent(cfg, output_stream=out, is_tty=True)

    # Colored when ANSI is present and _is_tty True
    monkeypatch.setattr(agent_mod, "ANSI", object())
    a._print_welcome_banner()
    s = out.getvalue()
    assert "Welcome to Local AI Agent." in s
    # contains ANSI color sequence start
    assert "\033[96m" in s

    # Plain when ANSI None or not TTY
    out2 = StringIO()
    b = Agent(cfg, output_stream=out2, is_tty=False)
    monkeypatch.setattr(agent_mod, "ANSI", None)
    b._print_welcome_banner()
    assert "Welcome to Local AI Agent." in out2.getvalue()


def test_close_clients_closes_and_handles_exceptions():
    cfg = AgentConfig()
    a = Agent(cfg)

    class SClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class BClient:
        def close(self):
            raise RuntimeError("boom")

    a.search_client = SClient()
    a.embedding_client = BClient()
    # embedding_client.close may raise; ensure search_client was cleared and exception propagates
    with pytest.raises(RuntimeError):
        a._close_clients()
    assert a.search_client is None


def test_prompt_session_wrappers_and_read_user_query():
    cfg = AgentConfig()
    a = Agent(cfg)

    class IH:
        def __init__(self):
            self.prompted = False

        def prompt_messages(self):
            return ("msg", "suffix")

        def build_prompt_session(self):
            return "session"

        def ensure_prompt_session(self, s):
            return s or "created"

        def read_user_query(self, session):
            return "user input"

    a.input_handler = IH()
    # Use the InputHandler API directly (prompt/session helpers were
    # moved off Agent). Ensure delegation works via the handler and that
    # Agent still reads user input via its input_handler.
    msgs, suffix = a.input_handler.prompt_messages()
    assert msgs == "msg"
    assert suffix == "suffix"
    assert a.input_handler.build_prompt_session() == "session"
    # ensure_prompt_session sets internal prompt session on the handler; set
    # Agent's stored session to the created session so _read_user_query uses it.
    a._prompt_session = a.input_handler.ensure_prompt_session(a._prompt_session)
    assert a._prompt_session == "created"
    assert a._read_user_query() == "user input"


def test_run_handles_exit_and_interrupt(monkeypatch):
    cfg = AgentConfig()
    out = StringIO()
    a = Agent(cfg, output_stream=out, is_tty=False)

    # exit path using /quit command
    class IHExit:
        def read_user_query(self, session):
            return "/quit"

    a.input_handler = IHExit()
    a._prompt_session = None
    a.run()  # should exit cleanly via command handler
    assert "Goodbye" in out.getvalue()

    # exit path using backward-compatible 'exit'
    out2 = StringIO()
    a2 = Agent(cfg, output_stream=out2, is_tty=False)

    class IHExitOldStyle:
        def read_user_query(self, session):
            return "exit"

    a2.input_handler = IHExitOldStyle()
    a2._prompt_session = None
    a2.run()  # should exit cleanly via backward compatibility
    assert "Goodbye" in out2.getvalue()

    # interrupt path - should print message and continue loop, then exit on second call
    out3 = StringIO()
    a3 = Agent(cfg, output_stream=out3, is_tty=False)

    class IHInterrupt:
        def __init__(self):
            self.call_count = 0

        def read_user_query(self, session):
            self.call_count += 1
            if self.call_count == 1:
                raise KeyboardInterrupt()
            return "/quit"  # exit on second call

    a3.input_handler = IHInterrupt()
    a3._prompt_session = None
    a3.run()  # should catch interrupt and continue, then exit
    assert "Interrupted" in out3.getvalue()


def test_ddg_results_constructs_search_client_when_none(monkeypatch):
    cfg = AgentConfig()
    a = Agent(cfg)
    # remove existing client
    a.search_client = None

    class FakeSearchClient:
        def __init__(self, cfg, normalizer=None, notify_retry=None):
            pass

        def fetch(self, q):
            return ["r1"]

    monkeypatch.setattr(agent_mod, "_search_client_mod", SimpleNamespace(SearchClient=FakeSearchClient))
    res = a._ddg_results("query")
    assert res == ["r1"]
