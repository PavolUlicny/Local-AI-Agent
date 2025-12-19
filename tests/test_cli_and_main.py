import logging
import argparse

import pytest

from src import cli


def test_build_arg_parser_defaults_and_flags():
    parser = cli.build_arg_parser()
    ns = parser.parse_args(["--no-auto-search", "--robot-model", "rm1", "--max-rounds", "3"])
    assert ns.no_auto_search is True
    assert ns.robot_model == "rm1"
    assert ns.max_rounds == 3


def test_configure_logging_nullhandler_when_no_console_or_file(monkeypatch):
    # Force reconfiguration
    cli.configure_logging("info", None, False, force=True)
    handlers = logging.getLogger().handlers
    assert handlers, "Handlers should be installed even when console and file are disabled"
    # When log_console False and no file, ensure there is a NullHandler or no StreamHandler
    assert not any(isinstance(h, logging.StreamHandler) for h in handlers)


def test_main_exits_on_ollama_check_failure(monkeypatch):
    import src.main as main_mod

    # Make check_and_start_ollama raise an exception to cause sys.exit
    monkeypatch.setattr(
        main_mod, "check_and_start_ollama", lambda exit_on_failure=True: (_ for _ in ()).throw(Exception("boom"))
    )
    ns = argparse.Namespace(log_level="INFO", log_file=None, log_console=True, question=None)
    with pytest.raises(SystemExit):
        main_mod.main(ns)


def test_main_calls_answer_once_when_question(monkeypatch):
    import src.main as main_mod

    called = {}

    def fake_check(*args, **kwargs):
        return True

    class FakeAgent:
        def __init__(self, cfg):
            pass

        def answer_once(self, q):
            called["q"] = q

        def run(self):
            called["run"] = True

    monkeypatch.setattr(main_mod, "check_and_start_ollama", fake_check)
    monkeypatch.setattr(main_mod, "Agent", FakeAgent)
    ns = argparse.Namespace(log_level="INFO", log_file=None, log_console=True, question=" hello ")
    main_mod.main(ns)
    assert called.get("q") == "hello"
