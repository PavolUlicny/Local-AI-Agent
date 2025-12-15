import pytest

import src.ollama as ollama


def test_is_installed_false(monkeypatch):
    monkeypatch.setattr(ollama, "is_installed", lambda: False)
    assert not ollama.is_installed()


def test_ensure_available_not_installed(monkeypatch):
    # Simulate missing ollama CLI
    monkeypatch.setattr(ollama, "is_installed", lambda: False)
    assert not ollama.ensure_available()


def test_ensure_available_already_ready(monkeypatch):
    # Simulate installed and ready
    monkeypatch.setattr(ollama, "is_installed", lambda: True)
    monkeypatch.setattr(ollama, "is_ready", lambda **kw: True)
    assert ollama.ensure_available()


def test_ensure_available_starts_and_waits(monkeypatch):
    # Simulate installed but not ready; starting succeeds and wait_for_ready returns True
    monkeypatch.setattr(ollama, "is_installed", lambda: True)
    monkeypatch.setattr(ollama, "is_ready", lambda **kw: False)

    class DummyProc:
        pid = 12345

    monkeypatch.setattr(ollama, "start_detached", lambda log_path=None: DummyProc())
    monkeypatch.setattr(ollama, "wait_for_ready", lambda wait_seconds, poll_interval, host=None, port=None: True)

    assert ollama.ensure_available(wait_seconds=1, poll_interval=0.01)


def test_check_and_start_ollama_exit_on_failure(monkeypatch):
    # When ensure_available returns False and exit_on_failure=True, SystemExit is raised
    monkeypatch.setattr(ollama, "ensure_available", lambda **kw: False)
    with pytest.raises(SystemExit):
        ollama.check_and_start_ollama(exit_on_failure=True)


def test_start_detached_not_installed(monkeypatch):
    # If not installed, start_detached should return None
    monkeypatch.setattr(ollama, "is_installed", lambda: False)
    assert ollama.start_detached() is None
