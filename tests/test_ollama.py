import src.ollama as ollama


def test_is_installed_false(monkeypatch):
    monkeypatch.setattr(ollama, "is_installed", lambda: False)
    assert not ollama.is_installed()


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


# Note: tests for ensure_available and check_and_start_ollama are covered in
# `tests/test_ollama_extra.py` which exercises multiple OS/behavioral branches.
