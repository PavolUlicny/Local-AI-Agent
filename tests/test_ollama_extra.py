from __future__ import annotations

import subprocess
import urllib.error

import pytest

from src import ollama


def test_get_default_log_path_posix(monkeypatch, tmp_path):
    # Ensure the function expands HOME and returns a path that looks like a
    # POSIX log path (e.g., contains '.local' or the provided HOME dir).
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(ollama.os, "name", "posix")
    p = ollama.get_default_log_path()
    p_norm = p.replace("\\", "/")
    assert ".local" in p_norm or str(tmp_path) in p_norm
    assert p_norm.endswith("installer_ollama.log")


def test_get_default_log_path_windows(monkeypatch, tmp_path):
    # Windows behavior: LOCALAPPDATA should be present in the returned path
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(ollama.os, "name", "nt")
    p = ollama.get_default_log_path()
    p_norm = p.replace("\\", "/")
    assert "ollama" in p_norm
    assert p_norm.endswith("installer_ollama.log")


def test_is_ready_true(monkeypatch):
    class DummyResp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(ollama.urllib.request, "urlopen", lambda url, timeout: DummyResp())
    assert ollama.is_ready()


def test_is_ready_false(monkeypatch):
    def bad(*args, **kwargs):
        raise urllib.error.URLError("no")

    monkeypatch.setattr(ollama.urllib.request, "urlopen", bad)
    assert not ollama.is_ready()


def test_start_detached_not_installed(monkeypatch):
    monkeypatch.setattr(ollama, "is_installed", lambda: False)
    assert ollama.start_detached() is None


def test_start_detached_popen_failure(monkeypatch):
    monkeypatch.setattr(ollama, "is_installed", lambda: True)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    assert ollama.start_detached(log_path=None) is None


def test_wait_for_ready_times_out(monkeypatch):
    monkeypatch.setattr(ollama, "is_ready", lambda **kw: False)
    assert not ollama.wait_for_ready(wait_seconds=0, poll_interval=0)


def test_ensure_available_not_installed(monkeypatch):
    monkeypatch.setattr(ollama, "is_installed", lambda: False)
    assert not ollama.ensure_available()


def test_check_and_start_ollama_exit_on_failure(monkeypatch):
    monkeypatch.setattr(ollama, "ensure_available", lambda **kw: False)
    with pytest.raises(SystemExit):
        ollama.check_and_start_ollama(exit_on_failure=True)
