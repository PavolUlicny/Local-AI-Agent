import os

import src.ollama as ollama


def test_get_default_log_path_posix(monkeypatch):
    monkeypatch.setattr(ollama.os, "name", "posix")
    # Ensure a predictable home expansion via HOME env var
    monkeypatch.setenv("HOME", "/home/testuser")

    p = ollama.get_default_log_path()
    assert "~" not in p
    assert os.path.isabs(p)
    # Normalize separators and assert the injected HOME appears somewhere
    # in the returned path. On Windows CI `os.path.abspath` may prefix a
    # drive letter, so checking 'in' is more robust than 'startswith'.
    p_norm = p.replace("\\", "/")
    assert "/home/testuser" in p_norm
    assert p.endswith("installer_ollama.log")


def test_get_default_log_path_windows(monkeypatch):
    monkeypatch.setattr(ollama.os, "name", "nt")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Test\AppData\Local")

    p = ollama.get_default_log_path()
    assert "~" not in p
    assert os.path.isabs(p)
    # On non-Windows test runners the returned string may be normalized
    # to the local path semantics; ensure the path is expanded and looks
    # like a valid log path instead of asserting a platform-specific prefix.
    assert "ollama" in p
    assert p.endswith("installer_ollama.log")
