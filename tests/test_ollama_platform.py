import builtins

import src.ollama as ollama


def test_start_detached_posix_uses_start_new_session_and_closes_log(monkeypatch, tmp_path):
    # Simulate installed on POSIX and verify start_new_session is set and log closed
    monkeypatch.setattr(ollama, "is_installed", lambda: True)
    monkeypatch.setattr(ollama.os, "name", "posix")

    recorded = {}

    class FakeProc:
        def __init__(self):
            self.pid = 4242

    def fake_popen(cmd, stdout=None, stderr=None, start_new_session=None):
        recorded["cmd"] = cmd
        recorded["kwargs"] = {"start_new_session": start_new_session}
        return FakeProc()

    monkeypatch.setattr(ollama.subprocess, "Popen", fake_popen)

    class FakeFile:
        def __init__(self):
            self.closed = False

        def write(self, data):
            pass

        def flush(self):
            pass

        def close(self):
            self.closed = True

    def fake_open(path, mode="a+"):
        recorded["open_path"] = path
        f = FakeFile()
        recorded["file_obj"] = f
        return f

    monkeypatch.setattr(builtins, "open", fake_open)

    proc = ollama.start_detached(log_path=str(tmp_path / "o.log"))
    assert proc is not None
    assert recorded["kwargs"]["start_new_session"] is True
    assert str(tmp_path) in str(recorded["open_path"])
    assert recorded["file_obj"].closed is True


def test_start_detached_windows_uses_creationflags_and_closes_log(monkeypatch, tmp_path):
    # Simulate installed on Windows and verify creationflags used and log closed
    monkeypatch.setattr(ollama, "is_installed", lambda: True)
    monkeypatch.setattr(ollama.os, "name", "nt")

    recorded = {}

    class FakeProc:
        def __init__(self):
            self.pid = 9999

    def fake_popen(cmd, stdout=None, stderr=None, creationflags=None):
        recorded["cmd"] = cmd
        recorded["kwargs"] = {"creationflags": creationflags}
        return FakeProc()

    monkeypatch.setattr(ollama.subprocess, "Popen", fake_popen)

    class FakeFile:
        def __init__(self):
            self.closed = False

        def write(self, data):
            pass

        def flush(self):
            pass

        def close(self):
            self.closed = True

    def fake_open(path, mode="a+"):
        recorded["open_path"] = path
        f = FakeFile()
        recorded["file_obj"] = f
        return f

    monkeypatch.setattr(builtins, "open", fake_open)

    proc = ollama.start_detached(log_path=str(tmp_path / "o_win.log"))
    assert proc is not None
    # creationflags should be present in kwargs (may be 0 on non-Windows hosts)
    assert "creationflags" in recorded["kwargs"]
    assert str(tmp_path) in str(recorded["open_path"])
    assert recorded["file_obj"].closed is True


def test_wait_for_ready_timeout_and_success(monkeypatch):
    # Timeout case: wait_seconds=0 should immediately return False when not ready
    monkeypatch.setattr(ollama, "is_ready", lambda **kw: False)
    assert not ollama.wait_for_ready(wait_seconds=0, poll_interval=0)

    # Success case: is_ready returns True immediately
    monkeypatch.setattr(ollama, "is_ready", lambda **kw: True)
    assert ollama.wait_for_ready(wait_seconds=1, poll_interval=0)


def test_check_and_start_ollama_success(monkeypatch):
    # ensure_available True should cause check_and_start_ollama to return True
    monkeypatch.setattr(ollama, "ensure_available", lambda **kw: True)
    assert ollama.check_and_start_ollama(exit_on_failure=False)


def test_start_detached_log_open_failure(monkeypatch):
    # If opening the log file raises, start_detached should still start using DEVNULL
    monkeypatch.setattr(ollama, "is_installed", lambda: True)
    monkeypatch.setattr(ollama.os, "name", "posix")

    class FakeProc:
        def __init__(self):
            self.pid = 77

    def fake_popen(cmd, stdout=None, stderr=None, start_new_session=None):
        return FakeProc()

    monkeypatch.setattr(ollama.subprocess, "Popen", fake_popen)

    def raising_open(path, mode="a+"):
        raise OSError("cannot open")

    import builtins as _builtins

    monkeypatch.setattr(_builtins, "open", raising_open)

    proc = ollama.start_detached(log_path="/nonexistent/dir/log")
    assert proc is not None
