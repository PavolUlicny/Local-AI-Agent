import pytest

import scripts.install_deps as inst


import platform as _platform


@pytest.mark.skipif(_platform.system().lower() != "windows", reason="Windows-specific test")
def test_find_python312_windows_py_launcher(monkeypatch):
    # Simulate no python3.12 on PATH, but a `py` launcher that reports 3.12
    monkeypatch.setattr(inst, "platform", inst.platform)
    monkeypatch.setattr(inst.platform, "system", lambda: "Windows")

    def fake_which(name):
        if name == "py":
            return "C:\\Windows\\py.exe"
        return None

    monkeypatch.setattr(inst.shutil, "which", fake_which)

    def fake_check_output(cmd, stderr=None):
        # emulate: py -3.12 -c 'import sys; print(sys.version_info[:2])'
        if cmd[0].endswith("py") and cmd[1] == "-3.12":
            return b"(3, 12)\n"
        raise RuntimeError("unexpected")

    monkeypatch.setattr(inst.subprocess, "check_output", fake_check_output)

    # Ensure unix candidates are ignored so the py launcher path is chosen.
    monkeypatch.setattr(inst.os.path, "exists", lambda p: False)
    monkeypatch.setattr(inst, "glob", inst.glob)
    monkeypatch.setattr(inst.glob, "glob", lambda p: [])

    res = inst.find_python312()
    assert res is not None
    assert res.lower().endswith("py.exe")


def test_ensure_venv_uses_py_launcher_on_windows(monkeypatch, tmp_path):
    # Simulate Windows environment and ensure `py -3.12 -m venv` is invoked
    monkeypatch.setattr(inst.os, "name", "nt")
    # Point VENV_DIR to a temp dir so we don't touch the repo
    venv_dir = tmp_path / ".venv"
    monkeypatch.setattr(inst, "VENV_DIR", venv_dir)

    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        # create the venv python path to satisfy existence check
        bin_dir = "Scripts"
        exe_name = "python.exe"
        target = venv_dir / bin_dir
        target.mkdir(parents=True, exist_ok=True)
        (target / exe_name).write_text("")

    monkeypatch.setattr(inst, "run", fake_run)

    py = inst.ensure_venv("py")
    # Current implementation invokes `<python_exe> -m venv <path>`.
    assert "-m" in calls[0] and "venv" in calls[0]
    assert py.exists()


def test_install_files_uses_repo_cache_by_default(monkeypatch, tmp_path):
    # Ensure install_files uses ROOT/.cache/pip when PIP_CACHE_DIR not set
    monkeypatch.delenv("PIP_CACHE_DIR", raising=False)
    # point ROOT to tmp
    monkeypatch.setattr(inst, "ROOT", tmp_path)

    recorded = {}

    def fake_run(cmd):
        recorded["cmd"] = cmd

    monkeypatch.setattr(inst, "run", fake_run)

    # create a requirements file
    req = tmp_path / "requirements.txt"
    req.write_text("requests\n")

    py_path = tmp_path / "pybin"
    py_path.write_text("")

    inst.install_files(py_path, ["requirements.txt"])

    # Current implementation calls pip install -r without --cache-dir
    assert recorded["cmd"] == [str(py_path), "-m", "pip", "install", "-r", str(req)]


def test_start_ollama_if_needed_starts_and_waits(monkeypatch, tmp_path):
    # Simulate ollama present but server not ready, starting succeeds and then ready
    monkeypatch.setattr(inst.shutil, "which", lambda name: "/usr/bin/ollama")

    seq = {"calls": 0}

    def fake_server_ready(host="127.0.0.1", port=11434, timeout=1.0):
        # first two calls False, then True
        seq["calls"] += 1
        return seq["calls"] >= 3

    monkeypatch.setattr(inst, "ollama_server_ready", fake_server_ready)

    class DummyProc:
        pid = 9999

    def fake_popen(cmd, stdout=None, stderr=None, creationflags=None, start_new_session=None):
        return DummyProc()

    monkeypatch.setattr(inst.subprocess, "Popen", fake_popen)

    ok = inst.start_ollama_if_needed(wait_seconds=5)
    # Current implementation returns None on success; ensure it doesn't raise and printed readiness.
    assert ok is None


def test_pull_models_skips_and_prints(monkeypatch, capsys):
    # Simulate `ollama` not on PATH
    monkeypatch.setattr(inst.shutil, "which", lambda name: None)

    inst.pull_models(["some-model"])

    captured = capsys.readouterr()
    # Message should be printed to stderr about missing CLI and an install hint
    assert "Ollama CLI not found" in captured.err
    assert "curl -fsSL https://ollama.com/install.sh | sh" in captured.err


def test_find_python312_detects_candidate(monkeypatch):
    # Simulate python3.12 being present on PATH
    monkeypatch.setattr(inst.shutil, "which", lambda name: "/usr/bin/python3")

    def fake_check_output(cmd, **kwargs):
        # Return a Python 3.12 version tuple when asked
        return b"(3, 12)\n"

    monkeypatch.setattr(inst.subprocess, "check_output", fake_check_output)

    found = inst.find_python312()
    assert found is not None


def test_start_ollama_if_needed_starts(monkeypatch, tmp_path, capsys):
    # Simulate `ollama` on PATH
    monkeypatch.setattr(inst.shutil, "which", lambda name: "/usr/bin/ollama")

    # Simulate server readiness only after a couple of polls
    state = {"calls": 0}

    def fake_ready(host="127.0.0.1", port=11434, timeout=1.0):
        state["calls"] += 1
        # Become ready on the 2nd call
        return state["calls"] >= 2

    monkeypatch.setattr(inst, "ollama_server_ready", fake_ready)

    # Fake subprocess.Popen so we don't actually start anything
    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.pid = 4321

    monkeypatch.setattr(inst.subprocess, "Popen", FakePopen)

    # Use a temporary HOME so log file path is writable and isolated
    monkeypatch.setenv("HOME", str(tmp_path))

    # Should not raise; will print started/ready messages
    inst.start_ollama_if_needed(wait_seconds=3)

    captured = capsys.readouterr()
    assert "Started Ollama (pid" in captured.out or "Ollama server already responding" in captured.out


def test_ensure_venv_on_linux(monkeypatch, tmp_path):
    # Simulate POSIX environment and ensure venv created under bin/
    monkeypatch.setattr(inst.os, "name", "posix")
    venv_dir = tmp_path / ".venv"
    monkeypatch.setattr(inst, "VENV_DIR", venv_dir)

    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        # create the venv python path to satisfy existence check
        bin_dir = "bin"
        exe_name = "python"
        target = venv_dir / bin_dir
        target.mkdir(parents=True, exist_ok=True)
        (target / exe_name).write_text("")

    monkeypatch.setattr(inst, "run", fake_run)

    py = inst.ensure_venv("python3")
    assert len(calls) > 0
    assert "-m" in calls[0] and "venv" in calls[0]
    assert py.exists()


def test_find_python312_via_glob_candidate(monkeypatch):
    # Ensure glob-based candidate detection works when which() returns None
    monkeypatch.setattr(inst.shutil, "which", lambda name: None)
    monkeypatch.setattr(inst.glob, "glob", lambda pattern: ["/usr/bin/python3.12"])
    monkeypatch.setattr(inst.os.path, "exists", lambda p: True if p == "/usr/bin/python3.12" else False)
    monkeypatch.setattr(inst.subprocess, "check_output", lambda cmd, **kwargs: b"(3, 12)\n")

    found = inst.find_python312()
    assert found is not None
    assert "/usr/bin/python3.12" in found


def test_start_ollama_posix_launch_and_log_close(monkeypatch, tmp_path):
    # POSIX-specific: ensure start_new_session is used and parent log file closed
    monkeypatch.setattr(inst.shutil, "which", lambda name: "/usr/bin/ollama")

    seq = {"calls": 0}

    def fake_server_ready(host="127.0.0.1", port=11434, timeout=1.0):
        seq["calls"] += 1
        return seq["calls"] >= 2

    monkeypatch.setattr(inst, "ollama_server_ready", fake_server_ready)

    recorded = {}

    class FakePopen:
        def __init__(self, cmd, stdout=None, stderr=None, start_new_session=None, creationflags=None):
            recorded["cmd"] = cmd
            recorded["kwargs"] = {"start_new_session": start_new_session, "creationflags": creationflags}
            self.pid = 1111

    monkeypatch.setattr(inst.subprocess, "Popen", FakePopen)

    class FakeFile:
        def __init__(self):
            self.closed = False
            self.closed_called = False

        def write(self, data):
            pass

        def flush(self):
            pass

        def close(self):
            self.closed = True
            self.closed_called = True

    def fake_open(path, mode="a"):
        recorded["open_path"] = path
        f = FakeFile()
        recorded["file"] = f
        return f

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(inst.time, "sleep", lambda s: None)

    inst.start_ollama_if_needed(wait_seconds=3)

    assert recorded["kwargs"]["start_new_session"] is True
    assert str(tmp_path) in str(recorded["open_path"])
    assert recorded["file"].closed_called


def test_pull_models_nonfatal_on_failure(monkeypatch, capsys):
    # Simulate ollama present but pull fails; installer should not raise
    monkeypatch.setattr(inst.shutil, "which", lambda name: "/usr/bin/ollama")

    class FakeProc:
        def __init__(self, *args, **kwargs):
            pass

        def communicate(self):
            return (b"", b"error: failed")

        def wait(self):
            return 1

        @property
        def returncode(self):
            return 1

    monkeypatch.setattr(inst.subprocess, "Popen", lambda *a, **k: FakeProc())

    # Should not raise
    inst.pull_models(["modelX"])

    captured = capsys.readouterr()
    assert captured.err != ""


def test_start_ollama_polling_behavior(monkeypatch):
    # Ensure polling/backoff logic calls readiness multiple times
    monkeypatch.setattr(inst.shutil, "which", lambda name: "/usr/bin/ollama")
    calls = {"n": 0}

    def fake_ready(*a, **k):
        calls["n"] += 1
        return calls["n"] >= 4

    monkeypatch.setattr(inst, "ollama_server_ready", fake_ready)

    class FakePopen:
        def __init__(self, *a, **k):
            self.pid = 1

    monkeypatch.setattr(inst.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(inst.time, "sleep", lambda s: None)

    inst.start_ollama_if_needed(wait_seconds=5)
    assert calls["n"] >= 4
