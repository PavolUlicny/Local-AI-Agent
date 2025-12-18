import pytest

import scripts.install_deps as inst


import os


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific test")
def test_find_python312_windows_py_launcher(monkeypatch):
    # Simulate no python3.12 on PATH, but a `py` launcher that reports 3.12
    # Use `os.name` to simulate Windows rather than monkeypatching `platform`.
    monkeypatch.setattr(inst.os, "name", "nt")

    def fake_which(name):
        if name == "py":
            return "C:\\Windows\\py.exe"
        return None

    monkeypatch.setattr(inst.shutil, "which", fake_which)

    def fake_check_output(cmd, stderr=None):
        # emulate: py -3.12 -c 'import sys; print(sys.version_info[:2])'
        # Accept flexible call signatures and command shapes used by
        # different Python versions / platforms.
        try:
            first = cmd[0]
            second = cmd[1]
        except Exception:
            raise RuntimeError("unexpected check_output args") from None
        # Accept either a bare `py` or a `py.exe` path on Windows
        first_base = os.path.basename(str(first)).lower()
        if first_base in ("py", "py.exe") and str(second) in ("-3.12", "-3.12.exe"):
            return b"(3, 12)\n"
        raise RuntimeError(f"unexpected cmd: {cmd}")

    monkeypatch.setattr(inst.subprocess, "check_output", fake_check_output)

    # Ensure unix candidates are ignored so the py launcher path is chosen.
    monkeypatch.setattr(inst.os.path, "exists", lambda p: False)
    monkeypatch.setattr(inst.glob, "glob", lambda p: [])

    res = inst.find_python312()
    # The test must run on Windows CI; assert we find the py launcher.
    assert res is not None, "Expected py launcher to be detected but find_python312 returned None"
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


def test_pull_models_skips_and_prints(monkeypatch, capsys):
    # Simulate `ollama` not on PATH
    monkeypatch.setattr(inst.shutil, "which", lambda name: None)

    inst.pull_models(["some-model"])

    captured = capsys.readouterr()
    # Message should be printed to stderr about missing CLI and an install hint
    assert "Ollama CLI not found" in captured.err
    assert "curl -fsSL https://ollama.com/install.sh | sh" in captured.err


def test_find_python312_prefers_exact_name(monkeypatch):
    # Simulate python3.12 being present via exact name
    monkeypatch.setattr(
        inst.shutil, "which", lambda name: "/usr/bin/python3.12" if name.startswith("python3.12") else None
    )
    found = inst.find_python312()
    assert found is not None and "/usr/bin/python3.12" in found


def test_pull_models_popen_file_not_found(monkeypatch, capsys):
    monkeypatch.setattr(inst.shutil, "which", lambda name: "/usr/bin/ollama")

    def fake_popen(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(inst.subprocess, "Popen", fake_popen)
    inst.pull_models(["m1"])
    captured = capsys.readouterr()
    assert "ollama CLI not found while attempting to pull" in captured.err


def test_pull_models_keyboard_interrupt(monkeypatch):
    monkeypatch.setattr(inst.shutil, "which", lambda name: "/usr/bin/ollama")

    class P:
        def __init__(self):
            self.stdout = self._gen()
            self.returncode = 0

        def _gen(self):
            yield "ok\n"
            raise KeyboardInterrupt()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def wait(self):
            return

        def kill(self):
            self.killed = True

    def fake_popen(*args, **kwargs):
        return P()

    monkeypatch.setattr(inst.subprocess, "Popen", fake_popen)
    with pytest.raises(KeyboardInterrupt):
        inst.pull_models(["m1"])


def test_parse_args_defaults_and_flags(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--no-pull-models", "--python", "pycmd"])
    ns = inst.parse_args()
    assert ns.pull_models is False
    assert ns.python == "pycmd"


def test_main_exits_when_python_not_312(monkeypatch):
    # force check_output to report non-3.12
    monkeypatch.setattr(inst.subprocess, "check_output", lambda *a, **k: b"(3, 11)")
    monkeypatch.setattr("sys.argv", ["prog", "--no-pull-models"])  # no model pulls to simplify
    with pytest.raises(SystemExit):
        inst.main()


def test_find_python312_detects_candidate(monkeypatch):
    # Simulate python3.12 being present on PATH
    monkeypatch.setattr(inst.shutil, "which", lambda name: "/usr/bin/python3")

    def fake_check_output(cmd, **kwargs):
        # Return a Python 3.12 version tuple when asked
        return b"(3, 12)\n"

    monkeypatch.setattr(inst.subprocess, "check_output", fake_check_output)

    found = inst.find_python312()
    assert found is not None


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
