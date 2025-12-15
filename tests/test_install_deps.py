import scripts.install_deps as inst


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
