from __future__ import annotations

import importlib


def test_pull_models_skips_when_no_ollama(capsys, monkeypatch):
    # Ensure module is loaded and mutate its global `ollama`
    mod = importlib.import_module("scripts.install_deps")
    monkeypatch.setattr(mod, "ollama", None)
    mod.pull_models(["foo:model"])
    captured = capsys.readouterr()
    assert "Ollama CLI not found" in captured.err


def test_pull_models_skips_when_ollama_not_installed(capsys, monkeypatch):
    mod = importlib.import_module("scripts.install_deps")

    class Dummy:
        @staticmethod
        def is_installed():
            return False

    monkeypatch.setattr(mod, "ollama", Dummy())
    mod.pull_models(["a"])
    captured = capsys.readouterr()
    assert "Ollama CLI not found" in captured.err


def test_find_python312_prefers_exact_name(monkeypatch):
    mod = importlib.import_module("scripts.install_deps")

    def fake_which(name):
        if name == "python3.12":
            return "/usr/bin/python3.12"
        return None

    monkeypatch.setattr(mod.shutil, "which", fake_which)
    assert mod.find_python312() == "/usr/bin/python3.12"


def test_find_python312_checks_candidate_version(monkeypatch):
    mod = importlib.import_module("scripts.install_deps")

    def fake_which(name):
        if name == "python3.12":
            return None
        if name == "python3":
            return "/usr/bin/python3"
        return None

    def fake_check_output(cmd, stderr=None):
        return b"(3, 12)"

    monkeypatch.setattr(mod.shutil, "which", fake_which)
    monkeypatch.setattr(mod.subprocess, "check_output", fake_check_output)
    assert mod.find_python312() == "/usr/bin/python3"
