from __future__ import annotations

import importlib


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
