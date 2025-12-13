#!/usr/bin/env python
"""Provision a local venv, install deps, and pull Ollama robot/assistant and embedding models by default."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence
import importlib

ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
DEFAULT_MODEL = "cogito:8b"
DEFAULT_EMBEDDING = "embeddinggemma:300m"


def run(cmd: Sequence[str]) -> None:
    """Echo and run a subprocess command."""
    print("→", " ".join(cmd))
    subprocess.check_call(cmd)


def venv_python() -> Path:
    """Return the venv's python path for POSIX/Windows."""
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    exe_name = "python.exe" if os.name == "nt" else "python"
    return VENV_DIR / bin_dir / exe_name


def ensure_venv(python_exe: str) -> Path:
    """Create .venv if missing and return its python path."""
    if not VENV_DIR.exists():
        run([python_exe, "-m", "venv", str(VENV_DIR)])
    py = venv_python()
    if not py.exists():
        raise RuntimeError(f"venv python missing at {py}")
    return py


def install_files(py: Path, files: Iterable[str]) -> None:
    for fname in files:
        path = ROOT / fname
        if not path.exists():
            print(f"skip missing {path}", file=sys.stderr)
            continue
        run([str(py), "-m", "pip", "install", "-r", str(path)])


def pull_models(models: Sequence[str]) -> None:
    if not models:
        return
    if not shutil.which("ollama"):
        print("ollama CLI not found; skip pulls", file=sys.stderr)
        return
    for model in models:
        run(["ollama", "pull", model])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_python = "python" if os.name == "nt" else sys.executable
    parser.add_argument(
        "--python",
        default=default_python,
        help=("Python interpreter used to create the venv (default: current; " "on Windows this defaults to 'python')"),
    )
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="Install only runtime deps (skip requirements-dev.txt)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pull-models",
        dest="pull_models",
        action="store_true",
        default=True,
        help="Pull configured role models after installing deps (default: on)",
    )
    group.add_argument(
        "--no-pull-models",
        dest="pull_models",
        action="store_false",
        help="Skip pulling Ollama models",
    )
    parser.add_argument(
        "--robot-model",
        default=None,
        help=(
            "Robot (planning/classifier) Ollama model to pull. If omitted, read from\n"
            "the project's configuration (`src.config.AgentConfig.robot_model`)."
        ),
    )
    parser.add_argument(
        "--assistant-model",
        default=None,
        help=(
            "Assistant (final answer) Ollama model to pull. If omitted, read from\n"
            "the project's configuration (`src.config.AgentConfig.assistant_model`)."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "Embedding model to pull. If omitted, read from the project's config\n"
            "(`src.config.AgentConfig.embedding_model`) or fall back to a built-in default."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Attempt to read configured defaults from the project's AgentConfig (if available).
    cfg = None
    # Ensure the repository root is on sys.path so `src.config` is importable
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    try:
        cfg_mod = importlib.import_module("src.config")
    except ModuleNotFoundError:
        try:
            cfg_mod = importlib.import_module("config")
        except Exception:
            cfg_mod = None
    if cfg_mod is not None and hasattr(cfg_mod, "AgentConfig"):
        try:
            # Direct attribute access is clearer and no less safe here than getattr.
            cfg = cfg_mod.AgentConfig()
        except Exception:
            cfg = None

    py = ensure_venv(args.python)
    run([str(py), "-m", "pip", "install", "-U", "pip"])

    files = ["requirements.txt"]
    if not args.runtime_only:
        files.append("requirements-dev.txt")
    install_files(py, files)

    if args.pull_models:
        # Resolve final model names: prefer CLI args, then project config, then built-in defaults.
        robot_model = args.robot_model or (getattr(cfg, "robot_model", None) if cfg else DEFAULT_MODEL)
        assistant_model = args.assistant_model or (getattr(cfg, "assistant_model", None) if cfg else DEFAULT_MODEL)
        embedding_model = args.embedding_model or (getattr(cfg, "embedding_model", None) if cfg else DEFAULT_EMBEDDING)

        # Pull robot + assistant models and the embedding model. Use a deterministic order
        # and de-duplicate identical model names.
        to_pull = []
        for m in (robot_model, assistant_model, embedding_model):
            if m and m not in to_pull:
                to_pull.append(m)
        # Print resolved model names for clarity before pulling.
        print(
            "Resolved models to pull:",
            "robot=",
            robot_model,
            "assistant=",
            assistant_model,
            "embedding=",
            embedding_model,
        )
        pull_models(to_pull)


if __name__ == "__main__":
    main()
