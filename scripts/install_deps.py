#!/usr/bin/env python
"""Provision a local venv, install deps, and pull Ollama models by default."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

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
    return VENV_DIR / bin_dir / "python"


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
        help="Pull Ollama models after installing deps (default: on)",
    )
    group.add_argument(
        "--no-pull-models",
        dest="pull_models",
        action="store_false",
        help="Skip pulling Ollama models",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Main Ollama model to pull (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING,
        help=(f"Embedding model to pull (default: {DEFAULT_EMBEDDING})"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    py = ensure_venv(args.python)
    run([str(py), "-m", "pip", "install", "-U", "pip"])

    files = ["requirements.txt"]
    if not args.runtime_only:
        files.append("requirements-dev.txt")
    install_files(py, files)

    if args.pull_models:
        pull_models([args.model, args.embedding_model])


if __name__ == "__main__":
    main()
