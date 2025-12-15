#!/usr/bin/env python
"""Provision a local venv, install deps, and pull Ollama robot/assistant and embedding models by default."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import platform
import glob
from pathlib import Path
from typing import Iterable, Sequence
import importlib
import time
import urllib.request
import urllib.error
import socket

ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
DEFAULT_MODEL = "cogito:8b"
DEFAULT_EMBEDDING = "embeddinggemma:300m"


def run(cmd: Sequence[str]) -> None:
    """Echo and run a subprocess command."""
    print("→", " ".join(cmd))
    subprocess.check_call(cmd)


def venv_python() -> Path:
    """Return the venv's python path for Linux/Windows."""
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
        print("→", "ollama", "pull", model)
        # Start the pull process and stream its combined stdout/stderr live so
        # the user sees progress as it happens (large model downloads).
        try:
            with subprocess.Popen(
                ["ollama", "pull", model], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            ) as proc:
                # Stream output line-by-line as it becomes available
                if proc.stdout is not None:
                    for line in proc.stdout:
                        print(line, end="")
                proc.wait()
                ret = proc.returncode
        except KeyboardInterrupt:
            print("Interrupted; terminating pull", file=sys.stderr)
            try:
                proc.kill()
            except Exception:
                pass
            raise
        except FileNotFoundError:
            print("ollama CLI not found while attempting to pull; skipping.", file=sys.stderr)
            return
        except Exception as e:
            print(f"Error running ollama pull: {e}", file=sys.stderr)
            # don't abort the installer on unexpected subprocess errors
            continue

        if ret != 0:
            print(f"Warning: failed to pull {model} (exit {ret})", file=sys.stderr)
            continue


def ollama_server_ready(host: str = "127.0.0.1", port: int = 11434, timeout: float = 1.0) -> bool:
    """Return True if the Ollama HTTP API responds at /api/tags.

    Uses a short timeout; intended for polling readiness.
    """
    url = f"http://{host}:{port}/api/tags"
    try:
        # If we can open the endpoint without raising, consider the server ready.
        # Avoid returning attributes from a loosely-typed HTTPResponse (which
        # may be `Any`) to satisfy static type checkers.
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except urllib.error.URLError:
        return False
    except socket.timeout:
        return False


def start_ollama_if_needed(wait_seconds: int = 30) -> None:
    """If `ollama` is on PATH and the server isn't responding, start it.

    Start attempts run detached so this script can continue; we poll until
    `wait_seconds` to check readiness. If starting fails, we print a warning
    and continue — pulls will be skipped if the server never comes up.
    """
    if not shutil.which("ollama"):
        # Nothing to do when ollama CLI isn't installed
        return

    if ollama_server_ready():
        print("Ollama server already responding; no need to start.")
        return

    print("ollama CLI found on PATH but server not responding — attempting to start 'ollama serve' in background...")
    # Start ollama serve detached
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        log = open(os.path.expanduser("~/.local/share/ollama/installer_ollama.log"), "a+")
    except Exception:
        log = None

    try:
        proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=log or subprocess.DEVNULL,
            stderr=log or subprocess.STDOUT,
            start_new_session=True,
            creationflags=creationflags,
        )
        # Reference the process so linters won't complain about an unused
        # assignment; also print the PID for diagnostics.
        pid = getattr(proc, "pid", None)
        if pid:
            print(f"Started ollama serve (pid {pid})")
    except FileNotFoundError:
        print("Failed to start Ollama: 'ollama' not found", file=sys.stderr)
        return
    except Exception as e:
        print(f"Failed to start Ollama: {e}", file=sys.stderr)
        return

    # Poll for readiness until timeout
    deadline = time.time() + float(wait_seconds)
    while time.time() < deadline:
        if ollama_server_ready():
            print("Ollama server is up and responding.")
            if log:
                log.close()
            return
        time.sleep(1)

    print(f"Warning: Ollama did not become ready within {wait_seconds}s. Check logs or run 'ollama serve' manually.")
    if log:
        log.close()


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


def find_python312() -> str | None:
    """Return a path to a Python 3.12 interpreter if available.

    Search order:
    - exact `python3.12` / `python3.12.exe` on PATH
    - `python3` / `python` on PATH but only if its version is 3.12
    - pyenv shims and pyenv versions
    - common Unix locations (`/usr/bin`, `/usr/local/bin`, `~/.local/bin`)
    - common Windows install locations under %LOCALAPPDATA% and Program Files
    """
    # 1) Exact names on PATH
    exe_names = ["python3.12", "python3.12.exe"]
    for name in exe_names:
        path = shutil.which(name)
        if path:
            return path

    # 2) Look for python3 or python and check version
    for candidate in (shutil.which("python3"), shutil.which("python")):
        if candidate:
            try:
                out = (
                    subprocess.check_output([candidate, "-c", "import sys; print(sys.version_info[:2])"])
                    .decode()
                    .strip()
                )
                if "(3, 12)" in out or "3, 12" in out:
                    return candidate
            except Exception:
                continue

    # 3) pyenv shims / versions
    try:
        home = os.path.expanduser("~")
        pyenv_shim = os.path.join(home, ".pyenv", "shims", "python3.12")
        if os.path.exists(pyenv_shim):
            return pyenv_shim
        for p in glob.glob(os.path.join(home, ".pyenv", "versions", "3.12*")):
            candidate = os.path.join(p, "bin", "python3.12")
            if os.path.exists(candidate):
                return candidate
    except Exception:
        pass

    # 4) Common Unix locations
    unix_candidates = [
        "/usr/bin/python3.12",
        "/usr/local/bin/python3.12",
        os.path.expanduser("~/.local/bin/python3.12"),
    ]
    for c in unix_candidates:
        if c and os.path.exists(c):
            return c

    # 5) Common Windows locations
    if platform.system().lower().startswith("win"):
        localapp = os.environ.get("LOCALAPPDATA")
        programfiles = os.environ.get("ProgramFiles")
        programfiles_x86 = os.environ.get("ProgramFiles(x86)")
        candidates = []
        if localapp:
            candidates.append(os.path.join(localapp, "Programs", "Python", "Python312", "python.exe"))
            candidates.append(os.path.join(localapp, "Microsoft", "WindowsApps", "python3.12.exe"))
        if programfiles:
            candidates.append(os.path.join(programfiles, "Python312", "python.exe"))
            candidates.append(os.path.join(programfiles, "Python", "Python312", "python.exe"))
        if programfiles_x86:
            candidates.append(os.path.join(programfiles_x86, "Python312", "python.exe"))
        for c in candidates:
            if c and os.path.exists(c):
                return c

    return None


def main() -> None:
    args = parse_args()

    # If the user didn't explicitly pass a custom python, prefer any discovered
    # Python 3.12 on the system (PATH, pyenv, common locations). If the user
    # supplied `--python`, respect that explicit choice.
    default_python = "python" if os.name == "nt" else sys.executable
    if args.python == default_python:
        py312 = find_python312()
        if py312:
            print(f"Found Python 3.12 at: {py312} — using it to create the venv")
            args.python = py312

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
        # If `ollama` is present on PATH try to start it so pulls can succeed.
        start_ollama_if_needed()
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
