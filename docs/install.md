## Ollama runtime installation

This project requires the Ollama runtime. Follow these steps to install it:

### Linux

```bash
# Download and run the official installer script
curl -fsSL https://ollama.com/install.sh | sh
```

> **Security note:** verify installer scripts before piping to a shell; see `docs/security.md` for guidance.

### Windows

Download and run the Windows installer from the official site: [ollama.com](https://ollama.com)

## Project installation

Ensure the Ollama runtime is installed first.

### Python version (required)

**Important:** This project is tested on **Python 3.12**. The installer will automatically try to locate and prefer a Python 3.12 interpreter on your system (searches PATH, pyenv shims, and common install locations). You do not need to explicitly run `python3.12 -m scripts.install_deps` — simply running `python3 -m scripts.install_deps` (or `python -m scripts.install_deps` on Windows where `python` points to the desired interpreter) is sufficient when a 3.12 interpreter is present.

If you want to force a specific interpreter, pass `--python /path/to/python` to the installer. We do not attempt to install Python automatically; if Python 3.12 is not available, please install it (for example via your OS package manager, `pyenv`, or the official Windows installer) before running the installer.

Other Python versions (for example, 3.10–3.11 or future releases) are untested and may produce installation or runtime errors.

### Agent runtime requirement: Ollama

The interactive agent (the `src.main` entrypoint) requires the Ollama runtime at startup. When you run the agent it will check for the `ollama` CLI on your `PATH` and probe the local HTTP API at `http://127.0.0.1:11434/api/tags`.

- If the `ollama` CLI is present but the HTTP API is not responding, the agent will attempt to start the Ollama runtime in the background and wait briefly for the service to become available.
- If `ollama` is not installed, or the agent cannot start or connect to the Ollama HTTP API within the timeout, the agent will exit with an error to avoid running in a degraded state.

If you need to run parts of the project (for example in CI) without Ollama, use the installer flags (`--no-pull-models`) and Makefile helpers to avoid model pulls; consider running only the non-interactive tests or adding CI-specific guards. If you prefer a non-fatal startup for interactive runs, I can add an explicit CLI opt-out (for example `--allow-no-ollama`) that lets the agent continue without Ollama.

### Prerequisites

- Debian/Ubuntu: install the system venv helper so the installer can create `.venv`:

```bash
sudo apt update && sudo apt install -y python3-venv
```

### Automated install (recommended)

The repository includes `scripts/install_deps.py`, a small installer that:

- creates a local virtual environment at `.venv`;
- installs runtime dependencies from `requirements.txt` (and dev deps by default);
- optionally pulls Ollama models (defaults: `cogito:8b` and `embeddinggemma:300m`).

This installer is covered by a small unit test `tests/test_install_deps.py` which verifies detection of Python 3.12, behavior when `ollama` is missing, and the auto-start/poll logic (via mocks). See `docs/development.md` for how to run the tests locally.

You do NOT need to start the Ollama server before running the installer. If the `ollama` CLI is available on your `PATH` and the HTTP API is not responding, the installer will attempt to start a local Ollama daemon and wait briefly for it to become available. If you prefer to manage Ollama manually you can start it in another terminal, but this is optional — the installer will try to start Ollama when needed.

Main installer commands

Linux (recommended):

```bash
git clone https://github.com/PavolUlicny/Local-AI-Agent.git
cd Local-AI-Agent
python3 -m scripts.install_deps
source .venv/bin/activate
```

Windows (PowerShell / cmd):

```powershell
git clone https://github.com/PavolUlicny/Local-AI-Agent.git
cd Local-AI-Agent
python -m scripts.install_deps
.\.venv\Scripts\activate
```

### Installer options

- `--runtime-only`: install only `requirements.txt` (skip `requirements-dev.txt`).
- `--no-pull-models`: do not run `ollama pull` for any models.
- `--robot-model` / `--assistant-model` / `--embedding-model`: override which models to pull.
- `--python`: choose a different Python executable to create the venv.

#### Examples

```bash
# Install runtime deps only
python3 -m scripts.install_deps --runtime-only

# Install deps but do not pull models
python3 -m scripts.install_deps --no-pull-models

# Pull different models
python3 -m scripts.install_deps --robot-model "llama3:8b" --assistant-model "llama3:8b" --embedding-model "embeddinggemma:300m"
```

### Notes about model pulls

- The installer will call `ollama pull` for the main model and the embedding model unless you pass `--no-pull-models`. If the `ollama` CLI is not on your `PATH` the script prints a warning and skips pulls so the installer still succeeds.

### Troubleshooting & notes

- The installer prefers an existing Python 3.12 interpreter; it will not try to download or install Python for you. If Python 3.12 is not available install it via your OS package manager or `pyenv` and re-run the installer.
- If the installer attempted to start Ollama but pulls fail, check the installer log at `~/.local/share/ollama/installer_ollama.log` (if created) and start Ollama manually in another terminal to inspect output. The installer polls `http://127.0.0.1:11434/api/tags` for readiness when starting the daemon.
- Model pulls stream their output to your terminal so you can monitor download progress; failed pulls are reported as warnings and do not abort dependency installation.

### Manual install (alternative)

If you prefer to set up the environment manually, you can start the Ollama runtime yourself or let the installer start it for you when model pulls are requested. Example manual steps:

```bash
# Start Ollama in a separate terminal
ollama serve

# Create and activate venv (Linux)
git clone https://github.com/PavolUlicny/Local-AI-Agent.git
cd Local-AI-Agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

# Pull recommended models (optional; requires Ollama runtime)
ollama pull cogito:8b
ollama pull embeddinggemma:300m
```

## Makefile install-related targets

Use the `Makefile` targets for convenience. Examples:

```bash
# Create venv and install runtime deps
make venv install

# Install runtime + dev deps
make install-dev

# Pull role models
make pull-model ROBOT_MODEL=cogito:8b ASSISTANT_MODEL=cogito:8b

# Run the repository installer via Make (added helper)
make install-deps
```

## Quick start (minimal)

1. (Optional) Start Ollama in one terminal — this is optional because the installer will attempt to start Ollama automatically when needed.

2. In another terminal:

```bash
make install-deps
source .venv/bin/activate
```

## System Requirements

Minimum: Combined GPU VRAM + system RAM of at least 20 GB. Examples: 16 GB RAM + 4 GB VRAM, or 20 GB RAM CPU‑only (may rely on swap; expect slower inference).

Recommended: 25+ GB combined memory for smoother context handling and reduced swapping. Examples: 16 GB RAM + 10 GB VRAM, 32 GB RAM CPU‑only, or 24 GB RAM + 8 GB VRAM.

Notes:

- More memory allows larger `--num-ctx` and fewer automatic rebuild (halving) events.
- Python: Confirmed to run on Python 3.12 (tested in CI). Other Python versions (for example, 3.10–3.11 or future releases) are untested and not guaranteed to work.
- OS: Linux is expected to work. Windows is supported for the Ollama runtime; Python venv activation commands differ.
- If running CPU‑only, ensure fast SSD swap; avoid paging spikes by lowering `--num-predict` if memory pressure appears.
- Smaller GPUs (≤4 GB VRAM) can still run but may force model quantization or offload; keep expectations modest.
