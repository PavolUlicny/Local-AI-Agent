## Ollama runtime installation

This project requires the Ollama runtime. Follow these steps to install it.

Note: macOS is not officially supported or tested by this project. The documentation and CI target Linux and Windows; if you must run on macOS you may try the Linux instructions at your own risk.

### Linux

On Linux you can use the official installer script. Prefer a two-step download + review pattern rather than piping directly to a shell:

```bash
# Download the installer, inspect it, then run it if you trust it
curl -fsSL -o ollama_install.sh https://ollama.com/install.sh
less ollama_install.sh    # inspect the script
sh ollama_install.sh
```

> Security note: avoid piping unfamiliar install scripts directly to a shell without review. See `docs/security.md` for guidance.

### Windows

On Windows download and run the official installer from [ollama.com](https://ollama.com). Modern Windows includes PowerShell which the installer supports.
If you prefer a CLI installer or have automated provisioning, download the MSI/installer manually and verify the publisher before running.

## Project installation

Ensure the Ollama runtime is installed first (the interactive agent requires it at startup).

### Python version (required)

Important: this project is tested on Python 3.12. The installer (`scripts/install_deps.py`) will attempt to locate a suitable Python 3.12 interpreter automatically by checking:

- exact `python3.12` / `python3.12.exe` on PATH
- the Windows `py` launcher (`py -3.12`) when available
- `python3` / `python` on PATH (only accepted if its version is 3.12)
- common `pyenv` shims and typical install locations

You do not need to run a very specific command to start the installer — use whichever `python` command on your system resolves to the intended interpreter. Examples:

- On Linux where `python3` is your 3.12 interpreter:

```bash
python3 -m scripts.install_deps
```

- On Windows where `python` may be the correct interpreter or you prefer the `py` launcher:

```powershell
python -m scripts.install_deps
```

If you want to force a specific interpreter, pass `--python /path/to/python` (or on Windows an explicit path like `--python "C:\\Python312\\python.exe"`) to the installer. The installer will verify the chosen interpreter is Python 3.12 and will exit with an actionable error if not.

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

The installer is covered by `tests/test_install_deps.py` which exercises Python detection and the Ollama start/poll logic via mocks. See `docs/development.md` for running tests locally.

You do NOT need to start the Ollama server before running the installer. If the `ollama` CLI is on your `PATH` and the HTTP API at `http://127.0.0.1:11434` is not responding, the installer will attempt to start the Ollama daemon and wait for it to become available. If you prefer to manage Ollama manually, start it in another terminal; otherwise the installer will attempt to start it when required for pulling models.

Main installer commands (examples)

Linux (recommended):

```bash
git clone https://github.com/PavolUlicny/Local-AI-Agent.git
cd Local-AI-Agent
python3 -m scripts.install_deps
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
git clone https://github.com/PavolUlicny/Local-AI-Agent.git
cd Local-AI-Agent
python -m scripts.install_deps
.\.venv\Scripts\Activate.ps1
```

Windows (cmd.exe):

```cmd
git clone https://github.com/PavolUlicny/Local-AI-Agent.git
cd Local-AI-Agent
python -m scripts.install_deps
\.venv\Scripts\activate.bat
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
- If the installer attempted to start Ollama but pulls fail, check the installer log if created. On Linux this is commonly at `~/.local/share/ollama/installer_ollama.log`; on Windows check your `%LOCALAPPDATA%` Ollama folder or consult the Ollama runtime docs for the exact location. Start Ollama manually in another terminal to inspect output. The installer polls `http://127.0.0.1:11434/api/tags` for readiness when starting the daemon.
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

- The RAM/VRAM examples above are approximate; actual memory requirements depend on model size, quantization, and workload.

- More memory allows larger `--num-ctx` and fewer automatic rebuild (halving) events.
- Python: Confirmed to run on Python 3.12 (tested in CI). Other Python versions (for example, 3.10–3.11 or future releases) are untested and not guaranteed to work.
- OS: Linux is expected to work. Windows is supported for the Ollama runtime; Python venv activation commands differ.
- If running CPU‑only, ensure fast SSD swap; avoid paging spikes by lowering `--num-predict` if memory pressure appears.
- Smaller GPUs (≤4 GB VRAM) can still run but may force model quantization or offload; keep expectations modest.
