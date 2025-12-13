## Ollama runtime installation

This project requires the Ollama runtime. Follow these steps to install it:

### Linux

```bash
# Download and run the official installer script
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows

Download and run the Windows installer from the official site: [ollama.com](https://ollama.com)

## Project installation

Ensure the Ollama runtime is installed first.

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

Start the Ollama server in one terminal:

```bash
ollama serve
```

Then run the installer from a second terminal (Linux):

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

- The installer will call `ollama pull` for the main model and the embedding model unless
  you pass `--no-pull-models`. If the `ollama` CLI is not on your `PATH` the script prints a
  warning and skips pulls so the installer still succeeds.

### Manual install (alternative)

If you prefer to set up the environment manually:

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

# Pull recommended models (optional)
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

1. Start Ollama in one terminal:

```bash
ollama serve
```

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
- Python: tested in CI with Python 3.12. Earlier 3.10–3.11 may work but are not guaranteed.
- OS: Linux is expected to work. Windows is supported for the Ollama runtime; Python venv activation commands differ.
- If running CPU‑only, ensure fast SSD swap; avoid paging spikes by lowering `--num-predict` if memory pressure appears.
- Smaller GPUs (≤4 GB VRAM) can still run but may force model quantization or offload; keep expectations modest.
