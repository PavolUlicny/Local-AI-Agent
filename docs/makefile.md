## Makefile usage & examples

This file documents common `Makefile` targets and examples for running the project locally.

### Quick overview

- `make help` — print available targets and short descriptions.
- `make venv` — create a local virtualenv at `.venv` (if missing) and upgrade `pip`.
- `make install` — create venv (if needed) and install `requirements.txt`.
- `make install-dev` — install runtime + development dependencies (`requirements-dev.txt`).
- `make install-deps` — wrapper that runs `python -m scripts.install_deps` (installer script).
- `make dev-setup` — `install` + `pull-model` (pulls configured Ollama models).
- `make pull-model` — run `ollama pull` for role models (overrides via `ROBOT_MODEL`/`ASSISTANT_MODEL`).
- `make serve-ollama` — run the Ollama server in the foreground (optional; the installer can start Ollama automatically).
- `make run` — run the interactive agent (`python -m src.main`).
- `make ask` — one-shot question (pass `QUESTION="..."`).
- `make run-no-search` — run one-shot with `--no-auto-search`.
- `make run-search` — run one-shot forcing search (`--force-search`).
- `make check` / `make check-ollama` — quick import and Ollama checks.
- `make smoke` — installs deps then runs the no-network smoke test.
- `make clean` — remove caches and `__pycache__` files.

### Passing flags / environment variables

Many Makefile targets accept variables that map directly to CLI flags. Example:

```bash
# run with custom knobs
make run QUESTION="What's the capital of France?" MAX_ROUNDS=2 SEARCH_MAX_RESULTS=3 DDG_BACKEND=duckduckgo LOG_LEVEL=INFO
```

Notes:

- Hyphenated CLI flags are mapped to `Makefile` variables by replacing `-` with `_`.
  For example `--max-rounds` => `MAX_ROUNDS` and `--assistant-model` => `ASSISTANT_MODEL`.
- Boolean flags are treated as truthy values in Make variables; set them to `1`, `true`, or `on`.
  Example: `make run NO_AUTO_SEARCH=1` will pass `--no-auto-search` to the CLI.
- Long values (paths) containing spaces should be quoted: `make run LOG_FILE="/tmp/my logs/agent.log"`.

### Examples

- Create venv and install runtime deps only:

```bash
make venv install
```

- Install runtime + dev deps (pinned) and pull models configured in Makefile env:

```bash
make install-dev dev-setup ROBOT_MODEL=cogito:8b ASSISTANT_MODEL=cogito:8b
```

- Pull specific role models without running the installer:

```bash
make pull-model ROBOT_MODEL=llama3:8b ASSISTANT_MODEL=llama3:8b
```

- Run the interactive agent (inherit default models & flags):

```bash
make run
# or with overrides
make run ASSISTANT_MODEL=cogito:8b ROBOT_MODEL=cogito:8b LOG_LEVEL=DEBUG
```

- Ask a single question and exit:

```bash
make ask QUESTION="What is 2+2?"
```

- Run a one-shot with web search forced:

```bash
make run-search QUESTION="Capital of France?" MAX_ROUNDS=1 SEARCH_MAX_RESULTS=2
```

- Run the smoke checks (installs then smoke):

```bash
make smoke
```

Windows / non-Make environments

- If you're on Windows without GNU `make`, use Git Bash, WSL, or run the underlying Python commands directly. Examples are present in `docs/install.md`.

### Maintenance & help

- Keep the `Makefile` target help text (`##`) accurate; `make help` parses and displays them.
- If you add new CLI flags, consider mapping them into the `Makefile` variables section (top of `Makefile`) so they are easily discoverable.

### Questions or changes

If you'd like, I can extend this page with a short table showing the most-used `Makefile` variables and their equivalent CLI flags, or add a `make ci-docs` target to run a markdown link-check across `docs/`.
