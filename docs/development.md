## Development

### Install development packages

It is recommended to create and activate the project venv first, then install dev dependencies into the venv.

Note: macOS is not officially supported or tested by this project. The instructions target Linux and Windows; macOS users may try the Linux steps at their own risk.

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

### Pre-commit hooks (format, lint, type-check)

Ensure `.venv` is activated or prefix these commands with the venv Python (`./.venv/bin/` on Linux, `.\.venv\Scripts\` on Windows).

```bash
pre-commit install
pre-commit run --all-files
```

### Ruff lint and MyPy type check (mirrors CI)

```bash
ruff check src tests scripts
mypy --config-file=pyproject.toml src
```

### Smoke test (no network calls)

Run smoke tests using the venv Python so the environment matches CI (after activating `.venv` or by using the venv python explicitly):

Linux:

```bash
./.venv/bin/python -m scripts.smoke
```

Windows (PowerShell / cmd):

```powershell
.\.venv\Scripts\python -m scripts.smoke
```

### Pytest suite

Run tests using the venv Python to ensure the interpreter and installed deps match CI:

Linux:

```bash
./.venv/bin/python -m pytest -q
```

Windows (PowerShell / cmd):

```powershell
\.venv\Scripts\python -m pytest -q
```

### Run via module entrypoint (recommended)

Run using the venv Python (or the appropriate system Python if you prefer):

Linux:

```bash
./.venv/bin/python -m src.main --question "Hello"
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\python -m src.main --question "Hello"
```
