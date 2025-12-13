## Development

Install development packages:

```bash
pip install -r requirements-dev.txt
```

Pre-commit hooks (format, lint, type-check):

```bash
pre-commit install
pre-commit run --all-files
```

Ruff lint and MyPy type check (mirrors CI):

```bash
ruff check src tests scripts
mypy --config-file=pyproject.toml src
```

Smoke test (no network calls):

Linux:

```bash
python3 -m scripts.smoke
```

Windows:

```powershell
python -m scripts.smoke
```

Pytest suite:

```bash
pytest
```

Run via module entrypoint (recommended):

```bash
python3 -m src.main --question "Hello"
```
