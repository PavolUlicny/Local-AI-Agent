## Continuous Integration (CI)

The repository includes a GitHub Actions workflow at `.github/workflows/ci.yml` that runs on pushes and pull requests across branches (the workflow runs on push and pull_request events for all branches).

CI performs the following high-level steps:

- Checkout repository
- Set up Python 3.12
- Install runtime (and optionally dev) dependencies
- Run pre-commit hooks (`pre-commit run --all-files`)
- Run a no-network smoke test
- Run `pytest`

For reproducibility and to avoid downloading large models in CI, the workflow invokes the installer with explicit flags to disable model pulls. The recommended CI command is:

```bash
# Install dependencies but do NOT pull Ollama models (fast, deterministic)
python -m scripts.install_deps --no-pull-models
```

Notes and best practices for CI:

- Use the `--no-pull-models` flag in CI to avoid network-heavy model downloads and to keep runs fast and cache-friendly.
- Cache the project venv or pip wheel cache between runs where your CI provider supports it to speed up installs.
- Run `pre-commit` and `ruff`/`mypy` in CI to catch formatting and type errors early.
- The CI workflow is intentionally conservative: it installs deps without pulling models, runs smoke tests (no-network), then runs the full test suite.

Locally, you can emulate CI by following the Development section and running the same linters and tests.
