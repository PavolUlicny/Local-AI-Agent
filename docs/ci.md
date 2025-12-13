## Continuous Integration (CI)

The repository includes a GitHub Actions workflow at `.github/workflows/ci.yml` that runs on pushes and pull requests to `main`.

CI steps include:

- Checkout repository
- Set up Python 3.12
- Install dependencies via `scripts.install_deps` (no model pulls in CI)
- Run pre-commit hooks (`pre-commit run --all-files`)
- Run smoke test (no-network)
- Run `pytest`

Locally, you can emulate CI by following the Development section and running the same linters and tests.
