## Troubleshooting

Note: Linux and Windows are the primary supported platforms and are the focus of CI testing. macOS may work but is not officially tested here; adapt the POSIX instructions as needed and file an issue if you hit platform-specific problems.

| Symptom | Cause | Action |
| --------- | ------- | -------- |
| `Model 'xyz' not found` | Ollama model not pulled | `ollama pull xyz` then retry. |
| `Assistant model 'xyz' not found. Run 'ollama pull xyz' and retry.` | Agent detected a missing Ollama model (either at install-time or runtime) | Pull the model with `ollama pull xyz` or set an alternative model via `--assistant-model` / `--robot-model`. If this happens during automated runs (installer or CI) use `--no-pull-models` to skip pulls and inspect logs for the exact failure reason. |
| Frequent context rebuild logs | Oversized conversation or results | Reduce `--max-rounds`, `--max-context-turns`, or reduce `--assistant-num-ctx` / `--robot-num-ctx` to avoid hitting the model context window. |
| Many rate limit warnings | DDGS provider throttling | Lower concurrency (accept defaults) or narrow `--ddg-backend` to a single engine like `duckduckgo`. |
| Empty search results | Provider mix mismatch | Retry with a specific backend (`--ddg-backend duckduckgo`/`bing`) or broaden query phrasing. |
| No new suggestions | Planning chain conservative or truncation | Increase `--max-followup-suggestions` or verify not hitting truncation caps. |
| `ModuleNotFoundError: No module named 'langchain_community'` | Wrong Python interpreter or deps not installed | Activate the project venv and reinstall dependencies. Linux: `source .venv/bin/activate && pip install -r requirements.txt` or `./.venv/bin/python -m pip install -r requirements.txt`; run via `./.venv/bin/python -m src.main`. Windows (PowerShell): `.\.venv\Scripts\Activate.ps1; pip install -r requirements.txt` or `.\.venv\Scripts\python -m pip install -r requirements.txt`; run via `.\.venv\Scripts\python -m src.main`. |
| `pre-commit` alters files locally | Hooks include auto-fixers (ruff). Re-run `git add` after fixes. | Use `pre-commit run --all-files --show-diff-on-failure` in CI modes. |
| `--log-file` path with spaces fails | Makefile quoting added, but direct CLI still needs quotes | Use `--log-file "/path/with spaces/agent.log"`. |

## Known Limitations

- DDGS provider scraping can vary over time; identical queries may yield different snippets or ordering across runs.
- The assistant model uses nonzero temperature by default for answers, so responses are not bit‑for‑bit deterministic.
- No JavaScript execution or full page rendering is performed; only titles, URLs, and snippet text are used.
- Topic selection does not perform context‑length rebuilds; if it fails, the system proceeds without selecting a topic for that turn.
- No persistent storage of conversations beyond process lifetime; restarting the program resets topics and history.
