# Local AI Agent

> An extensible, self-steering local LLM assistant that can optionally enrich answers with iterative, relevance‑filtered DDGS (DuckDuckGo/Bing/Brave) web searches while tracking multi‑topic conversational context.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Core Architecture](#core-architecture)
4. [Project Structure](#project-structure)
5. [Execution Pipeline](#execution-pipeline)
# Local AI Agent

An extensible, self-steering local LLM assistant that can optionally enrich answers with iterative, relevance-filtered web searches while tracking multi-topic conversational context.

Quick start

```bash
# Create venv, install deps, and pull configured models (recommended)
make install-deps

# Start Ollama server in another terminal
ollama serve

# Activate venv and run one-shot
source .venv/bin/activate
python -m src.main --question "Hello"
```

Documentation

Full documentation has been moved to the `docs/` folder. See `docs/index.md` for a table of contents and detailed guides (installation, architecture, usage, development, CI, security, troubleshooting, and contributing).

License: MIT — see `LICENSE` for full text.

### `src/agent.py`

Owns the runtime orchestration: topic selection, search decision, seed generation, iterative query planning/filtering, search execution, truncation, answer synthesis with/without search context, interactive loop, and resilience (context‑length rebuilds). It now delegates distinct responsibilities to `SearchClient`, `SearchOrchestrator`, `EmbeddingClient`, and `TopicManager` while maintaining the overall flow and rebuild counters.

### `src/search_client.py`

Wraps DDGS/Brave/Bing queries with retry/backoff, timeout management, and result normalization before the orchestration loop inspects them.

### `src/embedding_client.py`

Encapsulates Ollama embedding lifecycle, including caching, error logging, and safe fallbacks whenever the runtime cannot return a vector.

### `src/search_orchestrator.py`

Contains the pending-query loop that deduplicates URLs, runs relevance filters, expands keyword sets, and triggers planning/fill passes until round budgets are satisfied.

### `src/topic_manager.py`

Manages topic creation/updating, keyword pruning, embedding blending, and selection bookkeeping once a turn is answered.

### `src/text_utils.py`

Holds truncation policies, regex validation helpers, datetime formatting, and logic for normalizing or selecting seed queries.

### `src/keywords.py`

Implements tokenization, stopword filtering, keyword extraction/pruning, heuristic follow-up detection, and lightweight relevance scoring.

### `src/topic_utils.py`

Defines the `Topic` dataclass, cosine similarity helpers, turn tail selection, topic briefs, and prior-response aggregation utilities.

### `src/url_utils.py`

Provides URL canonicalization helpers to support consistent deduplication hashes.

## Topic & Context Management

- Each topic retains a list of `(user, assistant)` turns.
- On each new query, `_select_topic` attempts to classify relationship (FOLLOW_UP / EXPAND / NEW_TOPIC) via a dedicated chain. If no sufficiently overlapping keywords, a new topic begins.
- A semantic embedding (default: `embeddinggemma:300m`) is blended into each topic so cosine similarity can rescue related follow-ups even when keywords diverge.
- Topics exceeding `MAX_TOPICS` are pruned FIFO.
- Turn history per topic is trimmed to `max_context_turns * 2` preserving a sliding window.
- Keywords aggregated from user query, assistant answer, and truncated search result text; pruned to `MAX_TOPIC_KEYWORDS` by frequency ordering.
- Search result snippets and follow-up queries also leverage embeddings: high-similarity snippets can bypass the LLM relevance gate, while clearly off-topic query suggestions are dropped before invoking the classifier.

## Search Strategy & Filtering

1. Decide necessity (SEARCH vs NO_SEARCH).
2. Generate a seed query (fallback to original if generation fails).
3. Maintain a queue of pending queries (seed + planned follow‑ups) up to `--max-rounds`.
4. For each query:

- Fetch `--search-max-results` results (with retry/backoff).
- Deduplicate by URL (canonicalized) and SHA‑256 hash of assembled title/URL/snippet.
- Apply fast keyword intersection relevance; escalate borderline cases to LLM (YES/NO) within `--max-relevance-llm-checks` budget.
- Expand topic keyword set from accepted results.
- Plan new queries; gate each with query filter classifier.
- Perform fill cycles (additional planning passes) until either round capacity or fill attempt limits reached.
- Only queries that yield ≥1 accepted result consume a round; queries with zero accepted results are discarded without decrementing remaining rounds.

5. Truncate aggregated corpus to `MAX_SEARCH_RESULTS_CHARS` before answer synthesis.

## Robustness & Error Handling

- Graceful handling of missing Ollama model (`ollama pull <model>` hint logged).
- Detects context length / token window errors and performs controlled recovery: progressively halves `num_ctx` (never below 2048), caps `num_predict` at ≤ 50% of the updated context (never below 512), rebuilds affected chains, and limits attempts per stage via `MAX_REBUILD_RETRIES`.
- Retries / exponential backoff for DDGS providers (rate limit or transient failures) with jitter.
- Fallback paths: failing seed → use original query; failing planning → skip suggestions; failing relevance → drop result.
- Deduplication: canonicalized URLs + SHA‑256 hash of assembled title/URL/snippet.
- Classifier defaults:
  - Search decision: malformed/exceptional outputs default to SEARCH; exhausted context after rebuilds falls back to NO_SEARCH.
  - Query filter/result filter: default to NO (conservative).
  - Clarification: A "malformed output" is text that fails regex validation; this differs from an execution error/exception. Malformed → default token; error/exception → fallback behaviors above.
- Context-length rebuilds are implemented for: search decision, seed generation, result relevance checks, planning, query filter, and final answer stages. Topic selection (context classification) does not rebuild on context errors; it proceeds without selecting a topic when needed.

## CLI Arguments

| Flag | Short | Default | Description |
| ------ | ------- | --------- | ------------- |
| `--no-auto-search` | `--nas` | off | Force NO_SEARCH unless logic changed manually. |
| `--force-search` | `--fs` | off | Bypass classifier and always perform SEARCH for the current turn. |
| `--max-rounds` | `--mr` | `12` | Upper bound on search query rounds (seed + planned). |
| `--max-context-turns` | `--mct` | `8` | Turns retained from a topic for context injection. |
| `--max-followup-suggestions` | `--mfs` | `6` | Max query suggestions per planning cycle. |
| `--max-fill-attempts` | `--mfa` | `3` | Extra planning passes to fill remaining slots. |
| `--max-relevance-llm-checks` | `--mrlc` | `2` | LLM relevance validations for borderline results per query. |
| `--assistant-num-ctx` | `--anc` | `8192` | Context window tokens for assistant chains. |
| `--robot-num-ctx` | `--rnc` | `8192` | Context window tokens for robot (classifier/planner) chains. |
| `--assistant-num-predict` | `--anp` | `4096` | Generation cap for assistant chains. |
| `--robot-num-predict` | `--rnp` | `512` | Generation cap for classifier/planner chains (keeps them fast/cheap). |
| `--robot-temp` | `--rt` | `0.0` | Temperature for classifier/planner chains. |
| `--assistant-temp` | `--at` | `0.6` | Temperature for final answer chain. |
| `--robot-top-p` | `--rtp` | `0.4` | Top-p for robot model. |
| `--assistant-top-p` | `--atp` | `0.8` | Top-p for assistant model. |
| `--robot-top-k` | `--rtk` | `20` | Top-k for robot model. |
| `--assistant-top-k` | `--atk` | `80` | Top-k for assistant model. |
| `--robot-repeat-penalty` | `--rrp` | `1.1` | Repeat penalty robot chain. |
| `--assistant-repeat-penalty` | `--arp` | `1.2` | Repeat penalty assistant chain. |
| `--ddg-region` | `--dr` | `us-en` | DDGS regional hint forwarded to providers. |
| `--ddg-safesearch` | `--dss` | `moderate` | DDGS safe search level. |
| `--ddg-backend` | `--db` | `auto` | DDGS backend(s): `auto`, `duckduckgo`, `bing`, `brave`, ... |
| `--search-max-results` | `--smr` | `5` | Result fetch count per query. |
| `--search-retries` | `--sr` | `3` | Max retries for failed search (shorter backoff to keep the UI responsive). |
| `--search-timeout` | `--st` | `10.0` | Per-request DDGS timeout in seconds. |
| `--log-level` | `--ll` | `WARNING` | Logging verbosity. |
| `--log-file` | `--lf` | `None` | Optional file log path. |
| `--log-console` | `--lc` | `on` | Emit logs to stderr when enabled; pass `--no-log-console` to keep the console clean. Without `--log-file`, logs are discarded. |
| `--question` | `--q` | `None` | One-shot non-interactive question mode. |
| `--embedding-model` | `--em` | `embeddinggemma:300m` | Ollama embedding model used for topic similarity checks. |
| `--embedding-similarity-threshold` | `--est` | `0.35` | Minimum cosine similarity for a topic to be considered when no keywords overlap. |
| `--embedding-history-decay` | `--ehd` | `0.65` | Weight [0-1) that keeps prior topic embeddings when blending in a new turn (lower = faster adaptation). |
| `--embedding-result-similarity-threshold` | `--erst` | `0.5` | Semantic similarity needed for a search result to skip the LLM relevance gate. |
| `--embedding-query-similarity-threshold` | `--eqst` | `0.3` | Minimum similarity before a planned query is passed to the LLM query filter. |
| `--robot-model` | `--rm` | `cogito:8b` | Ollama model for robot (planning/classifier) chains. |
| `--assistant-model` | `--am` | `cogito:8b` | Ollama model for assistant (final answer) chains. |

## Using Different Models

You can substitute any Ollama model by pulling it first and passing role-specific flags. Examples:

- Linux:

```bash
ollama pull llama3:8b   # example
python3 -m src.main --robot-model llama3:8b --assistant-model llama3:8b
```

- Windows:

```bash
ollama pull llama3:8b
python -m src.main --robot-model llama3:8b --assistant-model llama3:8b
```

Key adjustments when switching models:

- Context window (`--assistant-num-ctx` / `--robot-num-ctx`): Must not exceed the model's maximum (see the [Ollama model catalog](https://ollama.com/library) for limits). Common defaults:
  - Llama 3 (8B/7B): 8192
  - Mistral 7B: 8192
  - Phi / small instruct models: 4096–8192
  - Some extended variants (e.g., 32K finetunes) allow larger windows; set `--num-ctx` accordingly.
- Predict length (`--num-predict`): Keep this comfortably below available combined memory. Large values increase RAM/VRAM footprint linearly with active layers.
- Memory tradeoff: If you hit automatic halving rebuilds early, lower the assistant/robot context (`--assistant-num-ctx` / `--robot-num-ctx`) first; only then reduce `--assistant-num-predict` / `--robot-num-predict`.
- Quantization: Using a more heavily quantized model (e.g., Q4_K_M) reduces memory, allowing higher `--num-ctx` without rebuilds.
- Robot vs Assistant: Temperatures (`--robot-temp`, `--assistant-temp`) and top‑p/k are independent of model choice; adjust only for style. Some models need slightly lower `--assistant-top-p` to avoid repetition.
- Smaller models (<4B) may benefit from reducing `--num-predict` (e.g., 2048–4096) to preserve responsiveness.
- You may also need to adjust prompts for some models.

Search backend notes:

- `--ddg-backend auto` (default): lets DDGS fan out across multiple providers (DuckDuckGo, Bing, Brave, etc.) and deduplicate merged results.
- `--ddg-backend duckduckgo`: constrain to DuckDuckGo-derived engines for deterministic snippets.
- `--ddg-backend bing` / `brave` / other provider names: target a single upstream engine when you want stable formatting or to avoid throttling another vendor.

Recommended tuning workflow:

1. Start with model's documented max context (e.g., 8192).
2. Run a few multi‑round searches; if rebuild logs appear, drop `--num-ctx` by ~25%.
3. If still rebuilding, cap `--num-predict` at half of current (`8192 → 4096`).
4. Raise again incrementally once stable.

Model examples:

```text
llama3:8b        --num-ctx 8192   --num-predict 4096–8192
mistral:7b       --num-ctx 8192   --num-predict 4096–8192
phi4-mini:3.8b   --num-ctx 4096   --num-predict 2048–4096
codellama:7b     --num-ctx 8192   --num-predict 4096–8192 (may lower temp for code)
```

If you exceed model context, you'll see context errors and forced halving; better to preconfigure than rely on repeated rebuild cycles.

## System Requirements

Minimum: Combined GPU VRAM + system RAM of at least 20 GB.
Examples: 16 GB RAM + 4 GB VRAM, or 20 GB RAM CPU‑only (may rely on swap; expect slower inference).

Recommended: 25+ GB combined memory for smoother context handling and reduced swapping.
Examples: 16 GB RAM + 10 GB VRAM, 32 GB RAM CPU‑only, or 24 GB RAM + 8 GB VRAM.

Notes:

- More memory allows larger `--num-ctx` and fewer automatic rebuild (halving) events.
- Python: tested in CI with Python 3.12. Earlier 3.10–3.11 may work but are not guaranteed.
- OS: Linux is expected to work. Windows is supported for the Ollama runtime; Python venv activation commands differ.
- If running CPU‑only, ensure fast SSD swap; avoid paging spikes by lowering `--num-predict` if memory pressure appears.
- Smaller GPUs (≤4 GB VRAM) can still run but may force model quantization or offload; keep expectations modest.
- If you have <20 GB combined memory, choose a smaller or more aggressively quantized model and lower `--num-ctx` (see [Using Different Models](#using-different-models)).
- If you have 30–35+ GB combined memory, you can raise `--num-ctx` (e.g., +25%) or run a larger model (see [Using Different Models](#using-different-models)).

## Ollama runtime installation

This project requires the Ollama runtime. Follow these steps to install it:

- Linux:

```bash
# Download and run the official installer script
curl -fsSL https://ollama.com/install.sh | sh
```

- Windows:

Download and run the Windows installer from the official site: [https://ollama.com](https://ollama.com/)

## Project installation

Ensure the Ollama runtime is installed first (see "Ollama runtime installation").

Prerequisites

- Debian/Ubuntu: install the system venv helper so the installer can create `.venv`:

```bash
sudo apt update && sudo apt install -y python3-venv
```

Automated install (recommended)

The repository includes `scripts/install_deps.py`, a small installer that:

- creates a local virtual environment at `.venv`;
- installs runtime dependencies from `requirements.txt` (and dev deps by default);
- optionally pulls Ollama models (defaults: `cogito:8b` and `embeddinggemma:300m`).

Start the Ollama server in one terminal:

```bash
ollama serve
```

Then run the installer from a second terminal (POSIX):

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

Installer options (see `scripts/install_deps.py`):

- `--runtime-only`: install only `requirements.txt` (skip `requirements-dev.txt`).
- `--no-pull-models`: do not run `ollama pull` for any models.
- `--robot-model` / `--assistant-model` / `--embedding-model`: override which models to pull.
- `--python`: choose a different Python executable to create the venv.

Examples:

```bash
# Install runtime deps only
python3 -m scripts.install_deps --runtime-only

# Install deps but do not pull models
python3 -m scripts.install_deps --no-pull-models

# Pull different models
python3 -m scripts.install_deps --robot-model "llama3:8b" --assistant-model "llama3:8b" --embedding-model "embeddinggemma:300m"
```

Notes about model pulls

- The installer will call `ollama pull` for the main model and the embedding model unless
  you pass `--no-pull-models`. If the `ollama` CLI is not on your `PATH` the script prints a
  warning and skips pulls so the installer still succeeds.

Manual install (alternative)

If you prefer to set up the environment manually:

```bash
# Start Ollama in a separate terminal
ollama serve

# Create and activate venv (POSIX)
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

Windows manual (PowerShell / cmd):

```powershell
git clone https://github.com/PavolUlicny/Local-AI-Agent.git
cd Local-AI-Agent
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
ollama pull cogito:8b
ollama pull embeddinggemma:300m
```

Development dependencies

By default `scripts/install_deps.py` installs dev dependencies as well. To explicitly install
dev tooling after creating the venv:

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Clarifications

- The `ollama` package listed in `requirements.txt` is optional at runtime — the agent talks to
  the Ollama server over its HTTP API; keeping the package installed is harmless and can help with
  some local integrations.
- If you change embedding models, update the `--embedding-model` flag (or the `EMBEDDING_MODEL`
  variable in the Makefile) and `ollama pull` the model so the embedding-based features initialize.

## Quick Start Examples

### 1. Start services and activate the virtualenv

1. Start the Ollama server in one terminal:

   ```bash
   ollama serve
   ```

2. In a second terminal, activate the project environment:

   - **Linux**

     ```bash
     cd Local-AI-Agent
     source .venv/bin/activate
     ```

   - **Windows (PowerShell)**

     ```powershell
     cd Local-AI-Agent
     .\.venv\Scripts\activate
     ```

Keep this terminal open for the commands below.

### 2. Everyday commands

#### Interactive session

- Linux: `python3 -m src.main`
- Windows: `python -m src.main`

#### Ask one question and exit

- Linux: `python3 -m src.main --question "Explain the difference between variance and standard deviation"`
- Windows: `python -m src.main --question "Explain the difference between variance and standard deviation"`

#### Force reasoning without searches

- Linux: `python3 -m src.main --no-auto-search --question "Derive the quadratic formula"`
- Windows: `python -m src.main --no-auto-search --question "Derive the quadratic formula"`

#### Increase search aggressiveness

- Linux: `python3 -m src.main --max-rounds 20 --search-max-results 8`
- Windows: `python -m src.main --max-rounds 20 --search-max-results 8`

## Using the Makefile

These shortcuts mirror the CLI and help standardize local runs. Pick whichever workflow matches your shell.

### If you are not using GNU Make (PowerShell/CMD)

Windows does not ship with GNU Make or a POSIX shell. You can either install it (Git Bash, MSYS2, WSL, or `choco install make`) or simply run the underlying Python commands:

```powershell
# make run
python -m src.main

# make ask
python -m src.main --question "What is 2+2?"

# make run-no-search
python -m src.main --no-auto-search --question "Derive the quadratic formula"

# make run-search
python -m src.main --question "Capital of France?" --max-rounds 1 --search-max-results 2 --ddg-backend duckduckgo --log-level INFO

# make check
python -c "from src.config import AgentConfig; from src.agent import Agent; Agent(AgentConfig(no_auto_search=True, question='healthcheck')); print('OK')"

# make smoke
python -m scripts.smoke
```

Activate the virtual environment beforehand (`.\.venv\Scripts\activate`) just like in the Quick Start and Development sections.

### With GNU Make (Linux or Windows + Git Bash/MSYS2/WSL)

Install `make` if it is not already available. Then run:

```bash
# One-time setup: create venv, install deps, pull default models
# Example: specify both role models if you want custom choices
make dev-setup ROBOT_MODEL=cogito:8b ASSISTANT_MODEL=cogito:8b

# Start Ollama runtime (foreground)
make serve-ollama

# Interactive agent
make run

# One-shot Q&A
make ask QUESTION="What is 2+2?"

# One-shot without web search
make run-no-search QUESTION="Derive the quadratic formula"

# One-shot configured for web search (auto gate may still choose NO_SEARCH)
make run-search QUESTION="Capital of France?" MAX_ROUNDS=1 SEARCH_MAX_RESULTS=2 DDG_BACKEND=duckduckgo LOG_LEVEL=INFO

# Health checks
make check-ollama   # prints local models if Ollama is up
make check          # quick import/instantiate sanity check

# Cleanup caches
make clean

### Install via project installer

You can run the repository installer (`scripts/install_deps.py`) directly via Make:

```bash
# Create `.venv`, install runtime and dev deps, and pull configured Ollama models
make install-deps

# Alias (convenience)
make bootstrap
```

Notes:

- Override `ROBOT_MODEL`/`ASSISTANT_MODEL` in `make dev-setup ROBOT_MODEL=<name:tag> ASSISTANT_MODEL=<name:tag>` to pull different role models.
- `run-search` tunables: `MAX_ROUNDS`, `SEARCH_MAX_RESULTS`, `DDG_BACKEND`, `LOG_LEVEL` (inherits CLI defaults unless overridden: `12`, `5`, `auto`, `WARNING`). The target simply wires these flags through—automatic search gating still decides whether each question actually hits the web.
- Supported DDGS backend names include `duckduckgo`, `bing`, `brave`, `google`, `mojeek`, `wikipedia`, `yahoo`, and `yandex`; use `auto` (default) to fan out across providers.
- `NO_AUTO_SEARCH` is treated as a boolean flag. Only truthy values enable it: `1,true,TRUE,yes,YES,on,ON`. Setting `NO_AUTO_SEARCH=0` will not enable the flag.
- `LOG_FILE` supports paths with spaces (quoted automatically). Example: `make run LOG_FILE="/tmp/agent logs/agent.log"`.
- Disable console logging (so only the assistant answer shows) with `LOG_CONSOLE=0` or directly via `--no-log-console`; add `--log-file` if you want the logs persisted, otherwise they are dropped.
- Embedding knobs are mirrored as environment variables: `EMBEDDING_MODEL`, `EMBEDDING_SIMILARITY_THRESHOLD`, `EMBEDDING_HISTORY_DECAY`, `EMBEDDING_RESULT_SIMILARITY_THRESHOLD`, and `EMBEDDING_QUERY_SIMILARITY_THRESHOLD` feed the same CLI flags (e.g., `make run EMBEDDING_MODEL=embeddinggemma:300m`).

## Development

- Install development packages:

```bash
pip install -r requirements-dev.txt
```

- Pre-commit hooks (format, lint, type-check):

```bash
pre-commit install
pre-commit run --all-files
```

- Ruff lint and MyPy type check (mirrors CI):

```bash
ruff check src tests scripts
mypy --config-file=pyproject.toml src
```

- Smoke test (no network calls):

Linux:

```bash
python3 -m scripts.smoke
```

Windows:

```bash
python -m scripts.smoke
```

- Pytest suite (unit utilities + agent orchestration):

```bash
pytest
```

- Run via module entrypoint (recommended):

Linux:

```bash
python3 -m src.main --question "Hello"
```

Windows:

```bash
python -m src.main --question "Hello"
```

## Continuous Integration (CI)

- GitHub Actions workflow (`.github/workflows/ci.yml`) runs on pushes/PRs to `main`:
  - `pre-commit run --all-files` (includes ruff fix/format and mypy)
  - `ruff check src tests scripts`
  - `mypy src`
  - `python -m scripts.smoke` (no-network smoke test)

Locally, you can emulate this with the commands in the Development section.

## Configuration Guidelines

- Lower `--robot-temp` to strengthen determinism in planning/classification; keep near 0 for reproducibility.
- Raise `--assistant-temp` if answers feel too rigid; lower for more formal precision.
- If encountering repeated context length rebuilds, pre‑adjust `--num-ctx` downward instead of relying on automatic halving.
- Expanding `--max-rounds` increases latency & potential redundancy; consider balancing with stricter `--max-relevance-llm-checks`.

## Performance Considerations

- Each search round = network latency + multiple prompt invocations (relevance + planning + filtering). Tune rounds and search backend for speed.
- Keyword filtering is O(n * k) where n = tokens, k = stopwords; negligible versus LLM inference cost.
- Rebuilds (context halving) reduce quality by shrinking available memory; avoid oversized inputs to sustain higher token budgets.

## Troubleshooting

| Symptom | Cause | Action |
| --------- | ------- | -------- |
| `Model 'xyz' not found` | Ollama model not pulled | `ollama pull xyz` then retry. |
| Frequent context rebuild logs | Oversized conversation or results | Reduce `--max-rounds`, `--max-context-turns`, or initial token settings. |
| Many rate limit warnings | DDGS provider throttling | Lower concurrency (accept defaults) or narrow `--ddg-backend` to a single engine like `duckduckgo`. |
| Empty search results | Provider mix mismatch | Retry with a specific backend (`--ddg-backend duckduckgo`/`bing`) or broaden query phrasing. |
| No new suggestions | Planning chain conservative or truncation | Increase `--max-followup-suggestions` or verify not hitting truncation caps. |
| `ModuleNotFoundError: No module named 'langchain_community'` | Wrong Python interpreter or deps not installed | Activate the venv and reinstall: Linux: `source .venv/bin/activate && pip install -r requirements.txt`. Run via `python3 -m src.main`. Windows: `.\.venv\Scripts\activate && pip install -r requirements.txt`. Run via `python -m src.main`. |
| `pre-commit` alters files locally | Hooks include auto-fixers (ruff). Re-run `git add` after fixes. | Use `pre-commit run --all-files --show-diff-on-failure` in CI modes. |
| `--log-file` path with spaces fails | Makefile quoting added, but direct CLI still needs quotes | Use `--log-file "/path/with spaces/agent.log"`. |

## Security & Safety Notes

- All network calls flow through DDGS text search providers (DuckDuckGo/Bing/Brave); no external code execution beyond HTTP GET for search snippets.
- User input is directly embedded into prompts; avoid placing secrets or credentials in queries.
- URL canonicalization strips default ports and `www.` but retains query string—beware PII embedded in copied URLs.
- The system does not attempt adversarial prompt injection mitigation beyond rigid output regex validation for classifier stages.
- License (MIT) permits broad reuse, modification, and distribution; ensure compliance with third‑party content usage from search results.

Optional hardening tips:

- Run `pip-audit` in CI to surface dependency CVEs:

```bash
pip install pip-audit
pip-audit
```

- Treat search snippets as untrusted content within prompts; avoid following instructions contained in results.

## Known Limitations

- DDGS provider scraping can vary over time; identical queries may yield different snippets or ordering across runs.
- The assistant model uses nonzero temperature by default for answers, so responses are not bit‑for‑bit deterministic.
- No JavaScript execution or full page rendering is performed; only titles, URLs, and snippet text are used, which may miss dynamic content.
- Topic selection does not perform context‑length rebuilds; if it fails, the system proceeds without selecting a topic for that turn.
- No persistent storage of conversations beyond process lifetime; restarting the program resets topics and history.
- Rate limiting by DDGS providers may still occur despite retries and backoff; reducing rounds or switching backend can help.
- The system does not include adversarial prompt‑injection defenses beyond strict classifier output validation.

## Contributing

- Fork, create a feature branch, and open a PR.
- Run `pre-commit run --all-files` before committing to keep diffs clean.
- If you add dependencies, update `requirements.txt` and consider adding a smoke check for new wiring.

## License

This project is released under the **MIT License**. See `LICENSE` for full text.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for prompt/chain abstractions.
- [Ollama](https://ollama.com/) for local model serving.
- [DDGS](https://pypi.org/project/ddgs/) for resilient multi-provider search.

---
For questions, improvements, or integration ideas, feel free to open issues or submit PRs.
