## CLI Arguments

The agent exposes many CLI flags for tuning behavior. Use the table below for quick reference.

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

See `src/config.py` for full defaults and the `Makefile` wrappers for common combinations.

## Using Different Models

You can substitute any Ollama model by pulling it first and passing role-specific flags. Examples:

```bash
ollama pull llama3:8b
python3 -m src.main --robot-model llama3:8b --assistant-model llama3:8b
```

Adjust `--num-ctx` and `--num-predict` according to the model's capabilities.

## Quick Start Examples

### Interactive session

```bash
# Linux
python3 -m src.main
```

```powershell
# Windows (PowerShell / cmd)
python -m src.main
```

### One-shot (single question)

```bash
# Linux
python3 -m src.main --question "Explain the difference between variance and standard deviation"
```

```powershell
# Windows (PowerShell / cmd)
python -m src.main --question "Explain the difference between variance and standard deviation"
```

### One-shot without search

```bash
# Linux
python3 -m src.main --no-auto-search --question "Derive the quadratic formula"
```

```powershell
# Windows (PowerShell / cmd)
python -m src.main --no-auto-search --question "Derive the quadratic formula"
```

Windows users: the `Makefile` targets are primarily for Unix-like shells; on Windows use the direct `python -m src.main` commands shown above, or run the Makefile under Git Bash / WSL if you prefer `make` helpers.

## Performance Considerations

- Each search round adds latency; tune `--max-rounds` and `--search-max-results` for speed.
- Rebuilds (context halving) reduce available context; reduce `--num-ctx` if you see many rebuilds.

## Configuration Guidelines

- Lower `--robot-temp` to strengthen determinism in planning/classification; keep near 0 for reproducibility.
- Raise `--assistant-temp` if answers feel too rigid; lower for more formal precision.
- If encountering repeated context length rebuilds, pre‑adjust `--num-ctx` downward instead of relying on automatic halving.
- Expanding `--max-rounds` increases latency & potential redundancy; consider balancing with stricter `--max-relevance-llm-checks`.
