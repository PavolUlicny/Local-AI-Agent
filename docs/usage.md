## CLI Arguments

The agent exposes many CLI flags for tuning behavior. Common flags include:

- `--no-auto-search` (`--nas`) — Force NO_SEARCH unless changed manually.
- `--force-search` (`--fs`) — Always perform SEARCH.
- `--max-rounds` (`--mr`) — Upper bound on search query rounds (seed + planned).
- `--search-max-results` (`--smr`) — Result fetch count per query.
- `--assistant-num-ctx` / `--robot-num-ctx` — Context windows for assistant/robot.
- `--assistant-num-predict` / `--robot-num-predict` — Generation caps.
- `--robot-model` / `--assistant-model` — Ollama model names for each role.

See `src/config.py` for full defaults and the `Makefile` wrappers for common combinations.

## Using Different Models

You can substitute any Ollama model by pulling it first and passing role-specific flags. Examples:

```bash
ollama pull llama3:8b
python3 -m src.main --robot-model llama3:8b --assistant-model llama3:8b
```

Adjust `--num-ctx` and `--num-predict` according to the model's capabilities.

## Quick Start Examples

Interactive session:

```bash
# Linux
python3 -m src.main

# One-shot
python3 -m src.main --question "Explain the difference between variance and standard deviation"

# One-shot without search
python3 -m src.main --no-auto-search --question "Derive the quadratic formula"
```

## Performance Considerations

- Each search round adds latency; tune `--max-rounds` and `--search-max-results` for speed.
- Rebuilds (context halving) reduce available context; reduce `--num-ctx` if you see many rebuilds.
