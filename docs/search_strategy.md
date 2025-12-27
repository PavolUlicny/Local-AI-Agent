## Search Strategy & Filtering

1. Decide necessity (SEARCH vs NO_SEARCH) via a classifier chain.
2. Generate initial queries using the planning prompt.
3. Maintain a queue of pending queries up to `--max-rounds`.
4. For each query:

- Fetch `--search-max-results` results (with retry/backoff).
- Deduplicate by URL (canonicalized) and SHA‑256 hash of title/URL/snippet.
- Apply fast keyword intersection relevance; escalate borderline cases to LLM (YES/NO).
- Expand topic keyword set from accepted results and plan new queries.

Only queries that yield ≥1 accepted result consume a round; queries with zero accepted results are discarded without decrementing remaining rounds.

## Robustness & Error Handling

- Handles missing Ollama model with helpful hints.
- Detects context length errors and progressively halves `num_ctx` with capped rebuild attempts.
- Retries for DDGS providers with exponential backoff and jitter.
- Defaults for classifiers are conservative (e.g., query/result filter default to NO).

## Deduplication & Truncation

- URLs are canonicalized (see `src/url_utils.py`) and results are hashed for dedupe.
- Aggregated result corpus is truncated to `MAX_SEARCH_RESULTS_CHARS` before answer synthesis.
