# Local AI Agent

> An extensible, self-steering local LLM assistant that can optionally enrich answers with iterative, relevance‑filtered DuckDuckGo web searches while tracking multi‑topic conversational context.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Core Architecture](#core-architecture)
4. [Execution Pipeline](#execution-pipeline)
5. [Modules & Responsibilities](#modules--responsibilities)
6. [Topic & Context Management](#topic--context-management)
7. [Search Strategy & Filtering](#search-strategy--filtering)
8. [Robustness & Error Handling](#robustness--error-handling)
9. [CLI Arguments](#cli-arguments)
10. [Using Different Models](#using-different-models)
11. [System Requirements](#system-requirements)
12. [Ollama runtime installation](#ollama-runtime-installation)
13. [Project installation](#project-installation)
14. [Quick Start Examples](#quick-start-examples)
15. [Configuration Guidelines](#configuration-guidelines)
16. [Performance Considerations](#performance-considerations)
17. [Troubleshooting](#troubleshooting)
18. [Security & Safety Notes](#security--safety-notes)
19. [License](#license)

---

## Overview

Local AI Agent runs entirely on your machine, combining an Ollama‑served model with a controlled, multi‑round search loop. It automatically decides whether a user query needs external factual grounding (SEARCH) or can be answered with general reasoning (NO_SEARCH). When in search mode, it plans, filters, and iteratively refines queries, accumulates concise result snippets, and finally synthesizes an answer that avoids leaking implementation meta‑details.

Design goals:

- Deterministic, inspectable prompt chains (LangChain).
- Conservative context growth: pruning keywords & past turns.
- Resilience to context window overruns (automatic model parameter downscaling & chain rebuild).
- Separation of roles: a low‑temperature "robot" model for planning/decisions; a higher‑temperature assistant for final answer generation.

## Key Features

- Automatic search decision (SEARCH vs NO_SEARCH) via a dedicated prompt classifier.
- Seed query synthesis distinct from the raw user question to broaden retrieval.
- Iterative planning of follow‑up search queries (bounded by `--max-rounds`).
- Dual relevance filters:
  - Lightweight keyword intersection check.
  - LLM YES/NO validation for borderline cases (capped by `--max-relevance-llm-checks`).
- Query gating (LLM YES/NO) to avoid off‑topic expansion.
- Result deduplication (canonicalized URLs + SHA‑1 hash of assembled title/URL/snippet).
- Adaptive truncation of large text sections (conversation, prior answers, search corpus) with sensible cut heuristics.
- Multi‑topic memory with keyword pruning and turn window constraints.
- Automatic recovery from context length errors (progressive halving of `num_ctx` and `num_predict`).
- One‑shot mode via `--question` (non‑interactive).

## Core Architecture

High‑level components and their roles:

| Component | Role |
|-----------|------|
| Prompts (`prompts.py`) | Structured templates for each decision/action stage. |
| Robot LLM (low temp) | Deterministic classifiers & planners (search decision, relevance, query planning). |
| Assistant LLM (higher temp) | Final natural‑language answer synthesis. |
| Search Wrapper | DuckDuckGo results fetch with backoff & retries. |
| Topic Manager (`helpers.Topic`) | Maintains per-topic turns & evolving keyword set. |
| Keyword Utilities | Tokenization, stopword filtering, heuristic numeric filtering. |
| Rebuild Logic | Detects context length errors and rebuilds prompt chains with reduced parameters. |

Note: Robot and Assistant both use the same base model specified via `--model`, but with different temperatures and decoding parameters tailored to their roles.

## Execution Pipeline

```text
User Input
 │
 ├─► Context Classification (FOLLOW_UP / EXPAND / NEW_TOPIC)
 │       Select / create topic; gather recent turns & prior answers
 │
 ├─► Search Decision (SEARCH / NO_SEARCH)
 │       If NO_SEARCH → Answer directly
 │       If SEARCH → proceed
 │
 ├─► Seed Query Generation
 │
 ├─► Iterative Round Loop (≤ max-rounds)
 │       ├─► Execute Search (DuckDuckGo, N results)
 │       ├─► Result Relevance Filter (keyword heuristic + LLM YES/NO)
 │       ├─► Keyword Expansion & Dedup
 │       ├─► Planning Prompt (propose new queries)
 │       ├─► Query Filter Prompt (gate each candidate YES/NO)
 │       ├─► Optional Fill Attempts (extra planning cycles) to fill remaining query slots
 │       └─► Round Accounting: A round counts only if ≥1 result for that query is accepted; queries yielding 0 accepted results are dropped without consuming the round budget.
 │
 ├─► Aggregate & Truncate Results
 │
 └─► Final Answer Prompt (with or without search context)
```

## Modules & Responsibilities

### `src/main.py`

Entry point, CLI parsing, prompt chain construction (`PromptTemplate | LLM | StrOutputParser`), search loop orchestration, resilience/rebuild strategy, console streaming of final answer, interactive shell. Maintains global counters to cap rebuild attempts per stage.

### `src/prompts.py`

Resolves `PromptTemplate` for a range of LangChain versions (searches multiple import paths), then declares immutable templates governing output constraints and permissible phrasing. Each template enforces strict, minimal surface (e.g., plain YES/NO tokens) to simplify downstream validation.

### `src/helpers.py`

Utility layer: tokenization, stopword & numeric heuristic filtering, context/date formatting, URL canonicalization, keyword extraction, truncation rules, topic selection logic, keyword pruning frequency pass, regex validation for constrained outputs. Houses constants controlling size budgets and caps.

## Topic & Context Management

- Each topic retains a list of `(user, assistant)` turns.
- On each new query, `_select_topic` attempts to classify relationship (FOLLOW_UP / EXPAND / NEW_TOPIC) via a dedicated chain. If no sufficiently overlapping keywords, a new topic begins.
- Topics exceeding `MAX_TOPICS` are pruned FIFO.
- Turn history per topic is trimmed to `max_context_turns * 2` preserving a sliding window.
- Keywords aggregated from user query, assistant answer, and truncated search result text; pruned to `MAX_TOPIC_KEYWORDS` by frequency ordering.

## Search Strategy & Filtering

1. Decide necessity (SEARCH vs NO_SEARCH).
2. Generate a seed query (fallback to original if generation fails).
3. Maintain a queue of pending queries (seed + planned follow‑ups) up to `--max-rounds`.
4. For each query:

- Fetch `--search-max-results` results (with retry/backoff).
- Deduplicate by URL (canonicalized) and SHA‑1 hash of assembled title/URL/snippet.
- Apply fast keyword intersection relevance; escalate borderline cases to LLM (YES/NO) within `--max-relevance-llm-checks` budget.
- Expand topic keyword set from accepted results.
- Plan new queries; gate each with query filter classifier.
- Perform fill cycles (additional planning passes) until either round capacity or fill attempt limits reached.
- Only queries that yield ≥1 accepted result consume a round; queries with zero accepted results are discarded without decrementing remaining rounds.

5. Truncate aggregated corpus to `MAX_SEARCH_RESULTS_CHARS` before answer synthesis.

## Robustness & Error Handling

- Graceful handling of missing Ollama model (`ollama pull <model>` hint logged).
- Detects context length / token window errors (string pattern matching). On detection, halves `num_ctx` and recomputes `num_predict` (bounded ≥ 2048 / ≥ 512) then rebuilds chains; capped by `MAX_REBUILD_RETRIES` per stage.
- Automatic recovery from context length errors (progressively halves `num_ctx`; `num_predict` is reduced to ≤ half of the new context, never below 512, rather than always halving).
- Retries / exponential backoff for DuckDuckGo (rate limit or transient failures) with jitter.
- Fallback paths: failing seed → use original query; failing planning → skip suggestions; failing relevance → drop result.
- All LLM classifier outputs are normalized and regex-validated. On malformed output, defaults are stage-specific:
  - Search decision → defaults to SEARCH (expansive)
  - Query filter → defaults to NO (conservative)
  - Result filter → defaults to NO (conservative)
- If a classifier call errors (e.g., connection/model error), stage defaults apply:
  - Search decision errors → fall back to NO_SEARCH for that turn
  - Query filter/result filter errors → treat as NO (skip)
  - Clarification: A "malformed output" is text that fails regex validation; this differs from an execution error/exception. Malformed → default token; error/exception → fallback behaviors above.
- Context-length rebuilds are implemented for: search decision, seed generation, result relevance checks, planning, query filter, and final answer stages. Topic selection (context classification) does not rebuild on context errors; it proceeds without selecting a topic when needed.

## CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `cogito:8b` | Ollama model name/tag. |
| `--no-auto-search` | off | Force NO_SEARCH unless logic changed manually. |
| `--max-rounds` | `12` | Upper bound on search query rounds (seed + planned). |
| `--max-context-turns` | `8` | Turns retained from a topic for context injection. |
| `--max-followup-suggestions` | `6` | Max query suggestions per planning cycle. |
| `--max-fill-attempts` | `3` | Extra planning passes to fill remaining slots. |
| `--max-relevance-llm-checks` | `2` | LLM relevance validations for borderline results per query. |
| `--num-ctx` | `12288` | Initial context window tokens. |
| `--num-predict` | `8192` | Initial generation cap tokens. |
| `--robot-temp` | `0.0` | Temperature for classifier/planner chains. |
| `--assistant-temp` | `0.6` | Temperature for final answer chain. |
| `--robot-top-p` | `0.4` | Top‑p for robot model. |
| `--assistant-top-p` | `0.8` | Top‑p for assistant model. |
| `--robot-top-k` | `20` | Top‑k for robot model. |
| `--assistant-top-k` | `80` | Top‑k for assistant model. |
| `--robot-repeat-penalty` | `1.1` | Repeat penalty robot chain. |
| `--assistant-repeat-penalty` | `1.2` | Repeat penalty assistant chain. |
| `--ddg-region` | `us-en` | DuckDuckGo regional variant. |
| `--ddg-safesearch` | `moderate` | SafeSearch level. |
| `--ddg-backend` | `html` | Backend mode (html / lite / api). |
| `--search-max-results` | `5` | Result fetch count per query. |
| `--search-retries` | `4` | Max retries for failed search. |
| `--log-level` | `WARNING` | Logging verbosity. |
| `--log-file` | `None` | Optional file log path. |
| `--question` | `None` | One‑shot non‑interactive question mode. |

## Using Different Models

You can substitute any Ollama model for `cogito:8b` by pulling it first and adjusting runtime parameters:

```bash
ollama pull llama3:8b   # example
python3 -m src.main --model llama3:8b
```

Key adjustments when switching models:

- Context window (`--num-ctx`): Must not exceed the model's maximum (see the [Ollama model catalog](https://ollama.com/library) for limits). Common defaults:
  - Llama 3 (8B/7B): 8192
  - Mistral 7B: 8192
  - Phi / small instruct models: 4096–8192
  - Some extended variants (e.g., 32K finetunes) allow larger windows; set `--num-ctx` accordingly.
- Predict length (`--num-predict`): Keep this comfortably below available combined memory. Large values increase RAM/VRAM footprint linearly with active layers.
- Memory tradeoff: If you hit automatic halving rebuilds early, lower `--num-ctx` first; only then reduce `--num-predict`.
- Quantization: Using a more heavily quantized model (e.g., Q4_K_M) reduces memory, allowing higher `--num-ctx` without rebuilds.
- Robot vs Assistant: Temperatures (`--robot-temp`, `--assistant-temp`) and top‑p/k are independent of model choice; adjust only for style. Some models need slightly lower `--assistant-top-p` to avoid repetition.
- Smaller models (<4B) may benefit from reducing `--num-predict` (e.g., 2048–4096) to preserve responsiveness.
- You may also need to adjust prompts for some models.

Recommended tuning workflow:

1. Start with model's documented max context (e.g., 8192).
1. Run a few multi‑round searches; if rebuild logs appear, drop `--num-ctx` by ~25%.
1. If still rebuilding, cap `--num-predict` at half of current (`8192 → 4096`).
1. Raise again incrementally once stable.

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
- If running CPU‑only, ensure fast SSD swap; avoid paging spikes by lowering `--num-predict` if memory pressure appears.
- Smaller GPUs (≤4 GB VRAM) can still run but may force model quantization or offload; keep expectations modest.
- If you have <20 GB combined memory, choose a smaller or more aggressively quantized model and lower `--num-ctx` (see [Using Different Models](#using-different-models)).
- If you have 30–35+ GB combined memory, you can raise `--num-ctx` (e.g., +25%) or run a larger model (see [Using Different Models](#using-different-models)).

## Ollama runtime installation

This project requires the Ollama runtime. Follow these steps to install it:

- Linux (example using the official install script):

```bash
# Download and run the official installer script
curl -fsSL https://ollama.com/install.sh | sh
```

- Windows:

Download and run the Windows installer from the official site: [https://ollama.com](https://ollama.com/)

## Project installation

Ensure you have downloaded the Ollama runtime using the [tutorial above](#ollama-runtime-installation) before installing this project.

- Linux prerequisite (Debian/Ubuntu): install `python3-venv` so you can create virtual environments:

```bash
sudo apt update && sudo apt install -y python3-venv
```

- Project install:

In one terminal:

```bash
ollama serve
```

In another terminal:

```bash
git clone https://github.com/PavolUlicny/Local-AI-Agent.git
cd Local-AI-Agent
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
ollama pull cogito:8b   # or your chosen model
```

Notes:

- The Python package `ollama` in `requirements.txt` is only used for optional exception typing. LangChain communicates with the Ollama server directly; the pip package is not strictly required at runtime if the server is available (keeping it installed is harmless).

## Quick Start Examples

Interactive session:

```bash
python3 -m src.main
```

Ask a single question and exit:

```bash
python3 -m src.main --question "Explain the difference between variance and standard deviation"
```

Force reasoning without searches:

```bash
python3 -m src.main --no-auto-search --question "Derive the quadratic formula"
```

Increase search aggressiveness:

```bash
python3 -m src.main --max-rounds 20 --search-max-results 8
```

## Configuration Guidelines

- Lower `--robot-temp` to strengthen determinism in planning/classification; keep near 0 for reproducibility.
- Raise `--assistant-temp` if answers feel too rigid; lower for more formal precision.
- If encountering repeated context length rebuilds, pre‑adjust `--num-ctx` downward instead of relying on automatic halving.
- Expanding `--max-rounds` increases latency & potential redundancy; consider balancing with stricter `--max-relevance-llm-checks`.

## Performance Considerations

- Each search round = network latency + multiple prompt invocations (relevance + planning + filtering). Tune rounds and safesearch backend for speed.
- Keyword filtering is O(n * k) where n = tokens, k = stopwords; negligible versus LLM inference cost.
- Rebuilds (context halving) reduce quality by shrinking available memory; avoid oversized inputs to sustain higher token budgets.

## Troubleshooting

| Symptom | Cause | Action |
|---------|-------|--------|
| `Model 'xyz' not found` | Ollama model not pulled | `ollama pull xyz` then retry. |
| Frequent context rebuild logs | Oversized conversation or results | Reduce `--max-rounds`, `--max-context-turns`, or initial token settings. |
| Many rate limit warnings | DuckDuckGo throttling | Lower concurrency (accept defaults) or switch backend (`lite`). |
| Empty search results | Backend HTML scrape variability | Retry with `--ddg-backend api` or broaden query phrasing. |
| No new suggestions | Planning chain conservative or truncation | Increase `--max-followup-suggestions` or verify not hitting truncation caps. |

## Security & Safety Notes

- All network calls are outbound DuckDuckGo searches; no external code execution.
- User input is directly embedded into prompts; avoid placing secrets or credentials in queries.
- URL canonicalization strips default ports and `www.` but retains query string—beware PII embedded in copied URLs.
- The system does not attempt adversarial prompt injection mitigation beyond rigid output regex validation for classifier stages.
- License (Unlicense) permits unrestricted modification and distribution; ensure compliance with third‑party content usage from search results.

## License

This project is released under the **Unlicense** (public domain dedication). See `LICENSE` for full text.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for prompt/chain abstractions.
- [Ollama](https://ollama.com/) for local model serving.
- DuckDuckGo search wrapper from `langchain_community`.

---
For questions, improvements, or integration ideas, feel free to open issues or submit PRs.
