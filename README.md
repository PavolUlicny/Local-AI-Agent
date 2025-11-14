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
10. [Installation](#installation)
11. [Quick Start Examples](#quick-start-examples)
12. [Configuration Guidelines](#configuration-guidelines)
13. [Performance Considerations](#performance-considerations)
14. [Troubleshooting](#troubleshooting)
15. [Limitations & Future Work](#limitations--future-work)
16. [Security & Safety Notes](#security--safety-notes)
17. [License](#license)

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
 │       └─► Optional Fill Attempts until capacity reached
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
- Perform fill cycles until either round capacity or attempt limits reached.

5. Truncate aggregated corpus to `MAX_SEARCH_RESULTS_CHARS` before answer synthesis.

## Robustness & Error Handling

- Graceful handling of missing Ollama model (`ollama pull <model>` hint logged).
- Detects context length / token window errors (string pattern matching). On detection, halves `num_ctx` and recomputes `num_predict` (bounded ≥ 2048 / ≥ 512) then rebuilds chains; capped by `MAX_REBUILD_RETRIES` per stage.
- Retries / exponential backoff for DuckDuckGo (rate limit or transient failures) with jitter.
- Fallback paths: failing seed→use original query; failing planning→skip suggestions; failing relevance→drop result.
- All LLM classifier outputs normalized & regex validated; defaults chosen to safe expansive path (e.g., default SEARCH if classification malformed).
- If a classifier call itself errors (e.g., connection/model error), the agent falls back to NO_SEARCH for that turn; if the classifier returns malformed output, it defaults to SEARCH.
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
| `--num-ctx` | `8192` | Initial context window tokens. |
| `--num-predict` | `8192` | Initial generation cap tokens. |
| `--robot-temp` | `0.0` | Temperature for classifier/planner chains. |
| `--assistant-temp` | `0.7` | Temperature for final answer chain. |
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

## Installation

Requires a working [Ollama](https://ollama.com/) runtime.

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
python -m src.main
```

Ask a single question and exit:

```bash
python -m src.main --question "Explain the difference between variance and standard deviation"
```

Force reasoning without searches:

```bash
python -m src.main --no-auto-search --question "Derive the quadratic formula"
```

Increase search aggressiveness:

```bash
python -m src.main --max-rounds 20 --search-max-results 8
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

## Limitations & Future Work

- No persistent disk caching of search results between runs.
- No structured citation graph; answers consume raw text without provenance markers.
- DuckDuckGo HTML scraping may be brittle against layout changes.
- Relevance heuristics simple (keyword intersection + binary classifier); could upgrade to embedding similarity.
- Topics stored in memory only; restart resets context.
- No parallelization of result filtering (sequential chain calls).
- No explicit hallucination detection beyond constraints in prompts.

Potential enhancements:

- Persistent vector store & hybrid semantic retrieval.
- Pluggable search providers (e.g., Brave, local corpora, offline docs).
- Structured citation block appended post‑answer with link ranking.
- Embedding‑based cluster dedup & diversity scoring.
- Configurable persistence layer (SQLite or LiteFS) for long‑term topic memory.

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
