## Overview

Local AI Agent runs entirely on your machine, combining an Ollama‑served model with a controlled, multi‑round search loop. It automatically decides whether a user query needs external factual grounding (SEARCH) or can be answered with general reasoning (NO_SEARCH). When in search mode, it plans, filters, and iteratively refines queries, accumulates concise result snippets, and finally synthesizes an answer that avoids leaking implementation meta‑details.

### Key Features

- Automatic search decision (SEARCH vs NO_SEARCH) via a dedicated prompt classifier.
- Seed query synthesis distinct from the raw user question to broaden retrieval.
- Iterative planning of follow‑up search queries (bounded by `--max-rounds`).
- Dual relevance filters: keyword heuristics and LLM YES/NO validation, plus embedding-based shortcuts.
- Result deduplication (canonicalized URLs + SHA‑256 hash).
- Adaptive truncation, multi-topic memory, embedding-assisted relevance.

### Core Architecture

High‑level components and their roles:

| Component | Role |
| ----------- | ------ |
| Prompts (`prompts.py`) | Structured templates for each decision/action stage. |
| Robot LLM (low temp) | Deterministic classifiers & planners (search decision, relevance, query planning). |
| Assistant LLM (higher temp) | Final natural‑language answer synthesis. |
| Search Client (`search_client.SearchClient`) | DDGS metasearch with retry/backoff and normalized result payloads. |
| Search Orchestrator (`search_orchestrator.SearchOrchestrator`) | Coordinates the pending-query loop, deduplication, planning, and fill cycles. |
| Topic Manager (`topic_manager.TopicManager`) | Maintains per-topic turns, keyword sets, and blended embeddings. |
| Embedding Client (`embedding_client.EmbeddingClient`) | Provides cached, fault-tolerant access to Ollama embeddings. |
| Keyword/Text Utilities (`keywords.py`, `text_utils.py`) | Tokenization, regex validation, truncation, context/time formatting. |
| URL & Topic Utilities (`url_utils.py`, `topic_utils.py`) | URL canonicalization plus cosine similarity, tail turns, and topic briefs. |

### Execution Pipeline

```text
User Input
 │
 ├─► Context Classification (FOLLOW_UP / EXPAND / NEW_TOPIC)
 ├─► Search Decision (SEARCH / NO_SEARCH)
 ├─► Seed Query Generation
 ├─► Iterative Round Loop (≤ max-rounds)
 ├─► Aggregate & Truncate Results
 └─► Final Answer Prompt (with or without search context)
```

### Modules & Responsibilities

- `src/main.py`: Entry-point that wires CLI args to the `Agent` and starts execution.
- `src/agent.py`: Orchestration of context, search, and response synthesis.
- `src/chains.py`: Builds LangChain components and prompt chains.
- `src/search_client.py`: DDGS wrapper with retries and normalization.
- `src/search_orchestrator.py`: Query/pending loop logic and deduplication.
- `src/topic_manager.py`: Topic lifecycle and embedding blending.

Refer to the `src/` module files for implementation details.

### Project Structure

```text
Local-AI-Agent/
├─ LICENSE
├─ Makefile
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ requirements-dev.txt
├─ .pre-commit-config.yaml
├─ .github/
│  └─ workflows/
│     └─ ci.yml
├─ docs/
│  ├─ index.md
│  ├─ install.md
│  ├─ architecture.md
│  ├─ usage.md
│  ├─ search_strategy.md
│  ├─ topic_management.md
│  ├─ development.md
│  ├─ ci.md
│  ├─ security.md
│  ├─ troubleshooting.md
│  └─ contributing.md
├─ scripts/
│  ├─ install_deps.py    # convenience installer for dependencies and models
│  └─ smoke.py           # no-network smoke check used by CI
├─ src/
│  ├─ __init__.py
│  ├─ main.py
│  ├─ cli.py
│  ├─ config.py
│  ├─ agent.py
│  ├─ chains.py
│  ├─ prompts.py
│  ├─ search_client.py
│  ├─ search_orchestrator.py
│  ├─ embedding_client.py
│  ├─ model_utils.py
│  ├─ topic_manager.py
│  ├─ topic_utils.py
│  ├─ text_utils.py
│  ├─ keywords.py
│  ├─ url_utils.py
│  └─ exceptions.py
├─ tests/
│  ├─ conftest.py
│  ├─ agent_test_utils.py
│  ├─ test_agent_clients.py
│  ├─ test_agent_response_modes.py
│  ├─ test_agent_search_behaviors.py
│  ├─ test_agent_search_classifiers.py
│  ├─ test_keywords.py
│  ├─ test_text_utils.py
│  ├─ test_topic_manager.py
│  ├─ test_topic_utils.py
│  └─ test_url_utils.py
└─ .venv/                # (optional local python virtualenv, typically excluded from VCS)

Notes:
- `scripts/install_deps.py` is provided to bootstrap Python dependencies and pull any local model artifacts.
- Run the CLI via `python -m src.main` or the `Makefile` helpers for common workflows.
```
