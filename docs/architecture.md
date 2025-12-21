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
| Search Orchestrator (`search_orchestrator.SearchOrchestrator`) | Coordinates search rounds, deduplication, planning, and fill cycles. |
| Search Client (`search_client.SearchClient`) | DDGS metasearch with retry/backoff and normalized result payloads. |
| Search Context (`search_context`) | Bundles search state, services, and immutable context. |
| Topic Manager (`topic_manager.TopicManager`) | Maintains per-topic turns, keyword sets, and blended embeddings. |
| Embedding Client (`embedding_client.EmbeddingClient`) | Cached, fault-tolerant access to Ollama embeddings. |
| Keyword/Text Utilities (`keywords.py`, `text_utils.py`) | Keyword extraction, tokenization, truncation, formatting. |
| URL & Topic Utilities (`url_utils.py`, `topic_utils.py`) | URL canonicalization, cosine similarity, tail turns, topic briefs. |

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

**Core Agent:**

- `src/main.py`: Entry point wiring CLI args to the Agent.
- `src/agent.py`: Main agent orchestrator coordinating context, search, and response synthesis.
- `src/agent_utils.py`: Agent helper functions for query processing and context management.
- `src/config.py`: Configuration dataclass with all agent parameters and defaults.
- `src/cli.py`: CLI argument parser and logging configuration.

**Search System:**

- `src/search.py`: High-level search interface and orchestrator builder.
- `src/search_orchestrator.py`: Coordinates search rounds, deduplication, and planning.
- `src/search_planning.py`: Query planning and candidate generation.
- `src/search_processing.py`: Result processing and aggregation.
- `src/search_validation.py`: Result and query filtering logic.
- `src/search_client.py`: DDGS wrapper with retries and normalization.
- `src/search_context.py`: Context objects bundling search state and dependencies.
- `src/search_chain_utils.py`: Shared utilities for search chain operations.

**Context & Topics:**

- `src/context.py`: Query context dataclass for processing pipeline.
- `src/topic_manager.py`: Topic lifecycle and embedding blending.
- `src/topics.py`: Topic data structures and management.
- `src/topic_utils.py`: Topic utilities (similarity, briefs, tail turns).

**LLM Integration:**

- `src/chains.py`: Builds LangChain components and prompt chains.
- `src/prompts.py`: Structured templates for each decision/action stage.
- `src/response.py`: Response synthesis and formatting.
- `src/embedding_client.py`: Cached, fault-tolerant Ollama embeddings.
- `src/ollama.py`: Ollama runtime detection and startup.
- `src/model_utils.py`: Model-related utilities.

**Utilities:**

- `src/keywords.py`: Keyword extraction and filtering.
- `src/text_utils.py`: Text tokenization, truncation, and formatting.
- `src/url_utils.py`: URL canonicalization and normalization.
- `src/input_handler.py`: Interactive prompt/session helpers.
- `src/exceptions.py`: Custom exception types.

Refer to the `src/` module files for implementation details.
