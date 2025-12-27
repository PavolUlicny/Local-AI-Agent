## Overview

Local AI Agent runs entirely on your machine, combining an Ollama‑served model with a controlled, multi‑round search loop. It automatically decides whether a user query needs external factual grounding (SEARCH) or can be answered with general reasoning (NO_SEARCH). When in search mode, it plans, filters, and iteratively refines queries, accumulates concise result snippets, and finally synthesizes an answer that avoids leaking implementation meta‑details.

### Key Features

- Automatic search decision (SEARCH vs NO_SEARCH) via a dedicated prompt classifier.
- Query planning to generate initial and follow-up search queries (bounded by `--max-rounds`).
- Dual relevance filters: keyword heuristics and LLM YES/NO validation, plus embedding-based shortcuts.
- Result deduplication (canonicalized URLs + SHA‑256 hash).
- Adaptive truncation, conversation history management with auto-trim, embedding-assisted result filtering.

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
| Conversation Manager (`conversation.ConversationManager`) | Maintains conversation history with auto-trim to respect character budget. |
| Command Handler (`commands.CommandHandler`) | Processes slash commands like /quit, /clear, /compact, /stats, /help. |
| Embedding Client (`embedding_client.EmbeddingClient`) | Cached, fault-tolerant access to Ollama embeddings for result filtering. |
| Keyword/Text Utilities (`keywords.py`, `text_utils.py`) | Keyword extraction, tokenization, truncation, formatting. |
| URL Utilities (`url_utils.py`) | URL canonicalization and normalization. |

### Execution Pipeline

```text
User Input
 │
 ├─► Command Check (/quit, /clear, /compact, /stats, /help)
 ├─► Search Decision (SEARCH / NO_SEARCH)
 ├─► Query Planning & Validation
 ├─► Iterative Round Loop (≤ max-rounds)
 ├─► Aggregate & Truncate Results
 ├─► Final Answer Prompt (with conversation history + search results)
 └─► Add Turn to Conversation (auto-trim if over budget)
```

### Modules & Responsibilities

**Core Agent:**

- `src/main.py`: Entry point wiring CLI args to the Agent.
- `src/agent.py`: Main agent orchestrator coordinating context, search, and response synthesis.
- `src/agent_utils.py`: Agent helper functions for query processing and context management.
- `src/config.py`: Pydantic configuration model with declarative validation and field constraints.
- `src/cli.py`: CLI argument parser and logging configuration.
- `src/llm_lifecycle.py`: LLM lifecycle management (building, rebuilding with reduced context, restoration).
- `src/constants.py`: Type-safe enums and constant values (chain names, rebuild keys, etc.).

**Search System:**

- `src/search.py`: High-level search interface and orchestrator builder.
- `src/search_orchestrator.py`: Coordinates search rounds, deduplication, and planning.
- `src/search_planning.py`: Query planning and candidate generation.
- `src/search_processing.py`: Synchronous result processing and aggregation.
- `src/search_processing_async.py`: Asynchronous result processing for parallel operations.
- `src/search_processing_common.py`: Common processing utilities shared between sync and async.
- `src/search_parallel.py`: Parallel query execution coordination.
- `src/search_validation.py`: Result and query filtering logic.
- `src/search_client.py`: Synchronous DDGS wrapper with retries and normalization.
- `src/search_client_async.py`: Asynchronous DDGS wrapper for parallel searches.
- `src/search_context.py`: Context objects bundling search state and dependencies.
- `src/search_chain_utils.py`: Chain invocation utilities with retry logic.
- `src/search_retry_utils.py`: Shared retry logic, backoff, and exception handling for search clients.

**Conversation & Commands:**

- `src/conversation.py`: Conversation history management with auto-trim and statistics.
- `src/commands.py`: Slash command handling (/quit, /clear, /compact, /stats, /help).

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
- `src/url_utils.py`: URL canonicalization, normalization, and SSRF protection.
- `src/input_handler.py`: Interactive prompt/session helpers.
- `src/exceptions.py`: Custom exception hierarchy (ConfigurationError, InputValidationError, SearchError, etc.).
- `src/protocols.py`: Type protocols for dependency injection and interface contracts.

Refer to the `src/` module files for implementation details.
