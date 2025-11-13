# Local AI Agent (Ollama + Web Search)

An entirely local AI research assistant that runs on your machine using Ollama for LLM inference and DuckDuckGo for web search. It plans searches, filters results, tracks conversation topics, and writes clear, well-structured answers based only on what it has collected.

This repo is designed for privacy-first workflows: prompts and reasoning stay local; only search queries go out to DuckDuckGo.

## Features

- **Local inference:** Uses Ollama models via `langchain-ollama`.
- **Web search:** Fetches fresh information with DuckDuckGo and filters relevant results.
- **Context awareness:** Tracks topics, follow‑ups, and expansions using lightweight keywording.
- **Evidence-based answers:** Composes answers from collected snippets and earlier turns; avoids unsupported claims.
- **Noise control:** Query and result filters reduce irrelevant searches and spammy content.
- **Configurable behavior:** Tune model, temperatures, context limits, and search rounds in one place.

## How It Works

- **Model interface:** `OllamaLLM` streams responses from a locally hosted Ollama model (default: `cogito:8b`, configurable).
- **Search pipeline:**
  - Seeds a query from your prompt and prior context.
  - Plans additional queries if needed (up to a set limit).
  - Filters candidate queries (YES/NO) based on conversation relevance.
  - Fetches results via DuckDuckGo and filters for usefulness.
- **Answering:** A structured prompt integrates conversation history, prior responses, and search evidence to write a detailed, factual answer without meta-commentary.
- **Topic memory:** Extracts keywords from each turn; classifies new questions as FOLLOW_UP, EXPAND, or NEW_TOPIC to decide how much context to carry forward.

## Requirements

- **OS:** Linux, macOS, or Windows (WSL on Windows recommended).
- **Python:** 3.10+
- **Ollama:** Installed and running locally.

## Installation

1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
# Or follow official instructions for your platform: https://ollama.com/download
```

2. Pull a model

Recommended (used by this project):

```bash
ollama pull cogito:8b
```

Alternatives:

```bash
ollama pull llama3.1:8b
ollama pull mistral:7b

Note: If `cogito:8b` isn't available on your system, pick one of the alternatives (for example, `llama3.1:8b`) and update `used_model` in `src/localAiAgent.py` accordingly.
```

3. Create and activate a Python env

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

4. Install Python dependencies

Primary (recommended):

```bash
pip install -r requirements.txt
```

Alternative (manual install):

```bash
pip install -U \
  langchain \
  langchain-core \
  langchain-community \
  langchain-ollama \
  duckduckgo-search \
  ollama
```

Notes:

- The code dynamically resolves `PromptTemplate` for compatibility across recent LangChain versions; installing both `langchain` and `langchain-core` covers common setups.
- `duckduckgo-search` is required by `langchain_community` search utilities.
- The `ollama` Python package provides error types and utilities (separate from the Ollama runtime binary).

## Quick Start

Start the Ollama server (in a separate terminal):

```bash
ollama serve
```

Note: Ollama may auto-start on demand when first invoked; starting it manually is also fine.

Run the agent from the project root:

```bash
python src/localAiAgent.py
```

Then type your question at the prompt. Use `exit` or `quit` to leave.

Optional: verify your model responds before running the agent:

```bash
ollama run <your-model> -p "hello"
# e.g.
ollama run llama3.1:8b -p "hello"
```

## Configuration

Most knobs live in `main()` inside `src/localAiAgent.py`:

- **Model:** `used_model` (default `cogito:8b`). Set to any installed Ollama model (e.g., `llama3.1:8b`).
- **Generation:** `num_predict_`, `num_ctx_`, temperatures, `top_p`, `top_k`, repeat penalties — separately for “robot” (planning/filtering) and “assistant” (final answers).
- **Search:** `max_search_rounds`, `max_followup_suggestions` — controls how many queries it tries.
- **Context:** `max_context_turns` — how many recent turns to keep per topic.
- **Limits:** `MAX_CONVERSATION_CHARS`, `MAX_PRIOR_RESPONSE_CHARS`, `MAX_SEARCH_RESULTS_CHARS`, `MAX_TURN_KEYWORD_SOURCE_CHARS` — truncation budgets for inputs.

To change these, edit `src/localAiAgent.py` and re-run. If you prefer environment variables, you can add a small parsing layer or open an issue/PR.

Important: The defaults `num_ctx_ = 16384` and `num_predict_ = 16384` are intentionally high and may exceed some models' limits. If you encounter context/window or token-limit errors, reduce them (for example to `8192`/`4096`) to match your model's capabilities and available memory.

## Usage Tips

- Start with a clear question. If you need more depth, ask follow‑ups — the agent classifies them and carries context.
- If results seem stale, try specifying a time window or add a proper noun to the query.
- Switch models depending on hardware and latency: smaller models are faster; larger ones can be more complete.

## Troubleshooting

- **ImportError: langchain-ollama is required**
  - Install: `pip install -U langchain-ollama`

- **Ollama not found / not running**
  - Ensure the binary is installed and the service is active. Start it if needed: `ollama serve`. You can verify with `ollama list` or run a quick test: `ollama run llama3.1:8b`.

- **Model not found**
  - Pull it first: `ollama pull cogito:8b` (or change `used_model` to an installed one).
  - If `cogito:8b` isn't available, use an alternative like `llama3.1:8b` and update `used_model`.

- **Slow or truncated answers**
  - Increase `num_ctx_` and `num_predict_` if your model supports it and you have RAM/VRAM headroom.
  - Reduce `max_search_rounds` if web I/O is the bottleneck.

- **Context/window or token-limit errors**
  - Decrease `num_ctx_` and `num_predict_` to values supported by your model (common values are 4096–8192 for context).

- **Search errors or empty results**
  - Network issues or rate limits can affect DuckDuckGo. Try again, rephrase, or reduce planning depth.

## Privacy

- Inference and reasoning run locally via Ollama.
- Only outbound requests are DuckDuckGo search API queries via the community wrapper (this tool does not fetch full web pages).
- No analytics or telemetry are built into this tool.

## Project Structure

- `src/localAiAgent.py`: Main entry point with prompts, search planning, filtering, and answer synthesis.
- `LICENSE`: Unlicense — public domain dedication.

## Roadmap Ideas

- Add `.env`/CLI configuration and profiles.
- Pluggable search backends (e.g., Brave, Google CSE, local docs).
- Caching layer for search results.
- Optional RAG with local document stores.
- Unit tests for prompt flows and filters.

## License

This project is released under the Unlicense (public domain). See `LICENSE`.

## Acknowledgments

- [Ollama](https://ollama.com/) for local model hosting.
- [LangChain](https://python.langchain.com/) and community integrations.
- [duckduckgo-search](https://pypi.org/project/duckduckgo-search/) for web search access.
