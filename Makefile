SHELL := /bin/bash

# Configuration
MODEL               ?= cogito:8b
PY                  := .venv/bin/python
PIP                 := .venv/bin/pip
CURL                := curl -fsS
# Make variables aligned with CLI flag names (hyphens → underscores). Short aliases noted for quick lookup.
QUESTION            ?= # --question / --q
MAX_ROUNDS          ?= # --max-rounds / --mr
MAX_CONTEXT_TURNS   ?= # --max-context-turns / --mct
MAX_FOLLOWUP_SUGGESTIONS ?= # --max-followup-suggestions / --mfs
MAX_FILL_ATTEMPTS   ?= # --max-fill-attempts / --mfa
MAX_RELEVANCE_LLM_CHECKS ?= # --max-relevance-llm-checks / --mrlc
NUM_CTX             ?= # --num-ctx / --nc
NUM_PREDICT         ?= # --num-predict / --np
ROBOT_TEMP          ?= # --robot-temp / --rt
ASSISTANT_TEMP      ?= # --assistant-temp / --at
ROBOT_TOP_P         ?= # --robot-top-p / --rtp
ASSISTANT_TOP_P     ?= # --assistant-top-p / --atp
ROBOT_TOP_K         ?= # --robot-top-k / --rtk
ASSISTANT_TOP_K     ?= # --assistant-top-k / --atk
ROBOT_REPEAT_PENALTY ?= # --robot-repeat-penalty / --rrp
ASSISTANT_REPEAT_PENALTY ?= # --assistant-repeat-penalty / --arp
DDG_REGION          ?= # --ddg-region / --dr
DDG_SAFESEARCH      ?= # --ddg-safesearch / --dss
DDG_BACKEND         ?= # --ddg-backend / --db
SEARCH_MAX_RESULTS  ?= # --search-max-results / --smr
SEARCH_RETRIES      ?= # --search-retries / --sr
LOG_LEVEL           ?= # --log-level / --ll
LOG_FILE            ?= # --log-file / --lf
LOG_CONSOLE         ?= # --log-console / --lc
NO_AUTO_SEARCH      ?= # --no-auto-search / --nas
EMBEDDING_MODEL     ?= # --embedding-model / --em
EMBEDDING_SIMILARITY_THRESHOLD ?= # --embedding-similarity-threshold / --est
EMBEDDING_HISTORY_DECAY ?= # --embedding-history-decay / --ehd
EMBEDDING_RESULT_SIMILARITY_THRESHOLD ?= # --embedding-result-similarity-threshold / --erst
EMBEDDING_QUERY_SIMILARITY_THRESHOLD ?= # --embedding-query-similarity-threshold / --eqst

# Compose optional CLI args from provided Make variables
EXTRA_ARGS :=
EXTRA_ARGS += $(if $(MAX_ROUNDS), --max-rounds $(MAX_ROUNDS))
EXTRA_ARGS += $(if $(MAX_CONTEXT_TURNS), --max-context-turns $(MAX_CONTEXT_TURNS))
EXTRA_ARGS += $(if $(MAX_FOLLOWUP_SUGGESTIONS), --max-followup-suggestions $(MAX_FOLLOWUP_SUGGESTIONS))
EXTRA_ARGS += $(if $(MAX_FILL_ATTEMPTS), --max-fill-attempts $(MAX_FILL_ATTEMPTS))
EXTRA_ARGS += $(if $(MAX_RELEVANCE_LLM_CHECKS), --max-relevance-llm-checks $(MAX_RELEVANCE_LLM_CHECKS))
EXTRA_ARGS += $(if $(NUM_CTX), --num-ctx $(NUM_CTX))
EXTRA_ARGS += $(if $(NUM_PREDICT), --num-predict $(NUM_PREDICT))
EXTRA_ARGS += $(if $(ROBOT_TEMP), --robot-temp $(ROBOT_TEMP))
EXTRA_ARGS += $(if $(ASSISTANT_TEMP), --assistant-temp $(ASSISTANT_TEMP))
EXTRA_ARGS += $(if $(ROBOT_TOP_P), --robot-top-p $(ROBOT_TOP_P))
EXTRA_ARGS += $(if $(ASSISTANT_TOP_P), --assistant-top-p $(ASSISTANT_TOP_P))
EXTRA_ARGS += $(if $(ROBOT_TOP_K), --robot-top-k $(ROBOT_TOP_K))
EXTRA_ARGS += $(if $(ASSISTANT_TOP_K), --assistant-top-k $(ASSISTANT_TOP_K))
EXTRA_ARGS += $(if $(ROBOT_REPEAT_PENALTY), --robot-repeat-penalty $(ROBOT_REPEAT_PENALTY))
EXTRA_ARGS += $(if $(ASSISTANT_REPEAT_PENALTY), --assistant-repeat-penalty $(ASSISTANT_REPEAT_PENALTY))
EXTRA_ARGS += $(if $(DDG_REGION), --ddg-region $(DDG_REGION))
EXTRA_ARGS += $(if $(DDG_SAFESEARCH), --ddg-safesearch $(DDG_SAFESEARCH))
EXTRA_ARGS += $(if $(DDG_BACKEND), --ddg-backend $(DDG_BACKEND))
EXTRA_ARGS += $(if $(SEARCH_MAX_RESULTS), --search-max-results $(SEARCH_MAX_RESULTS))
EXTRA_ARGS += $(if $(SEARCH_RETRIES), --search-retries $(SEARCH_RETRIES))
EXTRA_ARGS += $(if $(LOG_LEVEL), --log-level $(LOG_LEVEL))
EXTRA_ARGS += $(if $(LOG_FILE), --log-file "$(LOG_FILE)")
EXTRA_ARGS += $(if $(filter 0 false FALSE no NO off OFF,$(LOG_CONSOLE)), --no-log-console)
# Treat only common truthy values as enabling the flag; plain "0" or empty will not
EXTRA_ARGS += $(if $(filter 1 true TRUE yes YES on ON,$(NO_AUTO_SEARCH)), --no-auto-search)
EXTRA_ARGS += $(if $(EMBEDDING_MODEL), --embedding-model $(EMBEDDING_MODEL))
EXTRA_ARGS += $(if $(EMBEDDING_SIMILARITY_THRESHOLD), --embedding-similarity-threshold $(EMBEDDING_SIMILARITY_THRESHOLD))
EXTRA_ARGS += $(if $(EMBEDDING_HISTORY_DECAY), --embedding-history-decay $(EMBEDDING_HISTORY_DECAY))
EXTRA_ARGS += $(if $(EMBEDDING_RESULT_SIMILARITY_THRESHOLD), --embedding-result-similarity-threshold $(EMBEDDING_RESULT_SIMILARITY_THRESHOLD))
EXTRA_ARGS += $(if $(EMBEDDING_QUERY_SIMILARITY_THRESHOLD), --embedding-query-similarity-threshold $(EMBEDDING_QUERY_SIMILARITY_THRESHOLD))

.PHONY: help venv install dev-setup pull-model serve-ollama run ask run-no-search run-search check-ollama check smoke clean

help: ## Show available targets
	@awk 'BEGIN{FS":.*?## "};/^[a-zA-Z0-9_.-]+:.*?## /{printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv: ## Create local virtualenv at .venv if missing
	@test -d .venv || python3 -m venv .venv
	@$(PIP) install -U pip

install: venv ## Install project dependencies
	@$(PIP) install -r requirements.txt

dev-setup: install pull-model ## Install deps and pull the default model

pull-model: ## Pull Ollama model (override MODEL=name:tag)
	@command -v ollama >/dev/null 2>&1 || { echo "Ollama CLI not found. See README to install runtime."; exit 1; }
	@echo "Pulling model: $(MODEL)"
	@ollama pull $(MODEL)

serve-ollama: ## Run Ollama server in foreground
	@command -v ollama >/dev/null 2>&1 || { echo "Ollama CLI not found. See README to install runtime."; exit 1; }
	@ollama serve

run: ## Start interactive agent
	@$(PY) -m src.main --model $(MODEL) $(EXTRA_ARGS)

ask: ## Ask one-shot question: make ask QUESTION="What is 2+2?"
	@test -n "$(QUESTION)" || { echo "Provide QUESTION=\"...\""; exit 1; }
	@$(PY) -m src.main --model $(MODEL) --question "$(QUESTION)" $(EXTRA_ARGS)

run-no-search: ## One-shot w/o web search: make run-no-search QUESTION="Derive the quadratic formula"
	@test -n "$(QUESTION)" || { echo "Provide QUESTION=\"...\""; exit 1; }
	@$(PY) -m src.main --model $(MODEL) --no-auto-search --question "$(QUESTION)" $(EXTRA_ARGS)

# Tunables (align with CLI flags, hyphens → underscores). Examples:
# make run-search QUESTION="Capital of France?" MAX_ROUNDS=1 SEARCH_MAX_RESULTS=2 DDG_BACKEND=duckduckgo LOG_LEVEL=INFO
run-search: ## One-shot with web search
	@test -n "$(QUESTION)" || { echo "Provide QUESTION=\"...\""; exit 1; }
	@$(PY) -m src.main --model $(MODEL) --question "$(QUESTION)" $(EXTRA_ARGS)

check-ollama: ## Check Ollama server and list local models
	@$(CURL) http://localhost:11434/api/tags | head -c 400 && echo || { echo "Ollama not responding on :11434"; exit 1; }

check: ## Quick import check of agent + config
	@$(PY) -c "from src.config import AgentConfig; from src.agent import Agent; cfg=AgentConfig(no_auto_search=True, question='healthcheck'); Agent(cfg); print('OK')"

smoke: install ## Install deps then run the no-network smoke test
	@$(PY) -m scripts.smoke

clean: ## Remove cache files
	@find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name '*.pyc' -delete 2>/dev/null || true
