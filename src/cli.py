from __future__ import annotations

import argparse
import logging
import sys

try:
    from src.config import AgentConfig  # type: ignore
except ImportError:  # pragma: no cover - fallback for script-style runs
    from config import AgentConfig  # type: ignore


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local AI Agent with optional web search")

    defaults: AgentConfig = AgentConfig()

    parser.add_argument("--model", default=defaults.model, help="Ollama model to use")
    parser.add_argument("--no-auto-search", action="store_true", default=defaults.no_auto_search, help="Disable automatic web search decision")
    parser.add_argument("--max-rounds", type=int, default=defaults.max_rounds, help="Maximum search rounds")
    parser.add_argument("--max-context-turns", type=int, default=defaults.max_context_turns, help="Max turns from topic to keep in context")
    parser.add_argument("--max-followup-suggestions", type=int, default=defaults.max_followup_suggestions, help="Max planning suggestions per cycle")
    parser.add_argument("--max-fill-attempts", type=int, default=defaults.max_fill_attempts, help="How many planning fill attempts")
    parser.add_argument("--max-relevance-llm-checks", type=int, default=defaults.max_relevance_llm_checks, help="LLM checks allowed for borderline relevance")
    parser.add_argument("--num-ctx", type=int, default=defaults.num_ctx, help="Model context window tokens")
    parser.add_argument("--num-predict", type=int, default=defaults.num_predict, help="Max tokens to predict")
    parser.add_argument("--robot-temp", type=float, default=defaults.robot_temp, help="Temperature for robot LLM")
    parser.add_argument("--assistant-temp", type=float, default=defaults.assistant_temp, help="Temperature for assistant LLM")
    parser.add_argument("--robot-top-p", type=float, default=defaults.robot_top_p, help="Top-p for robot LLM")
    parser.add_argument("--assistant-top-p", type=float, default=defaults.assistant_top_p, help="Top-p for assistant LLM")
    parser.add_argument("--robot-top-k", type=int, default=defaults.robot_top_k, help="Top-k for robot LLM")
    parser.add_argument("--assistant-top-k", type=int, default=defaults.assistant_top_k, help="Top-k for assistant LLM")
    parser.add_argument("--robot-repeat-penalty", type=float, default=defaults.robot_repeat_penalty, help="Repeat penalty for robot LLM")
    parser.add_argument("--assistant-repeat-penalty", type=float, default=defaults.assistant_repeat_penalty, help="Repeat penalty for assistant LLM")
    parser.add_argument("--ddg-region", default=defaults.ddg_region, help="DuckDuckGo region, e.g. us-en, uk-en, de-de")
    parser.add_argument("--ddg-safesearch", default=defaults.ddg_safesearch, choices=["off", "moderate", "strict"], help="DuckDuckGo safesearch level")
    parser.add_argument("--ddg-backend", default=defaults.ddg_backend, choices=["html", "lite", "api"], help="DuckDuckGo backend to use")
    parser.add_argument("--search-max-results", type=int, default=defaults.search_max_results, help="Max results to fetch per query")
    parser.add_argument("--search-retries", type=int, default=defaults.search_retries, help="Retry attempts for transient search errors")
    parser.add_argument("--log-level", default=defaults.log_level, help="Logging level: DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--log-file", default=defaults.log_file, help="Optional log file path")
    parser.add_argument("--question", default=defaults.question, help="Run once with this question and exit")
    return parser


def configure_logging(level: str, log_file: str | None) -> None:
    level_upper = (level or "INFO").upper()
    numeric = getattr(logging, level_upper, logging.INFO)
    handlers = []
    console_handler = logging.StreamHandler(sys.stderr)
    handlers.append(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )
