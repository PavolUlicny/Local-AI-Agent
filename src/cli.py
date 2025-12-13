from __future__ import annotations

import argparse
import logging
import sys

from typing import TYPE_CHECKING, Any, cast
import importlib

_AgentConfig = None

if TYPE_CHECKING:  # make the AgentConfig type available to type checkers
    from src.config import AgentConfig
else:
    try:
        _config = importlib.import_module("src.config")
    except ModuleNotFoundError as exc:  # pragma: no cover - fallback for script-style runs
        missing_root = getattr(exc, "name", "").split(".")[0]
        if missing_root != "src":
            raise
        _config = importlib.import_module("config")

    _AgentConfig: Any = _config.AgentConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local AI Agent with optional web search")

    defaults: "AgentConfig" = cast("type[AgentConfig]", _AgentConfig)()

    parser.add_argument(
        "--robot-model",
        "--rm",
        default=defaults.robot_model,
        help="Ollama model to use for robot (planning/classifier) chains",
    )
    parser.add_argument(
        "--assistant-model",
        "--am",
        default=defaults.assistant_model,
        help="Ollama model to use for assistant (final answer) chains",
    )
    parser.add_argument(
        "--no-auto-search",
        "--nas",
        action="store_true",
        default=defaults.no_auto_search,
        help="Disable automatic web search decision",
    )
    parser.add_argument(
        "--force-search",
        "--fs",
        action="store_true",
        default=defaults.force_search,
        help="Always perform web search and skip the search decision classifier",
    )
    parser.add_argument("--max-rounds", "--mr", type=int, default=defaults.max_rounds, help="Maximum search rounds")
    parser.add_argument(
        "--max-context-turns",
        "--mct",
        type=int,
        default=defaults.max_context_turns,
        help="Max turns from topic to keep in context",
    )
    parser.add_argument(
        "--max-followup-suggestions",
        "--mfs",
        type=int,
        default=defaults.max_followup_suggestions,
        help="Max planning suggestions per cycle",
    )
    parser.add_argument(
        "--max-fill-attempts",
        "--mfa",
        type=int,
        default=defaults.max_fill_attempts,
        help="How many planning fill attempts",
    )
    parser.add_argument(
        "--max-relevance-llm-checks",
        "--mrlc",
        type=int,
        default=defaults.max_relevance_llm_checks,
        help="LLM checks allowed for borderline relevance",
    )
    parser.add_argument(
        "--assistant-num-ctx",
        "--anc",
        type=int,
        default=defaults.assistant_num_ctx,
        help="Context window tokens for assistant chains",
    )
    parser.add_argument(
        "--robot-num-ctx",
        "--rnc",
        type=int,
        default=defaults.robot_num_ctx,
        help="Context window tokens for robot (classifier/planner) chains",
    )
    parser.add_argument(
        "--assistant-num-predict",
        "--anp",
        type=int,
        default=defaults.assistant_num_predict,
        help="Max tokens to predict for assistant chains",
    )
    parser.add_argument(
        "--robot-num-predict",
        "--rnp",
        type=int,
        default=defaults.robot_num_predict,
        help="Max tokens to predict for robot (classifier/planner) chains",
    )
    parser.add_argument(
        "--robot-temp", "--rt", type=float, default=defaults.robot_temp, help="Temperature for robot LLM"
    )
    parser.add_argument(
        "--assistant-temp", "--at", type=float, default=defaults.assistant_temp, help="Temperature for assistant LLM"
    )
    parser.add_argument("--robot-top-p", "--rtp", type=float, default=defaults.robot_top_p, help="Top-p for robot LLM")
    parser.add_argument(
        "--assistant-top-p", "--atp", type=float, default=defaults.assistant_top_p, help="Top-p for assistant LLM"
    )
    parser.add_argument("--robot-top-k", "--rtk", type=int, default=defaults.robot_top_k, help="Top-k for robot LLM")
    parser.add_argument(
        "--assistant-top-k", "--atk", type=int, default=defaults.assistant_top_k, help="Top-k for assistant LLM"
    )
    parser.add_argument(
        "--robot-repeat-penalty",
        "--rrp",
        type=float,
        default=defaults.robot_repeat_penalty,
        help="Repeat penalty for robot LLM",
    )
    parser.add_argument(
        "--assistant-repeat-penalty",
        "--arp",
        type=float,
        default=defaults.assistant_repeat_penalty,
        help="Repeat penalty for assistant LLM",
    )
    parser.add_argument(
        "--ddg-region", "--dr", default=defaults.ddg_region, help="DDGS region hint, e.g. us-en, uk-en, de-de"
    )
    parser.add_argument(
        "--ddg-safesearch",
        "--dss",
        default=defaults.ddg_safesearch,
        choices=["off", "moderate", "strict"],
        help="DDGS safesearch level",
    )
    parser.add_argument(
        "--ddg-backend",
        "--db",
        default=defaults.ddg_backend,
        help=(
            "Comma-separated DDGS backends. Use 'auto' for provider mix or pick engines like duckduckgo, bing, brave."
        ),
    )
    parser.add_argument(
        "--search-max-results",
        "--smr",
        type=int,
        default=defaults.search_max_results,
        help="Max results to fetch per query",
    )
    parser.add_argument(
        "--search-retries",
        "--sr",
        type=int,
        default=defaults.search_retries,
        help="Retry attempts for transient search errors",
    )
    parser.add_argument(
        "--search-timeout",
        "--st",
        type=float,
        default=defaults.search_timeout,
        help="Per-request timeout (seconds) for DDGS search calls",
    )
    parser.add_argument(
        "--log-level", "--ll", default=defaults.log_level, help="Logging level: DEBUG, INFO, WARNING, ERROR"
    )
    parser.add_argument("--log-file", "--lf", default=defaults.log_file, help="Optional log file path")
    parser.add_argument(
        "--log-console",
        "--lc",
        action=argparse.BooleanOptionalAction,
        default=defaults.log_console,
        help="Enable console logging (pass --no-log-console to silence log statements on stderr)",
    )
    parser.add_argument("--question", "--q", default=defaults.question, help="Run once with this question and exit")
    parser.add_argument(
        "--embedding-model",
        "--em",
        default=defaults.embedding_model,
        help="Ollama embedding model to power topic similarity checks",
    )
    parser.add_argument(
        "--embedding-similarity-threshold",
        "--est",
        type=float,
        default=defaults.embedding_similarity_threshold,
        help="Minimum cosine similarity to consider a topic without keyword overlap",
    )
    parser.add_argument(
        "--embedding-history-decay",
        "--ehd",
        type=float,
        default=defaults.embedding_history_decay,
        help="Weight [0-1) that keeps prior topic embeddings when blending in new turns",
    )
    parser.add_argument(
        "--embedding-result-similarity-threshold",
        "--erst",
        type=float,
        default=defaults.embedding_result_similarity_threshold,
        help="Semantic similarity needed for search results to skip the LLM relevance gate",
    )
    parser.add_argument(
        "--embedding-query-similarity-threshold",
        "--eqst",
        type=float,
        default=defaults.embedding_query_similarity_threshold,
        help="Semantic similarity needed before a planned query is sent to the LLM filter",
    )
    return parser


def configure_logging(level: str, log_file: str | None, log_console: bool = True, *, force: bool = True) -> None:
    level_upper = (level or "INFO").upper()
    numeric = getattr(logging, level_upper, logging.INFO)
    handlers: list[logging.Handler] = []
    if log_console:
        console_handler = logging.StreamHandler(sys.stderr)
        handlers.append(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)
    if not handlers:
        # Respect --no-log-console even without a log file by discarding logs via NullHandler
        handlers.append(logging.NullHandler())
    if handlers:
        logging.basicConfig(
            level=numeric,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=handlers,
            force=force,
        )
    else:
        logging.getLogger().setLevel(numeric)
