from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING, cast

from . import config as _config

if TYPE_CHECKING:  # make the AgentConfig type available to type checkers
    from .config import AgentConfig


@dataclass
class ArgSpec:
    """Specification for a command-line argument."""

    name: str
    short: str | None = None
    arg_type: type | None = None
    default_attr: str | None = None
    action: Any = None  # Can be str or argparse.BooleanOptionalAction
    choices: list[str] | None = None
    help_text: str = ""


# Argument specifications in logical groups
_ARG_SPECS: list[ArgSpec] = [
    # Model selection
    ArgSpec("--robot-model", "--rm", default_attr="robot_model", help_text="Ollama model for robot chains"),
    ArgSpec("--assistant-model", "--am", default_attr="assistant_model", help_text="Ollama model for assistant chains"),
    # Search control
    ArgSpec(
        "--no-auto-search",
        "--nas",
        action="store_true",
        default_attr="no_auto_search",
        help_text="Disable automatic web search decision",
    ),
    ArgSpec(
        "--force-search",
        "--fs",
        action="store_true",
        default_attr="force_search",
        help_text="Always perform web search and skip the search decision classifier",
    ),
    # Search parameters
    ArgSpec("--max-rounds", "--mr", arg_type=int, default_attr="max_rounds", help_text="Maximum search rounds"),
    ArgSpec(
        "--max-conversation-chars",
        "--mcc",
        arg_type=int,
        default_attr="max_conversation_chars",
        help_text="Maximum characters to keep in conversation history (default: 64000 ≈ 16k tokens)",
    ),
    ArgSpec(
        "--compact-keep-turns",
        "--ckt",
        arg_type=int,
        default_attr="compact_keep_turns",
        help_text="Number of recent turns to keep when compacting (default: 10)",
    ),
    ArgSpec(
        "--max-followup-suggestions",
        "--mfs",
        arg_type=int,
        default_attr="max_followup_suggestions",
        help_text="Max planning suggestions per cycle",
    ),
    ArgSpec(
        "--max-fill-attempts",
        "--mfa",
        arg_type=int,
        default_attr="max_fill_attempts",
        help_text="How many planning fill attempts",
    ),
    ArgSpec(
        "--max-relevance-llm-checks",
        "--mrlc",
        arg_type=int,
        default_attr="max_relevance_llm_checks",
        help_text="LLM checks allowed for borderline relevance",
    ),
    # LLM context parameters
    ArgSpec(
        "--assistant-num-ctx",
        "--anc",
        arg_type=int,
        default_attr="assistant_num_ctx",
        help_text="Context window tokens for assistant chains",
    ),
    ArgSpec(
        "--robot-num-ctx",
        "--rnc",
        arg_type=int,
        default_attr="robot_num_ctx",
        help_text="Context window tokens for robot chains",
    ),
    ArgSpec(
        "--assistant-num-predict",
        "--anp",
        arg_type=int,
        default_attr="assistant_num_predict",
        help_text="Max tokens to predict for assistant chains",
    ),
    ArgSpec(
        "--robot-num-predict",
        "--rnp",
        arg_type=int,
        default_attr="robot_num_predict",
        help_text="Max tokens to predict for robot chains",
    ),
    # LLM sampling parameters
    ArgSpec("--robot-temp", "--rt", arg_type=float, default_attr="robot_temp", help_text="Temperature for robot LLM"),
    ArgSpec(
        "--assistant-temp",
        "--at",
        arg_type=float,
        default_attr="assistant_temp",
        help_text="Temperature for assistant LLM",
    ),
    ArgSpec("--robot-top-p", "--rtp", arg_type=float, default_attr="robot_top_p", help_text="Top-p for robot LLM"),
    ArgSpec(
        "--assistant-top-p",
        "--atp",
        arg_type=float,
        default_attr="assistant_top_p",
        help_text="Top-p for assistant LLM",
    ),
    ArgSpec("--robot-top-k", "--rtk", arg_type=int, default_attr="robot_top_k", help_text="Top-k for robot LLM"),
    ArgSpec(
        "--assistant-top-k", "--atk", arg_type=int, default_attr="assistant_top_k", help_text="Top-k for assistant LLM"
    ),
    ArgSpec(
        "--robot-repeat-penalty",
        "--rrp",
        arg_type=float,
        default_attr="robot_repeat_penalty",
        help_text="Repeat penalty for robot LLM",
    ),
    ArgSpec(
        "--assistant-repeat-penalty",
        "--arp",
        arg_type=float,
        default_attr="assistant_repeat_penalty",
        help_text="Repeat penalty for assistant LLM",
    ),
    # DuckDuckGo search settings
    ArgSpec("--ddg-region", "--dr", default_attr="ddg_region", help_text="DDGS region hint, e.g. us-en, uk-en, de-de"),
    ArgSpec(
        "--ddg-safesearch",
        "--dss",
        default_attr="ddg_safesearch",
        choices=["off", "moderate", "strict"],
        help_text="DDGS safesearch level",
    ),
    ArgSpec(
        "--ddg-backend",
        "--db",
        default_attr="ddg_backend",
        help_text="Comma-separated DDGS backends. Use 'auto' for provider mix or pick engines like duckduckgo, bing, brave.",
    ),
    ArgSpec(
        "--search-max-results",
        "--smr",
        arg_type=int,
        default_attr="search_max_results",
        help_text="Max results to fetch per query",
    ),
    ArgSpec(
        "--search-retries",
        "--sr",
        arg_type=int,
        default_attr="search_retries",
        help_text="Retry attempts for transient search errors",
    ),
    ArgSpec(
        "--search-timeout",
        "--st",
        arg_type=float,
        default_attr="search_timeout",
        help_text="Per-request timeout (seconds) for DDGS search calls",
    ),
    ArgSpec(
        "--max-concurrent-queries",
        "--mcq",
        arg_type=int,
        default_attr="max_concurrent_queries",
        help_text="Maximum queries to fetch in parallel per search round (1=sequential, 3=default)",
    ),
    # Logging
    ArgSpec("--log-level", "--ll", default_attr="log_level", help_text="Logging level: DEBUG, INFO, WARNING, ERROR"),
    ArgSpec("--log-file", "--lf", default_attr="log_file", help_text="Optional log file path"),
    ArgSpec(
        "--log-console",
        "--lc",
        action=argparse.BooleanOptionalAction,
        default_attr="log_console",
        help_text="Enable console logging (pass --no-log-console to silence log statements on stderr)",
    ),
    # One-shot mode
    ArgSpec("--question", "--q", default_attr="question", help_text="Run once with this question and exit"),
    # Embedding settings
    ArgSpec(
        "--embedding-model",
        "--em",
        default_attr="embedding_model",
        help_text="Ollama embedding model to power topic similarity checks",
    ),
    ArgSpec(
        "--embedding-similarity-threshold",
        "--est",
        arg_type=float,
        default_attr="embedding_similarity_threshold",
        help_text="Minimum cosine similarity to consider a topic without keyword overlap",
    ),
    ArgSpec(
        "--embedding-history-decay",
        "--ehd",
        arg_type=float,
        default_attr="embedding_history_decay",
        help_text="Weight [0-1) that keeps prior topic embeddings when blending in new turns",
    ),
    ArgSpec(
        "--embedding-result-similarity-threshold",
        "--erst",
        arg_type=float,
        default_attr="embedding_result_similarity_threshold",
        help_text="Semantic similarity needed for search results to skip the LLM relevance gate",
    ),
    ArgSpec(
        "--embedding-query-similarity-threshold",
        "--eqst",
        arg_type=float,
        default_attr="embedding_query_similarity_threshold",
        help_text="Semantic similarity needed before a planned query is sent to the LLM filter",
    ),
]


def _add_argument_from_spec(parser: argparse.ArgumentParser, spec: ArgSpec, defaults: "AgentConfig") -> None:
    """Add a single argument to the parser from its specification.

    Args:
        parser: ArgumentParser to add to
        spec: Argument specification
        defaults: Default config to extract default value from
    """
    # Build argument names
    names = [spec.name]
    if spec.short:
        names.append(spec.short)

    # Build kwargs for add_argument
    kwargs: dict[str, Any] = {"help": spec.help_text}

    if spec.default_attr:
        kwargs["default"] = getattr(defaults, spec.default_attr)

    if spec.arg_type:
        kwargs["type"] = spec.arg_type

    if spec.action:
        kwargs["action"] = spec.action

    if spec.choices:
        kwargs["choices"] = spec.choices

    parser.add_argument(*names, **kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser using data-driven specification.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description="Local AI Agent with optional web search")
    defaults: "AgentConfig" = cast("type[AgentConfig]", _config.AgentConfig)()

    # Add all arguments from specifications
    for spec in _ARG_SPECS:
        _add_argument_from_spec(parser, spec, defaults)

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
