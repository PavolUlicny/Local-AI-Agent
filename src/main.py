from __future__ import annotations

import argparse
import sys
import logging

# Support both module execution (python -m src.main) and script execution
# When run as a module, use relative imports. When run as a script, use absolute imports.
if __package__:
    # Running as module: python -m src.main
    from .cli import build_arg_parser, configure_logging
    from .config import AgentConfig
    from .agent import Agent
    from .ollama import check_and_start_ollama
else:
    # Running as script: python src/main.py
    # Add parent directory to path to support script execution
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.cli import build_arg_parser, configure_logging
    from src.config import AgentConfig
    from src.agent import Agent
    from src.ollama import check_and_start_ollama


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = build_arg_parser()
        args = parser.parse_args()
    configure_logging(args.log_level, args.log_file, args.log_console)
    logger = logging.getLogger(__name__)

    # Ensure Ollama is installed and available before continuing. If Ollama
    # is missing or cannot be started/responding, exit early with a non-zero
    # status to avoid running the agent in a degraded state. `ensure_available`
    # returns a boolean so the caller can decide how to handle failures.
    try:
        # Run the full Ollama check/start/wait workflow. Exit the process
        # on failure to match previous strict behavior.
        check_and_start_ollama(exit_on_failure=True)
    except Exception as e:
        logger.error("Unexpected error while checking Ollama availability: %s", e)
        sys.exit(1)
    cfg = AgentConfig(**vars(args))
    agent = Agent(cfg)
    if args.question:
        agent.answer_once(args.question.strip())
    else:
        agent.run()


if __name__ == "__main__":
    main()
