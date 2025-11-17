from __future__ import annotations

import argparse

# Prefer absolute imports when executed as a module: `python -m src.main`.
# Only fall back to local (script-style) imports when the 'src' package
# itself is missing. Do NOT mask dependency errors from inside submodules.
try:
    from src.cli import build_arg_parser, configure_logging
    from src.config import AgentConfig
    from src.agent import Agent
except ModuleNotFoundError as e:  # fallback only if the 'src' package is missing
    pkg = getattr(e, "name", "")
    if pkg and pkg.split(".")[0] == "src":
        from cli import build_arg_parser, configure_logging  # type: ignore
        from config import AgentConfig  # type: ignore
        from agent import Agent  # type: ignore
    else:
        # Propagate real dependency errors (e.g., langchain_community not installed)
        raise


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = build_arg_parser()
        args = parser.parse_args()
    configure_logging(args.log_level, args.log_file)
    cfg = AgentConfig(**vars(args))
    agent = Agent(cfg)
    if args.question:
        agent.answer_once(args.question.strip())
    else:
        agent.run()


if __name__ == "__main__":
    main()
