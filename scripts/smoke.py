from __future__ import annotations

import dataclasses
import sys


def main() -> int:
    # Ensure project root on path if executed directly
    # (GitHub Actions runs from repo root, so this is just extra safety.)
    sys.path.insert(0, ".")

    try:
        from src.cli import build_arg_parser
        from src.config import AgentConfig
        from src.agent import Agent
    except Exception as e:  # pragma: no cover - CI discovery failure
        print("IMPORT_FAIL:", type(e).__name__, str(e))
        return 1

    try:
        # 1) CLI defaults must equal AgentConfig defaults (single source of truth)
        parser = build_arg_parser()
        ns = parser.parse_args([])
        ns_defaults = vars(ns)
        cfg_defaults = dataclasses.asdict(AgentConfig())

        for k, v in cfg_defaults.items():
            if ns_defaults.get(k, object()) != v:
                print(f"DEFAULT_MISMATCH: {k}: cli={ns_defaults.get(k)} cfg={v}")
                return 1

        # 2) Instantiate Agent without invoking LLM or network
        #    (OllamaLLM objects are lazily used; DuckDuckGo wrapper is constructed only.)
        cfg = AgentConfig(
            no_auto_search=True,
            question="healthcheck",
            num_ctx=2048,
            num_predict=1024,
        )
        agent = Agent(cfg)

        # 3) Chains present and have expected interfaces
        expected = {
            "context",
            "seed",
            "planning",
            "result_filter",
            "query_filter",
            "search_decision",
            "response",
            "response_no_search",
        }
        keys = set(agent.chains.keys())
        missing = sorted(expected - keys)
        if missing:
            print("MISSING_CHAINS:", ", ".join(missing))
            return 1

        if not hasattr(agent.chains["context"], "invoke"):
            print("CHAIN_API_ERROR: context chain missing 'invoke'")
            return 1
        if not hasattr(agent.chains["response"], "stream"):
            print("CHAIN_API_ERROR: response chain missing 'stream'")
            return 1

        # 4) Verify LLM params applied from config
        if not (
            agent.llm_robot.num_ctx == cfg.num_ctx
            and agent.llm_assistant.num_ctx == cfg.num_ctx
            and agent.llm_robot.num_predict == cfg.num_predict
            and agent.llm_assistant.num_predict == cfg.num_predict
        ):
            print("LLM_PARAM_MISMATCH: num_ctx/num_predict not applied")
            return 1

        # 5) Search wrapper aligned with config
        if not (
            agent.search_api.region == cfg.ddg_region
            and agent.search_api.safesearch == cfg.ddg_safesearch
            and agent.search_api.backend == cfg.ddg_backend
        ):
            print("SEARCH_WRAPPER_MISMATCH: ddg settings not applied")
            return 1

        print("SMOKE_OK")
        return 0

    except Exception as e:  # pragma: no cover - defensive CI capture
        print("SMOKE_FAIL:", type(e).__name__, str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
