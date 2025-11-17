from __future__ import annotations

from typing import Any, Dict, Tuple
from langchain_core.output_parsers import StrOutputParser

import importlib

try:  # local or package import flexibility
    _prompts = importlib.import_module("src.prompts")
except Exception:  # pragma: no cover - fallback
    _prompts = importlib.import_module("prompts")

from langchain_ollama import OllamaLLM
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # only needed for type hints
    from src import config as _config

    AgentConfig = _config.AgentConfig


def build_llms(cfg: "AgentConfig") -> Tuple[OllamaLLM, OllamaLLM]:
    llm_robot = OllamaLLM(
        model=cfg.model,
        temperature=cfg.robot_temp,
        top_p=cfg.robot_top_p,
        top_k=cfg.robot_top_k,
        repeat_penalty=cfg.robot_repeat_penalty,
        num_predict=cfg.num_predict,
        num_ctx=cfg.num_ctx,
    )
    llm_assistant = OllamaLLM(
        model=cfg.model,
        temperature=cfg.assistant_temp,
        top_p=cfg.assistant_top_p,
        top_k=cfg.assistant_top_k,
        repeat_penalty=cfg.assistant_repeat_penalty,
        num_predict=cfg.num_predict,
        num_ctx=cfg.num_ctx,
    )
    return llm_robot, llm_assistant


def build_chains(llm_robot: OllamaLLM, llm_assistant: OllamaLLM) -> Dict[str, Any]:
    return {
        "context": _prompts.context_mode_prompt | llm_robot | StrOutputParser(),
        "seed": _prompts.seed_prompt | llm_robot | StrOutputParser(),
        "planning": _prompts.planning_prompt | llm_robot | StrOutputParser(),
        "result_filter": _prompts.result_filter_prompt | llm_robot | StrOutputParser(),
        "query_filter": _prompts.query_filter_prompt | llm_robot | StrOutputParser(),
        "search_decision": _prompts.search_decision_prompt | llm_robot | StrOutputParser(),
        "response": _prompts.response_prompt | llm_assistant | StrOutputParser(),
        "response_no_search": _prompts.response_prompt_no_search | llm_assistant | StrOutputParser(),
    }


__all__ = ["build_llms", "build_chains"]
