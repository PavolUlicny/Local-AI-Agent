from __future__ import annotations

from typing import Any, Dict, Tuple, TYPE_CHECKING
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

from . import prompts as _prompts

if TYPE_CHECKING:  # only needed for type hints
    from . import config as _config

    AgentConfig = _config.AgentConfig


def build_llms(cfg: "AgentConfig") -> Tuple[OllamaLLM, OllamaLLM]:
    robot_ctx = cfg.robot_num_ctx
    assistant_ctx = cfg.assistant_num_ctx
    robot_predict = cfg.robot_num_predict
    assistant_predict = cfg.assistant_num_predict
    # use the explicitly configured role models
    robot_model_name = cfg.robot_model
    assistant_model_name = cfg.assistant_model

    llm_robot = OllamaLLM(
        model=robot_model_name,
        temperature=cfg.robot_temp,
        top_p=cfg.robot_top_p,
        top_k=cfg.robot_top_k,
        repeat_penalty=cfg.robot_repeat_penalty,
        num_predict=robot_predict,
        num_ctx=robot_ctx,
    )
    llm_assistant = OllamaLLM(
        model=assistant_model_name,
        temperature=cfg.assistant_temp,
        top_p=cfg.assistant_top_p,
        top_k=cfg.assistant_top_k,
        repeat_penalty=cfg.assistant_repeat_penalty,
        num_predict=assistant_predict,
        num_ctx=assistant_ctx,
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
