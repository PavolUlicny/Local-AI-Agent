from __future__ import annotations

from typing import Dict, Tuple, TYPE_CHECKING
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

from . import prompts as _prompts
from .constants import ChainName

if TYPE_CHECKING:  # only needed for type hints
    from . import config as _config
    from .protocols import LLMChain

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
        keep_alive="5m",
    )
    llm_assistant = OllamaLLM(
        model=assistant_model_name,
        temperature=cfg.assistant_temp,
        top_p=cfg.assistant_top_p,
        top_k=cfg.assistant_top_k,
        repeat_penalty=cfg.assistant_repeat_penalty,
        num_predict=assistant_predict,
        num_ctx=assistant_ctx,
        keep_alive="0",
    )
    return llm_robot, llm_assistant


def build_chains(llm_robot: OllamaLLM, llm_assistant: OllamaLLM) -> Dict[ChainName, "LLMChain"]:
    """Build all LLM chains with type-safe chain names.

    Args:
        llm_robot: Configured robot LLM for quick reasoning tasks
        llm_assistant: Configured assistant LLM for response generation

    Returns:
        Dictionary mapping ChainName enum values to configured LLM chains
    """
    chains: Dict[ChainName, "LLMChain"] = {
        ChainName.SEED: _prompts.seed_prompt | llm_robot | StrOutputParser(),
        ChainName.PLANNING: _prompts.planning_prompt | llm_robot | StrOutputParser(),
        ChainName.RESULT_FILTER: _prompts.result_filter_prompt | llm_robot | StrOutputParser(),
        ChainName.QUERY_FILTER: _prompts.query_filter_prompt | llm_robot | StrOutputParser(),
        ChainName.SEARCH_DECISION: _prompts.search_decision_prompt | llm_robot | StrOutputParser(),
        ChainName.RESPONSE: _prompts.response_prompt | llm_assistant | StrOutputParser(),
        ChainName.RESPONSE_NO_SEARCH: _prompts.response_prompt_no_search | llm_assistant | StrOutputParser(),
    }
    return chains


__all__ = ["build_llms", "build_chains"]
