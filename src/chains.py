from __future__ import annotations

from typing import Dict, Tuple
from langchain_core.output_parsers import StrOutputParser

try:  # local or package import flexibility
    from src.prompts import (
        context_mode_prompt,
        seed_prompt,
        planning_prompt,
        result_filter_prompt,
        query_filter_prompt,
        search_decision_prompt,
        response_prompt,
        response_prompt_no_search,
    )
except ImportError:  # pragma: no cover - fallback
    from prompts import (
        context_mode_prompt,
        seed_prompt,
        planning_prompt,
        result_filter_prompt,
        query_filter_prompt,
        search_decision_prompt,
        response_prompt,
        response_prompt_no_search,
    )

from langchain_ollama import OllamaLLM
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # only needed for type hints
    try:
        from src.config import AgentConfig  # type: ignore
    except ImportError:  # pragma: no cover - fallback
        from config import AgentConfig  # type: ignore


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


def build_chains(llm_robot: OllamaLLM, llm_assistant: OllamaLLM) -> Dict[str, object]:
    return {
        "context": context_mode_prompt | llm_robot | StrOutputParser(),
        "seed": seed_prompt | llm_robot | StrOutputParser(),
        "planning": planning_prompt | llm_robot | StrOutputParser(),
        "result_filter": result_filter_prompt | llm_robot | StrOutputParser(),
        "query_filter": query_filter_prompt | llm_robot | StrOutputParser(),
        "search_decision": search_decision_prompt | llm_robot | StrOutputParser(),
        "response": response_prompt | llm_assistant | StrOutputParser(),
        "response_no_search": response_prompt_no_search | llm_assistant | StrOutputParser(),
    }

__all__ = ["build_llms", "build_chains"]
