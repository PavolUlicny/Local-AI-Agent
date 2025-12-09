from __future__ import annotations

from dataclasses import dataclass
import warnings


@dataclass(slots=True)
class AgentConfig:
    model: str = "cogito:8b"
    no_auto_search: bool = False
    force_search: bool = False
    max_rounds: int = 12
    max_context_turns: int = 8
    max_followup_suggestions: int = 6
    max_fill_attempts: int = 3
    max_relevance_llm_checks: int = 2
    assistant_num_ctx: int = 8192
    robot_num_ctx: int = 8192
    assistant_num_predict: int = 4096
    robot_num_predict: int = 512
    robot_temp: float = 0.0
    assistant_temp: float = 0.6
    robot_top_p: float = 0.4
    assistant_top_p: float = 0.8
    robot_top_k: int = 20
    assistant_top_k: int = 80
    robot_repeat_penalty: float = 1.1
    assistant_repeat_penalty: float = 1.2
    ddg_region: str = "us-en"
    ddg_safesearch: str = "moderate"
    ddg_backend: str = "auto"
    search_max_results: int = 5
    search_retries: int = 3
    search_timeout: float = 10.0
    log_level: str = "WARNING"
    log_file: str | None = None
    log_console: bool = True
    question: str | None = None
    embedding_model: str = "embeddinggemma:300m"
    embedding_similarity_threshold: float = 0.35
    embedding_history_decay: float = 0.65
    embedding_result_similarity_threshold: float = 0.5
    embedding_query_similarity_threshold: float = 0.3

    @property
    def auto_search_decision(self) -> bool:
        return not self.no_auto_search

    def __post_init__(self) -> None:
        # Up-front safety caps to reduce context-length rebuilds and OOMs on common 7B/8B local models.
        max_safe_ctx = 8192
        min_assistant_predict = 512
        min_robot_predict = 128

        if self.assistant_num_ctx > max_safe_ctx:
            warnings.warn(
                f"assistant_num_ctx capped to {max_safe_ctx} to fit typical 7B/8B contexts; was {self.assistant_num_ctx}",
                RuntimeWarning,
                stacklevel=2,
            )
            self.assistant_num_ctx = max_safe_ctx
        if self.robot_num_ctx > max_safe_ctx:
            warnings.warn(
                f"robot_num_ctx capped to {max_safe_ctx} to fit typical 7B/8B contexts; was {self.robot_num_ctx}",
                RuntimeWarning,
                stacklevel=2,
            )
            self.robot_num_ctx = max_safe_ctx

        if self.assistant_num_predict > self.assistant_num_ctx:
            warnings.warn(
                "assistant_num_predict cannot exceed assistant_num_ctx; capping to context window",
                RuntimeWarning,
                stacklevel=2,
            )
            self.assistant_num_predict = max(min_assistant_predict, self.assistant_num_ctx)
        elif self.assistant_num_predict < min_assistant_predict:
            self.assistant_num_predict = min_assistant_predict

        if self.robot_num_predict > self.robot_num_ctx:
            warnings.warn(
                "robot_num_predict cannot exceed robot_num_ctx; capping to context window",
                RuntimeWarning,
                stacklevel=2,
            )
            self.robot_num_predict = max(min_robot_predict, self.robot_num_ctx)
        elif self.robot_num_predict < min_robot_predict:
            self.robot_num_predict = min_robot_predict
