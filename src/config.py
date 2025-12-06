from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AgentConfig:
    model: str = "cogito:8b"
    no_auto_search: bool = False
    max_rounds: int = 12
    max_context_turns: int = 8
    max_followup_suggestions: int = 6
    max_fill_attempts: int = 3
    max_relevance_llm_checks: int = 2
    num_ctx: int = 12288
    num_predict: int = 8192
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
    search_retries: int = 4
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
