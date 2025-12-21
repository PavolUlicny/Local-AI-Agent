from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Any
from datetime import datetime, timezone
import logging

from . import text_utils as _text_utils_mod
from . import topic_utils as _topic_utils_mod
from . import keywords as _keywords_mod
from . import exceptions as _exceptions
from . import model_utils as _model_utils_mod


@dataclass(frozen=True)
class QueryContext:
    """Immutable context snapshot for processing a single user query.

    Contains all information needed for query processing: datetime, user input,
    conversation history, topic keywords, and selected topic context. Built once
    per query and passed through the processing pipeline.
    """

    current_datetime: str
    current_year: str
    current_month: str
    current_day: str
    question_keywords: List[str]
    question_embedding: List[float] | None
    selected_topic_index: int | None
    recent_history: list
    topic_keywords: Set[str]
    topic_brief_text: str
    topic_embedding_current: List[float] | None
    recent_for_prompt: list
    conversation_text: str
    prior_responses_text: str


def build_query_context(agent: Any, user_query: str) -> QueryContext:
    """Build a QueryContext for the given `agent` and `user_query`.

    This mirrors the original logic embedded in `Agent._build_query_context`.
    """
    cfg = agent.cfg
    current_datetime = _text_utils_mod.current_datetime_utc()
    dt_obj = datetime.now(timezone.utc)
    current_year = str(dt_obj.year)
    current_month = f"{dt_obj.month:02d}"
    current_day = f"{dt_obj.day:02d}"
    question_keywords = list(_keywords_mod.extract_keywords(user_query))
    question_embedding = agent.embedding_client.embed(user_query)
    try:
        selected_topic_index, recent_history, topic_keywords = _topic_utils_mod.select_topic(
            agent.chains["context"],
            agent.topics,
            user_query,
            set(question_keywords),
            cfg.max_context_turns,
            current_datetime,
            current_year,
            current_month,
            current_day,
            question_embedding=question_embedding,
            embedding_threshold=cfg.embedding_similarity_threshold,
        )
    except _exceptions.ResponseError as exc:  # Robot model not found, etc.
        message = str(exc)
        if "not found" in message.lower():
            _model_utils_mod.handle_missing_model(agent._mark_error, "Robot", cfg.robot_model)
            return QueryContext(
                current_datetime,
                current_year,
                current_month,
                current_day,
                question_keywords,
                question_embedding,
                None,
                [],
                set(question_keywords),
                "",
                None,
                [],
                "No prior relevant conversation.",
                "No prior answers for this topic.",
            )
        selected_topic_index = None
        recent_history = []
        topic_keywords = set(question_keywords)
    except Exception as exc:  # graceful fallback
        logging.warning(f"Context classification failed; proceeding without topic selection. Error: {exc}")
        selected_topic_index = None
        recent_history = []
        topic_keywords = set(question_keywords)
    topic_brief_text = ""
    if selected_topic_index is not None and selected_topic_index < len(agent.topics):
        topic_brief_text = _topic_utils_mod.topic_brief(agent.topics[selected_topic_index])
    topic_embedding_current: List[float] | None = None
    if selected_topic_index is not None and selected_topic_index < len(agent.topics):
        topic_embedding_current = agent.topics[selected_topic_index].embedding
    recent_for_prompt = _topic_utils_mod.tail_turns(recent_history, _text_utils_mod.MAX_PROMPT_RECENT_TURNS)
    conversation_text = _topic_utils_mod.format_turns(recent_for_prompt, "No prior relevant conversation.")
    if topic_brief_text:
        conversation_text = f"Topic brief:\n{topic_brief_text}\n\nRecent turns:\n{conversation_text}"
    conversation_text = _text_utils_mod.truncate_text(
        conversation_text, agent._char_budget(_text_utils_mod.MAX_CONVERSATION_CHARS)
    )
    prior_responses_text = (
        _topic_utils_mod.collect_prior_responses(
            agent.topics[selected_topic_index], max_chars=_text_utils_mod.MAX_PRIOR_RESPONSE_CHARS
        )
        if selected_topic_index is not None
        else "No prior answers for this topic."
    )
    if topic_brief_text:
        prior_responses_text = f"{topic_brief_text}\n\nRecent answers:\n{prior_responses_text}"
    prior_responses_text = _text_utils_mod.truncate_text(
        prior_responses_text, agent._char_budget(_text_utils_mod.MAX_PRIOR_RESPONSE_CHARS)
    )
    return QueryContext(
        current_datetime,
        current_year,
        current_month,
        current_day,
        question_keywords,
        question_embedding,
        selected_topic_index,
        recent_history,
        topic_keywords,
        topic_brief_text,
        topic_embedding_current,
        recent_for_prompt,
        conversation_text,
        prior_responses_text,
    )
