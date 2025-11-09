from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
import re
from typing import List, Optional, Set, Tuple

def _resolve_prompt_template():
    module_paths = [
        "langchain_core.prompts",
        "langchain.prompts",
        "langchain.prompts.prompt",
        "langchain.schema",
    ]
    for path in module_paths:
        try:
            module = import_module(path)
            template = getattr(module, "PromptTemplate", None)
            if template is not None:
                return template
        except ImportError:
            continue
    raise ImportError(
        "Could not import PromptTemplate from LangChain. "
        "Please ensure langchain>=0.0.200 or langchain-core is installed."
    )

PromptTemplate = _resolve_prompt_template()

try:
    from langchain_ollama import OllamaLLM
except ImportError as exc:
    raise ImportError(
        "langchain-ollama is required. Install it with 'pip install -U langchain-ollama'."
    ) from exc

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

try:
    from ollama._types import ResponseError
except ImportError:
    ResponseError = Exception

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
_STOP_WORDS: Set[str] = {
    "and",
    "about",
    "are",
    "ask",
    "buy",
    "can",
    "for",
    "from",
    "give",
    "going",
    "good",
    "gonna",
    "have",
    "here",
    "into",
    "just",
    "like",
    "more",
    "need",
    "please",
    "really",
    "should",
    "that",
    "than",
    "the",
    "then",
    "there",
    "think",
    "tell",
    "this",
    "those",
    "want",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}
_GENERIC_TOKENS: Set[str] = {"question", "questions", "info"}

TopicTurns = List[Tuple[str, str]]

@dataclass
class Topic:
    keywords: Set[str] = field(default_factory=set)
    turns: TopicTurns = field(default_factory=list)

def _extract_keywords(text: str) -> Set[str]:
    if not text:
        return set()
    tokens = _TOKEN_PATTERN.findall(text.lower())
    cleaned = {token.strip("\"'.!?") for token in tokens}
    return {
        token
        for token in cleaned
        if len(token) > 2 and token not in _STOP_WORDS and token not in _GENERIC_TOKENS
    }


def _is_relevant(text: str, topic_keywords: Set[str]) -> bool:
    if not topic_keywords:
        return True
    return bool(_extract_keywords(text).intersection(topic_keywords))

def _format_turns(turns: TopicTurns, fallback: str) -> str:
    if not turns:
        return fallback
    return "\n\n".join(
        f"User: {user}\nAssistant: {assistant}" for user, assistant in turns
    )

def _collect_prior_responses(
    topic: Topic,
    limit: int = 3,
    max_chars: int = 1200,
) -> str:
    if not topic.turns:
        return "No prior answers for this topic."

    snippets: List[str] = []
    budget = max(max_chars // max(1, limit), 200)
    for _, assistant in topic.turns[-limit:]:
        snippet = assistant.strip()
        if len(snippet) > budget:
            truncated = snippet[: budget].rsplit(" ", 1)[0].rstrip(".,;:")
            snippet = f"{truncated}..."
        snippets.append(snippet)

    return "\n\n---\n\n".join(snippets) or "No prior answers for this topic."

def _select_topic(
    llm: OllamaLLM,
    context_prompt: object,
    topics: List[Topic],
    question: str,
    base_keywords: Set[str],
    max_context_turns: int,
) -> Tuple[Optional[int], TopicTurns, Set[str]]:
    if not topics:
        return None, [], set(base_keywords)

    scored = sorted(
        ((len(topic.keywords.intersection(base_keywords)), idx) for idx, topic in enumerate(topics)),
        reverse=True,
    )
    top_candidates = [item for item in scored if item[0] > 0][:3]

    if not top_candidates:
        return None, [], set(base_keywords)

    decisions: List[Tuple[str, int, TopicTurns]] = []
    for _, idx in top_candidates:
        topic = topics[idx]
        recent_turns = topic.turns[-max_context_turns:]
        decision_raw = llm.invoke(
            context_prompt.format(
                recent_conversation=_format_turns(recent_turns, "No prior conversation."),
                new_question=question,
            )
        )
        normalized = str(decision_raw).strip().upper()
        decisions.append((normalized, idx, recent_turns))
        if normalized == "FOLLOW_UP":
            return idx, recent_turns, base_keywords.union(topics[idx].keywords)

    for normalized, idx, recent_turns in decisions:
        if normalized == "EXPAND":
            return idx, recent_turns, base_keywords.union(topics[idx].keywords)

    return None, [], set(base_keywords)

def main() -> None:
    max_context_turns = 6
    max_search_rounds = 5
    max_followup_suggestions = 2
    used_model = "solar:10.7b"

    llm = OllamaLLM(
        model=used_model,
        temperature=0.85,
        top_p=0.95,
        top_k=64,
        repeat_penalty=1.05,
        num_predict=4096,
        num_ctx=4096,
    )
    search = DuckDuckGoSearchRun(
        api_wrapper=DuckDuckGoSearchAPIWrapper(region="us-en", safesearch="moderate")
    )

    response_prompt = PromptTemplate(
        input_variables=[
            "conversation_history",
            "search_results",
            "user_question",
            "prior_responses",
        ],
        template=(
            "ROLE: You are a knowledgeable and precise explainer who writes detailed, well-structured, and factual answers.\n\n"
            "TASK: Integrate all relevant information from the search results and prior discussion to produce a thorough explanation. "
            "Focus on clarity, depth, and accuracy rather than brevity. Write as if you are teaching or summarizing for understanding.\n\n"
            "You can only provide plain text responses. Do not include images, tables, charts, or any other non-text content.\n\n"
            "Do not fabricate information. If the search results do not contain relevant information, state that you could not find an answer based on the provided data.\n\n"
            "Do not reference the search process, the assistant, or any meta-commentary in your answer.\n\n"
            "Do not mention 'your knowledge cutoff' or similar phrases.\n\n"
            "Do not mention search results or searches in your answer.\n\n"
            "INCLUDE WHEN RELEVANT:\n"
            "- Key facts, definitions, mechanisms, or background context\n"
            "- Historical or technical explanations\n"
            "- Logical causes, effects, and relationships\n"
            "- Broader implications or related concepts\n"
            "- Examples or analogies that improve clarity\n\n"
            "STYLE:\n"
            "- Write in paragraphs, not lists.\n"
            "- Avoid repetition, filler phrases, or meta-commentary.\n"
            "- Do not mention searches, prompts, or the assistant itself.\n"
            "- Keep tone factual, neutral, and educational.\n"
            "- When the topic involves current or time-sensitive data, advise checking official live sources.\n\n"
            "Conversation so far:\n{conversation_history}\n\n"
            "Earlier answers:\n{prior_responses}\n\n"
            "Search evidence:\n{search_results}\n\n"
            "User question:\n{user_question}"
        ),
    )

    planning_prompt = PromptTemplate(
        input_variables=[
            "conversation_history",
            "user_question",
            "results_to_date",
            "suggestion_limit",
            "known_answers",
        ],
        template=(
            "ROLE: You suggest new web search queries.\n\n"
            "TASK: Generate up to {suggestion_limit} specific search queries "
            "that could add *new* information about the user’s question.\n\n"
            "RULES:\n"
            "- Only output the queries, one per line.\n"
            "- Do not add anything, just output the queries themselves.\n"
            "- Do not repeat already covered information.\n"
            "- Do not say anything like 'Here are some suggestions:' or 'Here are new search queries', only type the queries themselves.\n"
            "- Do not add formatting like bullet points or numbering.\n"
            "- Do not use quotes around queries.\n"
            "- Do not add any notes or commentary.\n"
            "- Do not explain or comment.\n"
            "- Output exactly one search query per line.\n"
            "- If nothing new is useful, output only: NONE.\n\n"
            "Conversation:\n{conversation_history}\n\n"
            "Known answers:\n{known_answers}\n\n"
            "User question:\n{user_question}\n\n"
            "Existing results:\n{results_to_date}"
        ),
    )

    seed_prompt = PromptTemplate(
        input_variables=["conversation_history", "user_question", "known_answers"],
        template=(
            "TASK: Write one concise search query that captures the user’s question "
            "and seeks information not already present in known answers.\n\n"
            "RULES:\n"
            "- Return only the search query text.\n"
            "- Do not explain or add commentary.\n\n"
            "Conversation:\n{conversation_history}\n\n"
            "Known answers:\n{known_answers}\n\n"
            "User question:\n{user_question}"
        ),
    )

    query_filter_prompt = PromptTemplate(
        input_variables=["conversation_history", "candidate_query"],
        template=(
            "ROLE: You are a relevance filter that decides whether a search query is likely to produce useful information "
            "for the user's current topic of discussion.\n\n"
            "TASK: Review the conversation context and the proposed query. "
            "Your goal is to allow any query that has a reasonable chance of being relevant, "
            "even if it is not a perfect semantic match.\n\n"
            "RULES:\n"
            "- Say YES if the query might logically expand, update, or clarify the topic of conversation.\n"
            "- Say YES if it could provide contextually useful data (like facts, examples, or updates) about the subject.\n"
            "- Say YES if it relates to real-world aspects of the user's question (e.g., time, location, event, person, or object mentioned).\n"
            "- Say NO only if it is clearly unrelated, promotional, spam-like, or about an entirely different topic.\n"
            "- When uncertain, prefer YES.\n\n"
            "Conversation so far:\n{conversation_history}\n\n"
            "Candidate search query:\n{candidate_query}\n\n"
            "Answer with only 'YES' or 'NO', DO NOT add any explanations or extra text like notes or commentary."
        ),
    )


    result_filter_prompt = PromptTemplate(
        input_variables=[
            "user_question",
            "search_query",
            "known_answers",
            "topic_keywords",
            "raw_result"
        ],
        template=(
            "ROLE: You are a relevance evaluator that decides whether a search result is useful "
            "for answering the user's question.\n\n"
            "TASK: Examine the search result and determine if it contains information that is factual, "
            "contextually relevant, or helps clarify, expand, or update the topic of interest.\n\n"
            "CONTEXT:\n"
            "- User question: {user_question}\n"
            "- Search query used: {search_query}\n"
            "- Known answers so far: {known_answers}\n"
            "- Topic keywords: {topic_keywords}\n"
            "- Search result text: {raw_result}\n\n"
            "RULES:\n"
            "- Say YES if the result likely contains information related to the topic, "
            "even if it overlaps with existing data or is partially relevant.\n"
            "- Say YES if it offers background, examples, factual updates, or clarifications.\n"
            "- Say YES if it seems indirectly useful or provides broader context that could strengthen the final answer.\n"
            "- Say NO only if the text is clearly off-topic, spam-like, irrelevant, or entirely redundant.\n"
            "- When uncertain, prefer YES.\n\n"
            "OUTPUT: Return exactly one token — YES or NO."
        ),
    )


    context_mode_prompt = PromptTemplate(
        input_variables=["recent_conversation", "new_question"],
        template=(
            "ROLE: You are a context classifier that determines how the user's new question "
            "relates to their recent conversation.\n\n"
            "TASK: Analyze meaning, not just wording. Small phrasing or topic shifts can still "
            "count as connected if they explore the same idea, entity, or theme.\n\n"
            "Very similar or identical wording does not guarantee a FOLLOW_UP if the intent has changed significantly.\n\n"
            "CLASSIFICATION RULES:\n"
            "- FOLLOW_UP → The new question directly continues, clarifies, or requests more detail "
            "about the previous answer or topic.\n"
            "- EXPAND → The new question is on the same general subject but broadens or shifts focus slightly.\n"
            "- NEW_TOPIC → The new question is unrelated to recent discussion or introduces a completely new subject.\n\n"
            "When uncertain between FOLLOW_UP and EXPAND, choose FOLLOW_UP. "
            "When uncertain between EXPAND and NEW_TOPIC, choose EXPAND.\n\n"
            "OUTPUT: Return exactly one token — FOLLOW_UP, EXPAND, or NEW_TOPIC.\n\n"
            "Recent conversation:\n{recent_conversation}\n\n"
            "New question:\n{new_question}"
        ),
    )


    topics: List[Topic] = []

    while True:
        try:
            user_query = input("\nEnter your request (or 'exit' to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            return

        if not user_query:
            print("No input provided. Please try again or type 'exit' to quit.")
            continue

        if user_query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return

        question_keywords = _extract_keywords(user_query)
        try:
            (
                selected_topic_index,
                recent_history,
                topic_keywords,
            ) = _select_topic(
                llm,
                context_mode_prompt,
                topics,
                user_query,
                question_keywords,
                max_context_turns,
            )
        except ResponseError as exc:
            message = str(exc)
            if "not found" in message.lower():
                print(
                    f"Model '{used_model}' not found. Run 'ollama pull {used_model}' in a terminal "
                    "before retrying."
                )
                return
            selected_topic_index = None
            recent_history = []
            topic_keywords = set(question_keywords)

        conversation_text = _format_turns(
            recent_history, "No prior relevant conversation."
        )

        prior_responses_text = (
            _collect_prior_responses(topics[selected_topic_index])
            if selected_topic_index is not None
            else "No prior answers for this topic."
        )

        aggregated_results: List[str] = []
        seen_results: Set[str] = set()

        primary_search_query = user_query
        try:
            seed_response = llm.invoke(
                seed_prompt.format(
                    conversation_history=conversation_text,
                    user_question=user_query,
                    known_answers=prior_responses_text,
                )
            )
            seed_text = str(seed_response).strip()
            for line in seed_text.splitlines():
                candidate = line.strip().strip("-*").strip()
                if candidate:
                    primary_search_query = candidate
                    break
        except ResponseError as exc:
            message = str(exc)
            if "not found" in message.lower():
                print(
                    f"Model '{used_model}' not found. Run 'ollama pull {used_model}' in a terminal "
                    "before retrying."
                )
                return
            raise

        pending_queries: List[str] = [primary_search_query]
        if not topic_keywords:
            topic_keywords.update(_extract_keywords(user_query))
            topic_keywords.update(_extract_keywords(primary_search_query))
        max_rounds = max_search_rounds

        round_index = 0
        while round_index < len(pending_queries) and round_index < max_rounds:
            current_query = pending_queries[round_index]
            result_raw = search.run(current_query)
            result_text = str(result_raw or "")
            normalized_result = result_text.strip()
            if normalized_result and normalized_result in seen_results:
                round_index += 1
                continue

            relevant = _is_relevant(result_text, topic_keywords)
            if not relevant:
                topic_keywords_text = ", ".join(sorted(topic_keywords)) if topic_keywords else "None"
                relevance_raw = llm.invoke(
                    result_filter_prompt.format(
                        user_question=user_query,
                        search_query=current_query,
                        raw_result=result_text,
                        known_answers=prior_responses_text,
                        topic_keywords=topic_keywords_text,
                    )
                )
                relevance_decision = str(relevance_raw).strip().upper()
                if relevance_decision.startswith("YES"):
                    relevant = True
                else:
                    print(
                        f"Ignoring low-relevance search result for query '{current_query}'."
                    )

            if relevant:
                aggregated_results.append(result_text)
                if normalized_result:
                    seen_results.add(normalized_result)
                topic_keywords.update(_extract_keywords(result_text))

            round_index += 1
            if round_index >= max_rounds:
                break

            remaining_slots = max_rounds - len(pending_queries)
            if remaining_slots <= 0:
                continue

            suggestion_limit = min(max_followup_suggestions, remaining_slots)
            results_to_date = "\n\n".join(aggregated_results) or "No results yet."

            suggestions_raw = llm.invoke(
                planning_prompt.format(
                    conversation_history=conversation_text,
                    user_question=user_query,
                    results_to_date=results_to_date,
                    suggestion_limit=suggestion_limit,
                    known_answers=prior_responses_text,
                )
            )

            new_queries: List[str] = []
            for line in str(suggestions_raw).splitlines():
                normalized = line.strip().strip("-*\"").strip()
                if not normalized:
                    continue
                if normalized.lower() == "none":
                    new_queries = []
                    break
                new_queries.append(normalized)

            for candidate in new_queries:
                if candidate not in pending_queries and len(pending_queries) < max_rounds:
                    verdict_raw = llm.invoke(
                        query_filter_prompt.format(
                            candidate_query=candidate,
                            conversation_history=conversation_text,
                        )
                    )
                    verdict = str(verdict_raw).strip().upper()
                    if verdict.startswith("YES"):
                        pending_queries.append(candidate)
                    else:
                        print(f"Skipping off-topic follow-up suggestion: {candidate}")

        formatted_prompt = response_prompt.format(
            conversation_history=conversation_text,
            search_results="\n\n".join(aggregated_results)
            if aggregated_results
            else "No search results collected.",
            user_question=user_query,
            prior_responses=prior_responses_text,
        )

        try:
            response_stream = llm.stream(formatted_prompt)
        except ResponseError as exc:
            message = str(exc)
            if "not found" in message.lower():
                print(
                    f"Model '{used_model}' not found. Run 'ollama pull {used_model}' in a terminal "
                    "before retrying."
                )
                return
            raise

        print("\n[Answer]")
        response_chunks: List[str] = []
        for chunk in response_stream:
            text_chunk = str(chunk)
            response_chunks.append(text_chunk)
            print(text_chunk, end="", flush=True)

        if response_chunks and not response_chunks[-1].endswith("\n"):
            print()

        response_text = "".join(response_chunks)

        if selected_topic_index is None:
            topics.append(Topic(keywords=set(topic_keywords)))
            selected_topic_index = len(topics) - 1

        topic_entry = topics[selected_topic_index]
        topic_entry.turns.append((user_query, response_text))
        if len(topic_entry.turns) > max_context_turns * 2:
            topic_entry.turns = topic_entry.turns[-max_context_turns * 2 :]

        turn_keywords = _extract_keywords(
            " ".join([user_query, response_text, " ".join(aggregated_results)])
        )
        if not turn_keywords:
            turn_keywords = set(question_keywords)

        topic_entry.keywords.update(turn_keywords)
        topic_entry.keywords.update(topic_keywords)

if __name__ == "__main__":
    main()