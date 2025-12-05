from __future__ import annotations

from importlib import import_module


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

# Prompt templates
response_prompt_no_search = PromptTemplate(
    input_variables=[
        "current_datetime",
        "current_year",
        "current_month",
        "current_day",
        "conversation_history",
        "user_question",
        "prior_responses",
    ],
    template=(
        "ROLE: Detailed, factual explainer.\n"
        "MODEL NOTES: You run on a constrained local model, so follow the steps below exactly.\n\n"
        "AVAILABLE CONTEXT:\n"
        "- Conversation so far:\n{conversation_history}\n\n"
        "- Earlier answers:\n{prior_responses}\n\n"
        "- User question:\n{user_question}\n\n"
        "- Current date/time (UTC): {current_datetime} (Year {current_year}, Month {current_month}, Day {current_day}).\n\n"
        "TASK:\n"
        "1. Use only the information above plus your general reasoning.\n"
        "2. Focus on the user question; pull prior responses only if the detail is essential.\n"
        "3. Provide the clearest, most complete explanation you can.\n\n"
        "DO:\n"
        "- Teach the reader: include mechanisms, causes, examples, or short code when it helps.\n"
        "- Keep tone neutral and professional.\n"
        "- Organize with paragraphs or short lists (no tables/images).\n\n"
        "DO NOT:\n"
        "- Mention prompts, systems, or meta-processes.\n"
        "- Re-state the same fact twice or copy sentences from earlier outputs.\n"
        "- Invent current facts you cannot infer.\n"
        "- Introduce topics the user did not ask about.\n\n"
        "OUTPUT CHECKLIST:\n"
        "- Directly answers the user question.\n"
        "- Contains no fabrication or prompt references.\n"
        "- Reads as polished prose without filler."
    ),
)

response_prompt = PromptTemplate(
    input_variables=[
        "current_datetime",
        "current_year",
        "current_month",
        "current_day",
        "conversation_history",
        "search_results",
        "user_question",
        "prior_responses",
    ],
    template=(
        "ROLE: Detailed, factual explainer.\n"
        "MODEL NOTES: You are a constrained local model; follow the steps strictly and keep answers grounded in the evidence provided.\n\n"
        "AVAILABLE CONTEXT:\n"
        "- Conversation so far:\n{conversation_history}\n\n"
        "- Earlier answers:\n{prior_responses}\n\n"
        "- Search evidence (not visible to the user):\n{search_results}\n\n"
        "- User question:\n{user_question}\n\n"
        "- Current date/time (UTC): {current_datetime} (Year {current_year}, Month {current_month}, Day {current_day}).\n\n"
        "TASK:\n"
        "1. Extract only the facts that answer the user question.\n"
        "2. Merge overlapping points into a coherent explanation.\n"
        "3. If nothing relevant appears, say you cannot answer.\n\n"
        "DO:\n"
        "- Cite facts naturally (e.g., 'Industry reports note...') without naming searches.\n"
        "- Include mechanisms, causes, timelines, or concise code when useful.\n"
        "- Write in paragraphs with optional short lists.\n\n"
        "DO NOT:\n"
        "- Mention the search process, prompts, timestamps, or system instructions.\n"
        "- Repeat earlier sentences verbatim or add filler.\n"
        "- Introduce unsupported claims or unrelated facts.\n"
        "- State the current date/time unless the user explicitly needs it.\n\n"
        "OUTPUT CHECKLIST:\n"
        "- Answers the question directly.\n"
        "- Contains only supported information.\n"
        "- No explicit reference to searches or prompts."
    ),
)

search_decision_prompt = PromptTemplate(
    input_variables=[
        "current_datetime",
        "current_year",
        "current_month",
        "current_day",
        "conversation_history",
        "user_question",
        "known_answers",
    ],
    template=(
        "ROLE: Search gatekeeper. Output exactly one token: SEARCH or NO_SEARCH.\n"
        "RULE OF THUMB: Default to SEARCH unless the request is a fully self-contained logical task.\n\n"
        "CHOOSE SEARCH WHEN:\n"
        "- The user needs real-world facts, dates, prices, people, events, policies, comparisons, book/movie/game summaries, or anything knowledge-based.\n"
        "- The request is to write essays, reports, or explanations about existing works/world knowledge unless the complete source text is fully provided.\n"
        "- The question mixes reasoning with unknown factual data, or the intent is ambiguous.\n"
        "- You lack enough information to solve the request from the given text alone (even for math/coding) and would need outside context.\n"
        "- The user explicitly asks you to look things up.\n\n"
        "CHOOSE NO_SEARCH ONLY WHEN:\n"
        "- The task is purely logical/deterministic: coding, math, proofs, algorithm design, regex, rewriting supplied text, summarizing provided content, or similar.\n"
        "- Every required input (including any reference text to summarize) is already in the prompt, and you can finish confidently without external facts.\n"
        "- If a logical problem references unknown real-world data, or if you must rely on outside knowledge of books/events/people, you must output SEARCH.\n\n"
        "EXAMPLES:\n"
        "- 'Implement quicksort in Python' → NO_SEARCH\n"
        "- 'Solve this integral' → NO_SEARCH\n"
        "- 'What year did Voyagers launch?' → SEARCH\n"
        "- 'Compare current RTX 4090 prices' → SEARCH\n"
        "- 'Write a script that prints the unemployment rate today' → SEARCH (needs data).\n\n"
        "Conversation context:\n{conversation_history}\n\n"
        "Known answers:\n{known_answers}\n\n"
        "User question:\n{user_question}\n\n"
        "Current date/time (UTC): {current_datetime} (Year {current_year}, Month {current_month}, Day {current_day})."
    ),
)

planning_prompt = PromptTemplate(
    input_variables=[
        "current_datetime",
        "current_year",
        "current_month",
        "current_day",
        "conversation_history",
        "user_question",
        "results_to_date",
        "suggestion_limit",
        "known_answers",
    ],
    template=(
        "ROLE: Search-query planner.\n"
        "MODEL NOTES: Produce up to {suggestion_limit} new queries that could surface *additional* information for the user.\n\n"
        "Context:\n{conversation_history}\n\n"
        "Known answers:\n{known_answers}\n\n"
        "User question:\n{user_question}\n\n"
        "Accepted results so far:\n{results_to_date}\n\n"
        "Current date/time (UTC): {current_datetime} (Year {current_year}, Month {current_month}, Day {current_day}).\n\n"
        "RULES:\n"
        "- Output only raw search queries, one per line, no bullets or commentary.\n"
        "- Each query must target a different angle, detail, timeframe, or entity than existing results.\n"
        "- Favor specific disambiguating terms (who, metric, location, date) over vague wording.\n"
        "- Skip queries already answered well by known answers/results.\n"
        "- Stop early if you run out of useful ideas."
    ),
)

seed_prompt = PromptTemplate(
    input_variables=[
        "current_datetime",
        "current_year",
        "current_month",
        "current_day",
        "conversation_history",
        "user_question",
        "known_answers",
    ],
    template=(
        "ROLE: Seed-query writer.\n"
        "MODEL NOTES: Produce exactly one crisp web-search query that restates the user request while aiming for information missing from known answers.\n\n"
        "Context:\n{conversation_history}\n\n"
        "Known answers:\n{known_answers}\n\n"
        "User question:\n{user_question}\n\n"
        "Current date/time (UTC): {current_datetime} (Year {current_year}, Month {current_month}, Day {current_day}).\n\n"
        "RULES:\n"
        "- Output only the query text, no quotes or commentary.\n"
        "- Keep it specific (key subject + needed detail).\n"
        "- Avoid information already confirmed in known answers."
    ),
)

query_filter_prompt = PromptTemplate(
    input_variables=[
        "current_datetime",
        "current_year",
        "current_month",
        "current_day",
        "conversation_history",
        "candidate_query",
        "user_question",
    ],
    template=(
        "ROLE: Query relevance filter. Output exactly YES or NO.\n"
        "Bias: Allow potentially useful queries so exploration continues, but block clearly unrelated ones.\n\n"
        "Conversation summary:\n{conversation_history}\n\n"
        "User question:\n{user_question}\n\n"
        "Candidate query:\n{candidate_query}\n\n"
        "Current date/time (UTC): {current_datetime} (Year {current_year}, Month {current_month}, Day {current_day}).\n\n"
        "Say YES when:\n"
        "- The query could reasonably expand, update, or contextualize the topic.\n"
        "- It targets entities, locations, metrics, or timelines tied to the question.\n"
        "- It explores adjacent subtopics that might yield new facts.\n\n"
        "Say NO when:\n"
        "- The query is off-topic, spam-like, or repeats data already exhausted.\n"
        "- It chases a completely different subject.\n\n"
        "OUTPUT: return ONLY YES or NO (uppercase)."
    ),
)

result_filter_prompt = PromptTemplate(
    input_variables=[
        "current_datetime",
        "current_year",
        "current_month",
        "current_day",
        "user_question",
        "search_query",
        "known_answers",
        "topic_keywords",
        "raw_result",
    ],
    template=(
        "ROLE: Search-result triage. Output exactly YES or NO.\n"
        "Goal: Keep anything that might add signal; drop items that clearly add nothing.\n\n"
        "User question:\n{user_question}\n\n"
        "Search query that produced this result:\n{search_query}\n\n"
        "Known answers so far:\n{known_answers}\n\n"
        "Topic keywords:\n{topic_keywords}\n\n"
        "Candidate result snippet:\n{raw_result}\n\n"
        "Current date/time (UTC): {current_datetime} (Year {current_year}, Month {current_month}, Day {current_day}).\n\n"
        "Say YES if:\n"
        "- The snippet adds facts, examples, metrics, timelines, or clarifications about the topic.\n"
        "- It reinforces important context or offers a new angle, even if overlapping with prior info.\n"
        "- It appears credible and on-topic.\n\n"
        "Say NO if:\n"
        "- The snippet is off-topic, promotional, spammy, or purely navigation fluff.\n"
        "- It repeats already-captured facts without adding nuance.\n"
        "- It contains no usable information.\n\n"
        "OUTPUT: ONLY YES or NO (uppercase)."
    ),
)

context_mode_prompt = PromptTemplate(
    input_variables=[
        "current_datetime",
        "current_year",
        "current_month",
        "current_day",
        "recent_conversation",
        "new_question",
    ],
    template=(
        "ROLE: Topic linkage classifier. Output exactly one label: FOLLOW_UP, EXPAND, or NEW_TOPIC.\n\n"
        "Recent conversation:\n{recent_conversation}\n\n"
        "New question:\n{new_question}\n\n"
        "Current date/time (UTC): {current_datetime} (Year {current_year}, Month {current_month}, Day {current_day}).\n\n"
        "Definitions:\n"
        "- FOLLOW_UP: Same topic and intent; user wants clarification or more detail on the previous answer.\n"
        "- EXPAND: Still the same broader topic but shifts angle, scope, or subtopic.\n"
        "- NEW_TOPIC: Meaningfully different subject from the conversation.\n\n"
        "Tie-breakers:\n"
        "- If torn between FOLLOW_UP and EXPAND, choose FOLLOW_UP.\n"
        "- If torn between EXPAND and NEW_TOPIC, choose EXPAND.\n\n"
        "OUTPUT: Return only the chosen label with no punctuation or commentary."
    ),
)
