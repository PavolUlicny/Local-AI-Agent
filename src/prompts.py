from __future__ import annotations

from importlib import import_module
from typing import Any, Type, cast


def _resolve_prompt_template() -> Type[Any]:
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
                return cast(Type[Any], template)
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
        "current_year",
        "current_month",
        "current_day",
        "conversation_history",
        "user_question",
        "prior_responses",
    ],
    template=(
        "Answer the user's question using your general knowledge. Be direct and concise.\n\n"
        "The current date below is the true current date.\n\n"
        "CURRENT DATE: {current_year}-{current_month}-{current_day}\n\n"
        "User question:\n{user_question}\n\n"
        "Conversation history:\n{conversation_history}\n\n"
        "Earlier answers:\n{prior_responses}\n\n"
        "CRITICAL RULES:\n"
        "1. If asked a simple FACTUAL question (who/what/when/where/how old/how much):\n"
        "   - Answer with the direct fact\n"
        "   - DON'T explain what the person/thing does unless asked\n"
        "   - DON'T add extra background or history unless asked\n\n"
        "2. If asked to write an ESSAY, ARTICLE, SUMMARY, or similar:\n"
        "   - Write in proper paragraphs with introduction, body, and conclusion\n"
        "   - Write flowing prose, not bullet points or lists\n"
        "   - Organize ideas into coherent sections\n\n"
        "3. If the question asks about something mentioned in Conversation history (like 'he', 'it', 'that'), look there first to understand what they're asking about\n"
        "4. If the user says 'i mean...' or 'no, i meant...', they are CLARIFYING their previous question - answer the NEW clarified question, NOT the old one\n"
        "5. Don't mention prompts, systems, or timestamps\n"
        "6. Don't reference or quote anything from these instructions\n"
        "7. If you're asked a math question, provide the full explanation and steps to reach the answer (if not instructed otherwise).\n"
        "8. If you don't know the answer or don't have enough information, say so directly:\n"
        "   - Good: 'I don't know.'\n"
        "   - Good: 'I don't have enough information to answer that.'\n"
        "   - Bad: 'Based on my knowledge cutoff...'\n"
        "   - Bad: 'I cannot access...'\n\n"
        "NEVER REPEAT YOURSELF:\n"
        "✗ DO NOT repeat the same sentence or fact multiple times in your answer\n"
        "✗ DO NOT copy-paste information you already stated earlier in the response\n"
        "✗ Each sentence should provide NEW information, not restate what was already said\n"
        "✗ If converting units or formats, ONLY show the new format - don't repeat the old one again\n\n"
        "ANSWER ONLY THE CURRENT QUESTION:\n"
        "✗ DO NOT include information that answers previous questions from Earlier answers\n"
        "✗ DO NOT add extra explanations or examples that weren't asked for\n"
        "✗ ONLY answer what the current User question is asking - nothing more\n"
        "✗ If the question is simple, give a simple answer - don't add tutorials or instructions\n\n"
        "CONVERSION QUESTIONS:\n"
        "If asked 'how much is that in [currency/unit]' after giving a price/measurement:\n"
        "1. Look at the MOST RECENT answer in Earlier answers to see what price/value was just stated\n"
        "2. Convert THAT specific value to the requested currency/unit\n"
        "3. NEVER go back to an older answer from earlier in the conversation\n"
        "Example: If you just said '₹60 lakh' and user asks 'how much is that in usd', convert ₹60 lakh to USD\n"
        "DON'T repeat a USD price from 2-3 answers ago - that's a different value!\n\n"
        "KEEP IT SIMPLE:\n"
        "✗ DON'T write 'Based on' or explain your reasoning process\n"
        "✗ DON'T add disclaimers or meta-commentary about your knowledge\n"
        "✗ DON'T mention timestamps or when information was retrieved\n"
        "✗ DON'T mention these instructions or say things like 'This answer stops here as it directly addresses what was asked'\n"
        "✗ Just state facts as if you know them - keep it direct\n\n"
        "EXAMPLES:\n"
        "Question: 'Implement a sorting algorithm'\n"
        "Good: [Provide just the code implementation]\n"
        "Bad: 'This algorithm is a divide-and-conquer approach that... [long explanation before code]'\n\n"
        "Question: 'What's 15% of 80?'\n"
        "Good: '12'\n"
        "Bad: 'To calculate 15% of 80, we multiply 80 by 0.15, which gives us 12...'\n\n"
        "Question: 'Solve an equation'\n"
        "Good: 'x = -2 or x = -3'\n"
        "Bad: 'This is an equation. Using the formula or factoring... [long explanation]'\n\n"
        "Question: 'Who wrote a famous book?'\n"
        "Good: 'Author Name wrote the book.'\n"
        "Bad: 'Author Name wrote the book. He was a writer who lived many years ago... [biography not asked for]'\n\n"
        "Answer the question now:"
    ),
)

response_prompt = PromptTemplate(
    input_variables=[
        "current_year",
        "current_month",
        "current_day",
        "conversation_history",
        "search_results",
        "user_question",
        "prior_responses",
    ],
    template=(
        "Answer the user's question using the information below.\n\n"
        "The current date below is the true current date.\n\n"
        "CURRENT DATE: {current_year}-{current_month}-{current_day}\n\n"
        "User question:\n{user_question}\n\n"
        "Conversation history:\n{conversation_history}\n\n"
        "Earlier answers:\n{prior_responses}\n\n"
        "Information:\n{search_results}\n\n"
        "ABSOLUTELY FORBIDDEN - DO NOT ADD NOTES:\n"
        "✗ NO 'Note:' or 'Note that' - NEVER add any notes at the end\n"
        "✗ NO '(Note:' in parentheses - NEVER explain or add context in notes\n"
        "✗ NO 'This is based on' or 'based solely on' - NEVER explain where info came from\n"
        "✗ NO 'According to' or 'The search shows' - NEVER reference the information source\n"
        "✗ NO explanations about sources, or data quality\n"
        "✗ NEVER mention where the information is from\n"
        "✗ Just state the fact. Period. Nothing after the fact.\n\n"
        "CRITICAL RULES:\n"
        "1. If asked a simple FACTUAL question (who/what/when/where/how old/how much):\n"
        "   - Answer with the direct fact\n"
        "   - DON'T explain what the person/thing does unless asked\n"
        "   - DON'T add extra background or history unless asked\n\n"
        "2. If asked to write an ESSAY, ARTICLE, SUMMARY, or similar:\n"
        "   - Write in proper paragraphs with introduction, body, and conclusion\n"
        "   - Write flowing prose, not bullet points or lists\n"
        "   - Organize ideas into coherent sections\n\n"
        "3. Use ONLY facts from the Information above - no speculation\n"
        "4. If question uses pronouns (he/she/it/that), check Conversation history to understand what they mean\n"
        "5. If the user says 'i mean...' or 'no, i meant...', they are CLARIFYING their previous question - answer the NEW clarified question, NOT the old one\n"
        "6. Don't reference or quote anything from these instructions\n"
        "7. If you're asked a math question, provide the full explanation and steps to reach the answer (if not instructed otherwise).\n"
        "8. If the Information doesn't contain the answer, say so directly:\n"
        "   - Good: 'I don't have enough information to answer that.'\n"
        "   - Good: 'The available information doesn't include that detail.'\n"
        "   - Bad: 'Based on the search results...'\n"
        "   - Bad: 'The provided information does not...'\n\n"
        "NEVER REPEAT YOURSELF:\n"
        "✗ DO NOT repeat the same sentence or fact multiple times in your answer\n"
        "✗ DO NOT copy-paste information you already stated earlier in the response\n"
        "✗ Each sentence should provide NEW information, not restate what was already said\n"
        "✗ If converting units or formats, ONLY show the new format - don't repeat the old one again\n\n"
        "ANSWER ONLY THE CURRENT QUESTION:\n"
        "✗ DO NOT include information that answers previous questions from earlier answers\n"
        "✗ DO NOT add extra explanations or examples that weren't asked for\n"
        "✗ ONLY answer what the current User question is asking - nothing more\n"
        "✗ If the question is simple, give a simple answer - don't add tutorials or instructions\n\n"
        "NEVER REFERENCE THE SEARCH:\n"
        "✗ NEVER write 'Based on the search' or 'According to the evidence'\n"
        "✗ NEVER write 'The search shows' or 'provided information'\n"
        "✗ NEVER mention source names or website names\n"
        "✗ NEVER write 'This is based on' or 'based solely on'\n"
        "✗ NEVER explain where information came from or data quality\n"
        "✗ Just state facts as if you know them - don't explain the search process\n\n"
        "EXAMPLES:\n"
        "Question: 'How much does a product cost?'\n"
        "Good: 'The product costs $X per unit.'\n"
        "Bad: 'As of [date], the product costs $X per unit.\n\nNote: This is based solely on information from [source] regarding product prices over time. The price may fluctuate depending on various market factors...'\n\n"
        "Question: 'Who wrote a famous book?'\n"
        "Good: 'Author Name wrote the book.'\n"
        "Bad: 'Author Name wrote the book. They were a writer who lived from [dates]... [biography not asked for]'\n"
        "Bad: 'Based on the search results, Author Name wrote the book.'\n\n"
        "Question: 'How tall is a mountain?'\n"
        "Good: 'The mountain is X meters (Y feet) tall.'\n"
        "Bad: 'The mountain is X meters tall. It was first climbed in [year]... [history not asked for]'\n"
        "Bad: 'According to the search evidence, the mountain is X meters tall.'\n\n"
        "Question: 'Who is the leader?'\n"
        "Good: 'Person Name is the Leader.'\n"
        "Bad: 'Person Name is the Leader. (Note: Based only on the provided search evidence)'\n"
        "Bad: 'The search shows that Person Name is currently serving as Leader.'\n\n"
        "Question: 'How old is the person?' (asking about someone mentioned earlier)\n"
        "Good: 'They were born in [year] and died in [year] at age X.'\n"
        "Bad: 'They were X years old when they died. Note that some sources give different ages...'\n\n"
        "Question: 'Write an essay about a topic'\n"
        "Good: [Write proper essay with introduction, 3-4 body paragraphs, conclusion]\n"
        "Bad: 'Based on search evidence: - Point 1... - Point 2... [bullet points]'\n\n"
        "Answer the question now:"
    ),
)

search_decision_prompt = PromptTemplate(
    input_variables=[
        "current_year",
        "current_month",
        "current_day",
        "conversation_history",
        "user_question",
        "known_answers",
    ],
    template=(
        "YOUR JOB: Decide if the user's question needs a web search to answer, or if it can be answered without searching.\n\n"
        "OUTPUT FORMAT: Return exactly one word: SEARCH or NO_SEARCH\n\n"
        "The current date below is the true current date.\n\n"
        "CURRENT DATE: {current_year}-{current_month}-{current_day}\n\n"
        "EXAMPLES:\n"
        "Question: 'What year was a historical structure built?' → SEARCH (needs facts)\n"
        "Question: 'Who wrote a famous play?' → SEARCH (needs facts)\n"
        "Context: Just asked about an author / Question: 'When did he die?' → SEARCH (needs author's death date)\n"
        "Context: Just asked about a product / Question: 'How much do they cost?' → SEARCH (needs product price)\n"
        "Question: 'How much do fruits cost?' → SEARCH (needs current prices)\n"
        "Context: Just asked about fruits / Question: 'Where can I buy them?' → SEARCH (needs store info)\n"
        "Question: 'Implement a sorting algorithm' → NO_SEARCH (pure coding)\n"
        "Question: 'Solve an equation' → NO_SEARCH (pure math)\n"
        "Question: 'Calculate a percentage' → NO_SEARCH (pure calculation)\n\n"
        "User question:\n{user_question}\n\n"
        "Conversation context:\n{conversation_history}\n\n"
        "Known answers:\n{known_answers}\n\n"
        "DECISION CHECKLIST:\n"
        "□ Does the question need real-world facts not in Known answers? → SEARCH\n"
        "□ Does it ask about someone/something using pronouns (he/she/it/they/them/their/his/her)? Check Conversation context to see what they mean, then → SEARCH for that info\n"
        "□ Does it need current prices, dates, statistics, or news? → SEARCH\n"
        "□ Is it purely math, coding, logic, or text rewriting? → NO_SEARCH\n"
        "□ Is ALL needed information already in Known answers above? → NO_SEARCH\n\n"
        "IMPORTANT: Pronouns (he/she/it/they/them/their) always need SEARCH for actual facts.\n"
        "Example: 'How old is he?' → SEARCH (need person's age)\n"
        "Example: 'How much do their shares cost?' → SEARCH (need share price)\n\n"
        "When in doubt, choose SEARCH.\n\n"
        "OUTPUT: One word only."
    ),
)

planning_prompt = PromptTemplate(
    input_variables=[
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
        "YOUR JOB: Create search queries to find the EXACT information needed to answer the user's question. Focus ONLY on what the question directly asks for - nothing more.\n\n"
        "OUTPUT FORMAT: Return up to {suggestion_limit} search queries. One per line.\n"
        "- NO bullet points (-, *, •)\n"
        "- NO numbering (1., 2., 3.)\n"
        "- NO quotes (\" or ')\n"
        "- NO explanations or notes\n"
        "- Just plain search queries, nothing else\n\n"
        "The current date below is the true current date.\n\n"
        "CURRENT DATE: {current_year}-{current_month}-{current_day}\n\n"
        "EXAMPLES OF QUERY VARIATIONS:\n"
        "Original: 'best budget products'\n"
        "Variations:\n"
        "budget product reviews {current_year}\n"
        "products under X dollars user needs {current_year}\n"
        "cheap product reliability comparison {current_year}\n\n"
        "Original: 'topic effects'\n"
        "Variations:\n"
        "topic impact on category {current_year}\n"
        "topic statistics data {current_year}\n"
        "topic solutions approaches {current_year}\n\n"
        "User question:\n{user_question}\n\n"
        "Conversation context:\n{conversation_history}\n\n"
        "Known answers:\n{known_answers}\n\n"
        "Results found so far:\n{results_to_date}\n\n"
        "YOUR TASK:\n"
        "FIRST: Carefully check if Results found so far + Known answers contain COMPLETE information to FULLY answer the EXACT question asked.\n\n"
        "CRITICAL DECISION RULES:\n"
        "- Output NONE ONLY if you have ALL information needed to answer the EXACT question completely\n"
        "- If the question asks for X but you only found Y (related but different), you need MORE searches\n"
        "- If you have partial information but missing key details, you need MORE searches\n"
        "- If you're unsure whether you can fully answer, you need MORE searches\n\n"
        "EXAMPLES:\n"
        "Question: 'What is the current weather in City X?' / Results: 'Air quality in City X is poor'\n"
        "→ Need MORE searches (air quality ≠ weather; missing temperature, conditions, etc.)\n\n"
        "Question: 'How much does Product X cost?' / Results: 'Product X features: ...'\n"
        "→ Need MORE searches (features ≠ price)\n\n"
        "Question: 'Who is the CEO of Company X?' / Results: 'Company X was founded several years ago...'\n"
        "→ Need MORE searches (history ≠ current CEO)\n\n"
        "ONLY suggest queries if you genuinely need MORE information to answer the EXACT question.\n"
        "Do NOT suggest queries just because you can - only when NECESSARY to complete the answer.\n\n"
        "When creating queries:\n"
        "1. Focus ONLY on MISSING information that directly answers the question\n"
        "2. Do NOT search for related topics, context, or background information\n"
        "3. If question asks for price, search ONLY for price - not features or reviews\n"
        "4. If question asks for weather, search ONLY for weather - not air quality or climate\n"
        "5. ALWAYS add current year ({current_year}) to get latest information\n"
        "6. Only skip year for historical facts (when something was created, birth/death dates)\n\n"
        "Don't repeat what's already in Results found so far.\n"
        "Create up to {suggestion_limit} queries, but output fewer if that's all you need.\n"
        "If question is already answerable or nothing new to search, output NONE.\n\n"
        "OUTPUT: Plain queries only, one per line. No bullets, no numbers, no quotes, no notes."
    ),
)

seed_prompt = PromptTemplate(
    input_variables=[
        "current_year",
        "current_month",
        "current_day",
        "conversation_history",
        "user_question",
        "known_answers",
    ],
    template=(
        "YOUR JOB: Convert the user's question into a search query that will find the information they need.\n\n"
        "OUTPUT FORMAT: Return one search query. No quotes. No extra words.\n\n"
        "The current date below is the true current date.\n\n"
        "CURRENT DATE: {current_year}-{current_month}-{current_day}\n\n"
        "EXAMPLES:\n"
        "Question: 'What year was a language created?' → programming language creation year\n"
        "Question: 'Best budget products under $X' → best budget products under X dollars {current_year}\n"
        "Question: 'How does a process work?' → process name explanation {current_year}\n"
        "Question: 'Product comparison for users' → product comparison user needs performance {current_year}\n"
        "Question: 'Who is the leader?' → current leader {current_year}\n"
        "Question: 'How much do products cost?' → product price {current_year}\n"
        "Context: Just asked about an author / Question: 'When did he die?' → author name death date\n"
        "Context: Just asked about fruits / Question: 'Where can I buy them?' → where to buy fruits {current_year}\n"
        "Context: Just asked about a product / Question: 'How much do they cost?' → product name price {current_year}\n"
        "Context: Just asked about a person / Question: 'Does he have a partner?' → person name wife spouse married\n"
        "Context: Just asked about a person / Question: 'Is she married?' → person name husband spouse married\n\n"
        "User question:\n{user_question}\n\n"
        "Conversation context:\n{conversation_history}\n\n"
        "Known answers:\n{known_answers}\n\n"
        "RULES:\n"
        "1. Turn the question into a search query (3-8 words)\n"
        "2. Include the main topic + specific detail needed\n"
        "3. If question uses pronouns (he/she/it/they/them), check Conversation context to identify what they refer to, then include that in the query\n"
        "4. ALWAYS add current year ({current_year}) to get latest/current information\n"
        "5. Only skip the year for historical facts (birth/death dates, when something was created)\n"
        "6. Don't search for info already in Known answers\n\n"
        "OUTPUT: Query text only."
    ),
)

query_filter_prompt = PromptTemplate(
    input_variables=[
        "current_year",
        "current_month",
        "current_day",
        "candidate_query",
        "user_question",
    ],
    template=(
        "YOUR JOB: Decide if a search query is relevant to the user's question.\n\n"
        "OUTPUT FORMAT: Return exactly YES or NO\n\n"
        "The current date below is the true current date.\n\n"
        "CURRENT DATE: {current_year}-{current_month}-{current_day}\n\n"
        "EXAMPLES:\n"
        "Question: 'Best products for gaming' / Query: 'gaming product performance benchmarks' → YES\n"
        "Question: 'Best products for gaming' / Query: 'how to cook food' → NO\n"
        "Question: 'Environmental effects' / Query: 'environmental topic effects' → YES\n"
        "Question: 'Environmental effects' / Query: 'market prices today' → NO\n"
        "Question: 'Programming tutorials' / Query: '' (blank) → NO\n\n"
        "User question:\n{user_question}\n\n"
        "Candidate query:\n{candidate_query}\n\n"
        "IS THIS QUERY RELEVANT?\n"
        "Check these:\n"
        "□ Query is about the same topic as the question\n"
        "□ Query could find useful information to answer the question\n"
        "□ Query is not blank or gibberish\n\n"
        "If all 3 are true → YES\n"
        "If any are false → NO\n\n"
        "When in doubt, choose YES.\n\n"
        "OUTPUT: YES or NO only."
    ),
)

result_filter_prompt = PromptTemplate(
    input_variables=[
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
        "YOUR JOB: Decide if a search result snippet contains useful information to answer the user's question.\n\n"
        "OUTPUT FORMAT: Return exactly YES or NO\n\n"
        "The current date below is the true current date.\n\n"
        "CURRENT DATE: {current_year}-{current_month}-{current_day}\n\n"
        "EXAMPLES:\n"
        "Question: 'Product price' / Snippet: 'Products range from $X-$Y depending on features' → YES\n"
        "Question: 'Product price' / Snippet: 'Click here for deals! Subscribe now!' → NO\n"
        "Question: 'How a process works' / Snippet: 'The process converts input to output via mechanism' → YES\n"
        "Question: 'How a process works' / Snippet: '' (blank) → NO\n\n"
        "User question:\n{user_question}\n\n"
        "Search query:\n{search_query}\n\n"
        "Topic keywords:\n{topic_keywords}\n\n"
        "Result snippet:\n{raw_result}\n\n"
        "Known answers:\n{known_answers}\n\n"
        "IS THIS SNIPPET USEFUL?\n"
        "Check these:\n"
        "□ Snippet contains facts, numbers, or explanations about the topic\n"
        "□ Snippet is not spam, ads, or navigation links\n"
        "□ Snippet is not blank or gibberish\n"
        "□ Snippet adds something (can overlap with Known answers if it adds detail)\n\n"
        "If all 4 are true → YES\n"
        "If any are false → NO\n\n"
        "When in doubt, choose YES.\n\n"
        "OUTPUT: YES or NO only."
    ),
)
