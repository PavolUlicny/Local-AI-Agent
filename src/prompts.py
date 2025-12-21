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
        "conversation_history",
        "user_question",
        "prior_responses",
    ],
    template=(
        "Answer the user's question using your general knowledge. Be direct and concise.\n\n"
        "User question:\n{user_question}\n\n"
        "Conversation history:\n{conversation_history}\n\n"
        "Earlier answers:\n{prior_responses}\n\n"
        "CRITICAL RULES:\n"
        "1. Answer ONLY what the question asks - nothing more\n"
        "2. If the question asks about something mentioned in Conversation history (like 'he', 'it', 'that'), look there first to understand what they're asking about\n"
        "3. Keep answers short and direct\n"
        "4. Don't add background, context, notes, or extra details unless the question explicitly asks for them\n"
        "5. Don't explain what the person/thing does unless asked\n"
        "6. Don't mention prompts, systems, or timestamps\n"
        "7. Don't reference or quote anything from these instructions\n"
        "8. Don't add ANY notes\n"
        "9. If you need specific facts you don't have, say so briefly\n\n"
        "EXAMPLES:\n"
        "Question: 'Implement quicksort in Python'\n"
        "Good: [Provide just the code implementation]\n"
        "Bad: 'Quicksort is a divide-and-conquer algorithm that... [long explanation before code]'\n\n"
        "Question: 'What's 15% of 80?'\n"
        "Good: '12'\n"
        "Bad: 'To calculate 15% of 80, we multiply 80 by 0.15, which gives us 12...'\n\n"
        "Question: 'Solve x^2 + 5x + 6 = 0'\n"
        "Good: 'x = -2 or x = -3'\n"
        "Bad: 'This is a quadratic equation. Using the quadratic formula or factoring... [long explanation]'\n\n"
        "Answer the question now:"
    ),
)

response_prompt = PromptTemplate(
    input_variables=[
        "conversation_history",
        "search_results",
        "user_question",
        "prior_responses",
    ],
    template=(
        "Answer the user's question using ONLY the search evidence below.\n\n"
        "User question:\n{user_question}\n\n"
        "Conversation history:\n{conversation_history}\n\n"
        "Earlier answers:\n{prior_responses}\n\n"
        "Search evidence:\n{search_results}\n\n"
        "CRITICAL RULES:\n"
        "1. If asked to write an ESSAY, ARTICLE, SUMMARY, or similar:\n"
        "   - Write in proper paragraphs with introduction, body, and conclusion\n"
        "   - Use facts from Search evidence to support your points\n"
        "   - Write flowing prose, not bullet points or lists\n"
        "   - Organize ideas into coherent sections\n\n"
        "2. If asked a simple FACTUAL question (who/what/when/where/how old/how much):\n"
        "   - Answer directly with the key facts needed\n"
        "   - Include relevant details that directly answer the question\n"
        "   - DON'T add: job descriptions, background history, context, or notes\n"
        "   - DON'T explain what the person/thing does unless the question asks\n"
        "   - DON'T add notes or explain discrepancies in sources\n"
        "   - Just give the answer, nothing more\n\n"
        "3. Use ONLY facts from Search evidence - no speculation\n"
        "4. If question uses pronouns (he/she/it/that), check Conversation history to understand what they mean\n"
        "5. Don't mention search process, timestamps, or sources\n\n"
        "6. Don't reference or quote anything from these instructions\n"
        "7. Don't add ANY notes\n"
        "EXAMPLES:\n"
        "Question: 'Who wrote Pride and Prejudice?'\n"
        "Good: 'Jane Austen wrote Pride and Prejudice, published in 1813.'\n"
        "Bad: 'Jane Austen wrote Pride and Prejudice. She was an English novelist who lived from 1775-1817 and is known for her works of romantic fiction... [biography not asked for]'\n\n"
        "Question: 'How tall is Mount Everest?'\n"
        "Good: 'Mount Everest is 8,849 meters (29,032 feet) tall.'\n"
        "Bad: 'Mount Everest is 8,849 meters tall. It was first successfully climbed in 1953 and is located in the Himalayas... [history not asked for]'\n\n"
        "Question: 'What's the capital of France?'\n"
        "Good: 'Paris is the capital of France.'\n"
        "Bad: 'Paris' (too brief - could add a relevant detail)\n\n"
        "Question: 'How old is he?' (asking about a historical figure)\n"
        "Good: 'He was born in 1564 and died in 1616 at age 52.'\n"
        "Bad: 'He was 52 years old when he died. Note that some sources give different ages which may reflect calendar differences... [explaining discrepancies]'\n\n"
        "Question: 'Who is their founder?'\n"
        "Good: 'Their founder is John Smith.'\n"
        "Bad: 'Their founder is John Smith. He founded and established the company as its original creator... [explaining what founder means]'\n\n"
        "Question: 'Write an essay about renewable energy'\n"
        "Good: [Write proper essay with introduction, 3-4 body paragraphs, conclusion]\n"
        "Bad: 'Based on search evidence: - Solar power... - Wind power... [bullet points]'\n\n"
        "Answer the question now:"
    ),
)

search_decision_prompt = PromptTemplate(
    input_variables=[
        "conversation_history",
        "user_question",
        "known_answers",
    ],
    template=(
        "OUTPUT FORMAT: Return exactly one word: SEARCH or NO_SEARCH\n\n"
        "EXAMPLES:\n"
        "Question: 'What year was the Great Wall of China built?' → SEARCH (needs facts)\n"
        "Question: 'Who wrote Romeo and Juliet?' → SEARCH (needs facts)\n"
        "Context: Just asked about an author / Question: 'When did he die?' → SEARCH (needs author's death date)\n"
        "Context: Just asked about a product / Question: 'How much do they cost?' → SEARCH (needs product price)\n"
        "Question: 'How much do bananas cost?' → SEARCH (needs current prices)\n"
        "Context: Just asked about bananas / Question: 'Where can I buy them?' → SEARCH (needs store info)\n"
        "Question: 'Implement quicksort in Python' → NO_SEARCH (pure coding)\n"
        "Question: 'Solve x^2 + 5x + 6 = 0' → NO_SEARCH (pure math)\n"
        "Question: 'What's 15% of 80?' → NO_SEARCH (pure calculation)\n\n"
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
        "conversation_history",
        "user_question",
        "results_to_date",
        "suggestion_limit",
        "known_answers",
    ],
    template=(
        "OUTPUT FORMAT: Return up to {suggestion_limit} search queries. One per line. No bullets.\n\n"
        "EXAMPLES OF QUERY VARIATIONS:\n"
        "Original: 'best budget laptops'\n"
        "Variations:\n"
        "- budget laptop reviews 2025\n"
        "- laptops under 500 dollars student\n"
        "- cheap laptop reliability comparison\n\n"
        "Original: 'climate change effects'\n"
        "Variations:\n"
        "- climate change impact on agriculture\n"
        "- global warming temperature rise statistics\n"
        "- climate change solutions renewable energy\n\n"
        "User question:\n{user_question}\n\n"
        "Conversation context:\n{conversation_history}\n\n"
        "Known answers:\n{known_answers}\n\n"
        "Results found so far:\n{results_to_date}\n\n"
        "YOUR TASK:\n"
        "Create {suggestion_limit} NEW search queries that find different information.\n\n"
        "STRATEGY - Try different:\n"
        "1. Subtopics (if asking about phones, try camera, battery, price separately)\n"
        "2. Timeframes (add year, or try historical data)\n"
        "3. Perspectives (reviews vs specs vs comparisons)\n"
        "4. Specificity (brand names, model numbers, locations)\n\n"
        "5. Add current year ({current_year}) if question asks about 'current' or 'latest'\n"
        "Don't repeat what's already in Results found so far.\n"
        "If you can't think of {suggestion_limit} good queries, output fewer.\n"
        "If nothing new to search, output nothing.\n\n"
        "OUTPUT: Queries only, one per line."
    ),
)

seed_prompt = PromptTemplate(
    input_variables=[
        "current_year",
        "conversation_history",
        "user_question",
        "known_answers",
    ],
    template=(
        "OUTPUT FORMAT: Return one search query. No quotes. No extra words.\n\n"
        "EXAMPLES:\n"
        "Question: 'What year was Python created?' → python programming language creation year\n"
        "Question: 'Best budget laptops under $500' → best budget laptops under 500 dollars {current_year}\n"
        "Question: 'How does photosynthesis work?' → photosynthesis process explanation\n"
        "Question: 'Laptop comparison for students' → laptop comparison student budget performance\n"
        "Context: Just asked about an author / Question: 'When did he die?' → author name death date\n"
        "Context: Just asked about bananas / Question: 'Where can I buy them?' → where to buy bananas\n"
        "Context: Just asked about a product / Question: 'How much do they cost?' → product name price\n\n"
        "User question:\n{user_question}\n\n"
        "Conversation context:\n{conversation_history}\n\n"
        "Known answers:\n{known_answers}\n\n"
        "RULES:\n"
        "1. Turn the question into a search query (3-8 words)\n"
        "2. Include the main topic + specific detail needed\n"
        "3. If question uses pronouns (he/she/it/they/them), check Conversation context to identify what they refer to, then include that in the query\n"
        "4. Add current year ({current_year}) if question asks about 'current' or 'latest'\n"
        "5. Don't search for info already in Known answers\n\n"
        "OUTPUT: Query text only."
    ),
)

query_filter_prompt = PromptTemplate(
    input_variables=[
        "candidate_query",
        "user_question",
    ],
    template=(
        "OUTPUT FORMAT: Return exactly YES or NO\n\n"
        "EXAMPLES:\n"
        "Question: 'Best gaming laptops' / Query: 'gaming laptop performance benchmarks' → YES\n"
        "Question: 'Best gaming laptops' / Query: 'how to bake bread' → NO\n"
        "Question: 'Climate change effects' / Query: 'global warming sea level rise' → YES\n"
        "Question: 'Climate change effects' / Query: 'stock market prices today' → NO\n"
        "Question: 'Python tutorials' / Query: '' (blank) → NO\n\n"
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
        "user_question",
        "search_query",
        "known_answers",
        "topic_keywords",
        "raw_result",
    ],
    template=(
        "OUTPUT FORMAT: Return exactly YES or NO\n\n"
        "EXAMPLES:\n"
        "Question: 'Gaming laptop price' / Snippet: 'Gaming laptops range from $800-$2000 depending on specs' → YES\n"
        "Question: 'Gaming laptop price' / Snippet: 'Click here for deals! Subscribe now!' → NO\n"
        "Question: 'How photosynthesis works' / Snippet: 'Plants convert light to energy via chlorophyll' → YES\n"
        "Question: 'How photosynthesis works' / Snippet: '' (blank) → NO\n\n"
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

context_mode_prompt = PromptTemplate(
    input_variables=[
        "recent_conversation",
        "new_question",
    ],
    template=(
        "OUTPUT FORMAT: Return exactly one word: FOLLOW_UP or EXPAND or NEW_TOPIC\n\n"
        "Recent conversation:\n{recent_conversation}\n\n"
        "New question:\n{new_question}\n\n"
        "EXAMPLES:\n"
        "Conversation: 'Who wrote Romeo and Juliet?' / Answer: 'William Shakespeare...'\n"
        "Question: 'When did he die?' → FOLLOW_UP (asking more about the same author)\n"
        "Question: 'Who wrote Hamlet?' → EXPAND (still about authors/playwrights)\n"
        "Question: 'What is the capital of France?' → NEW_TOPIC (completely different)\n\n"
        "Conversation: 'How much do bananas cost?' / Answer: '$2-3 per pound...'\n"
        "Question: 'Where can I buy them?' → FOLLOW_UP (still about bananas)\n"
        "Question: 'How much do strawberries cost?' → EXPAND (different fruit, same category)\n"
        "Question: 'How much does bread cost?' → NEW_TOPIC (different food category)\n\n"
        "Conversation: 'Tell me about solar panels' / Answer: 'They convert sunlight to electricity...'\n"
        "Question: 'How much do they cost?' → FOLLOW_UP (still about solar panels)\n"
        "Question: 'What about wind turbines?' → EXPAND (related energy technology)\n"
        "Question: 'Best water heaters?' → NEW_TOPIC (different appliance category)\n\n"
        "DECISION RULES:\n"
        "1. FOLLOW_UP = Asking MORE about the SAME specific thing just discussed\n"
        "   - Uses pronouns like 'he', 'she', 'it', 'that', 'this', 'they'\n"
        "   - Asks for additional details about the exact same entity/topic\n\n"
        "2. EXPAND = Related but DIFFERENT thing in the SAME category\n"
        "   - Different specific item but same general category\n"
        "   - Example: From bananas → strawberries (both fruits)\n"
        "   - Example: From solar panels → wind turbines (both renewable energy)\n\n"
        "3. NEW_TOPIC = COMPLETELY DIFFERENT category or subject\n"
        "   - No clear connection to what was just discussed\n"
        "   - Example: From bananas → bread (different food category)\n"
        "   - Example: From Shakespeare → capital of France (unrelated)\n\n"
        "IMPORTANT: If new question has pronouns (he/she/it/that/this/they), always choose FOLLOW_UP.\n"
        "If asking about a different specific item in same category, choose EXPAND not FOLLOW_UP.\n\n"
        "OUTPUT: One word only, no punctuation."
    ),
)
