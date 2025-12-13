## Topic & Context Management

- Each topic retains a list of `(user, assistant)` turns.
- On each new query, `_select_topic` attempts to classify relationship (FOLLOW_UP / EXPAND / NEW_TOPIC) via a dedicated chain. If no sufficiently overlapping keywords, a new topic begins.
- A semantic embedding (default: `embeddinggemma:300m`) is blended into each topic so cosine similarity can rescue related follow-ups even when keywords diverge.
- Topics exceeding `MAX_TOPICS` are pruned FIFO.
- Turn history per topic is trimmed to `max_context_turns * 2` preserving a sliding window.
- Keywords aggregated from user query, assistant answer, and truncated search result text; pruned to `MAX_TOPIC_KEYWORDS` by frequency ordering.

Embedding-assisted relevance can bypass the LLM gate when similarity is high; topic briefs and prior responses are used to inform search decisions and final answer synthesis.
