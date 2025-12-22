# Implementation Plan: Question Expansion/Context Resolution Prompt

## Overview
Add a new prompt that runs FIRST in the query pipeline to resolve pronouns, references, and context from conversation history into a clear, standalone question. This expanded question will be used throughout the system instead of the original user query.

---

## Phase 1: Create the Prompt

### File: `src/prompts.py`

**Add new prompt after `context_mode_prompt` (around line 457):**

```python
question_expansion_prompt = PromptTemplate(
    input_variables=[
        "current_year",
        "current_month",
        "current_day",
        "conversation_history",
        "user_question",
        "known_answers",
    ],
    template=(
        "YOUR JOB: Expand the user's question into a clear, standalone question that includes all context from the conversation.\n\n"
        "OUTPUT FORMAT: Return ONLY the expanded question. No explanations, no notes, no quotes.\n\n"
        "CURRENT DATE: {current_year}-{current_month}-{current_day}\n\n"
        "EXAMPLES:\n"
        "Conversation: 'Who is the current US president?' / Answer: 'Donald Trump'\n"
        "User question: 'Who is his VP?'\n"
        "→ Who is Donald Trump's vice president?\n\n"
        "Conversation: 'How much do bananas cost?' / Answer: '$2 per pound'\n"
        "User question: 'Where can I buy them?'\n"
        "→ Where can I buy bananas?\n\n"
        "Conversation: 'Tell me about Tesla Model 3' / Answer: 'Electric sedan...'\n"
        "User question: 'How much does it cost?'\n"
        "→ How much does Tesla Model 3 cost?\n\n"
        "Conversation: Empty\n"
        "User question: 'What is the weather in Paris?'\n"
        "→ What is the weather in Paris?\n\n"
        "User question:\n{user_question}\n\n"
        "Conversation history:\n{conversation_history}\n\n"
        "Known answers:\n{known_answers}\n\n"
        "RULES:\n"
        "1. If question has pronouns (he/she/it/they/them/that/this), replace them with the actual entity from conversation\n"
        "2. If question has implicit references ('them', 'there'), make them explicit\n"
        "3. If question is already standalone with no pronouns/references, return it as-is\n"
        "4. Keep the question natural and conversational\n"
        "5. Do NOT add extra information not in the original question\n"
        "6. Do NOT change the intent or scope of the question\n\n"
        "OUTPUT: The expanded question only, nothing else."
    ),
)
```

---

## Phase 2: Add Chain

### File: `src/chains.py`

**Modify `build_chains()` function (line 47):**

```python
def build_chains(llm_robot: OllamaLLM, llm_assistant: OllamaLLM) -> Dict[str, Any]:
    return {
        "question_expansion": _prompts.question_expansion_prompt | llm_robot | StrOutputParser(),  # NEW
        "context": _prompts.context_mode_prompt | llm_robot | StrOutputParser(),
        "seed": _prompts.seed_prompt | llm_robot | StrOutputParser(),
        "planning": _prompts.planning_prompt | llm_robot | StrOutputParser(),
        "result_filter": _prompts.result_filter_prompt | llm_robot | StrOutputParser(),
        "query_filter": _prompts.query_filter_prompt | llm_robot | StrOutputParser(),
        "search_decision": _prompts.search_decision_prompt | llm_robot | StrOutputParser(),
        "response": _prompts.response_prompt | llm_assistant | StrOutputParser(),
        "response_no_search": _prompts.response_prompt_no_search | llm_assistant | StrOutputParser(),
    }
```

---

## Phase 3: Add Expansion Method to Agent

### File: `src/agent.py`

**Add new method after `_generate_search_seed()` (around line 252):**

```python
def _expand_question(self, ctx: "QueryContextType", user_query: str, prior_responses_text: str) -> str:
    """Expand user question to resolve pronouns and references from conversation history.

    Returns the expanded question, or the original if expansion fails.
    """
    from .agent_utils import invoke_chain_safe

    inputs = {
        "user_question": user_query,
        "conversation_history": ctx.conversation_text,
        "known_answers": prior_responses_text,
        "current_year": ctx.current_year,
        "current_month": ctx.current_month,
        "current_day": ctx.current_day,
    }

    try:
        success, expanded = invoke_chain_safe(
            self.chains["question_expansion"],
            inputs,
            self.rebuild_counts,
            "question_expansion",  # Add to rebuild_counts dict
            cfg=self.cfg,
            agent=self,
        )

        if success and expanded:
            expanded_clean = expanded.strip()
            if expanded_clean:
                logging.debug(f"Question expanded: '{user_query}' → '{expanded_clean}'")
                return expanded_clean

        logging.debug(f"Question expansion failed or empty, using original: '{user_query}'")
        return user_query

    except Exception as exc:
        logging.warning(f"Question expansion error: {exc}, using original question")
        return user_query
```

**Add to `__init__` rebuild_counts dict (line 89):**

```python
self.rebuild_counts = {
    "question_expansion": 0,  # NEW
    "search_decision": 0,
    "seed": 0,
    "relevance": 0,
    "planning": 0,
    "query_filter": 0,
    "answer": 0,
}
```

---

## Phase 4: Integrate into Query Pipeline

### File: `src/agent.py`

**Modify `_handle_query_core()` method (line 483):**

**BEFORE Phase 2 (Search Decision), add Phase 1.5:**

```python
def _handle_query_core(self, user_query: str, one_shot: bool) -> str | None:
    # Phase 1: Context & State Initialization
    self._clear_error()
    self._reset_rebuild_counts()
    ctx = build_query_context(self, user_query)
    cfg = self.cfg
    current_datetime = ctx.current_datetime
    current_year = ctx.current_year
    current_month = ctx.current_month
    current_day = ctx.current_day
    question_keywords = ctx.question_keywords
    question_embedding = ctx.question_embedding
    selected_topic_index = ctx.selected_topic_index
    topic_keywords = ctx.topic_keywords
    topic_embedding_current = ctx.topic_embedding_current
    conversation_text = ctx.conversation_text
    prior_responses_text = ctx.prior_responses_text

    aggregated_results: List[str] = []

    # Phase 1.5: Question Expansion (NEW)
    # Expand the question to resolve pronouns/references from conversation
    # Keep original user_query for final response, use expanded_question for search/logic
    expanded_question = self._expand_question(ctx, user_query, prior_responses_text)

    # Phase 2: Search Decision
    # USE expanded_question instead of user_query
    should_search = bool(cfg.force_search)
    if not should_search and cfg.auto_search_decision:
        try:
            should_search = self._decide_should_search(ctx, expanded_question, prior_responses_text)  # CHANGED
        except _exceptions.ResponseError as exc:
            # ... error handling ...
```

**Continue modifying throughout the method:**

Replace all instances of `user_query` with `expanded_question` EXCEPT:
- When building final response context (keep original for natural answer)
- When updating topics (keep original for context tracking)

**Specifically change:**
1. Line 507: `self._decide_should_search(ctx, expanded_question, prior_responses_text)`
2. Line 522: `self._generate_search_seed(ctx, expanded_question, prior_responses_text)`
3. Search orchestrator calls
4. Response generation (KEEP user_query here for natural answers)

---

## Phase 5: Update Context Building

### File: `src/context.py`

**Review `build_query_context()` - may need to pass `expanded_question`**

This depends on whether question_keywords/question_embedding should be based on:
- Original question (for topic matching)
- Expanded question (for better search)

**Decision:** Keep context building on original question, use expanded only for search/logic.

---

## Phase 6: Clean Up Downstream Prompts

### Files: `src/prompts.py`

Now that pronouns are resolved upfront, we can **simplify** these prompts:

**Remove pronoun-handling logic from:**

1. `seed_prompt` (lines 325-329, 336):
   - Remove examples with pronouns
   - Remove rule #3 about pronouns

2. `planning_prompt`:
   - Already focused on exact question

3. `search_decision_prompt` (lines 227, 231-233):
   - Remove pronoun checklist items
   - Remove pronoun examples

**This simplification makes prompts cleaner and reduces confusion.**

---

## Phase 7: Testing

### Create new test file: `tests/test_question_expansion.py`

```python
"""Tests for question expansion/context resolution."""

import pytest
from unittest.mock import Mock, MagicMock
from src.agent import Agent
from src.config import AgentConfig


class TestQuestionExpansion:
    """Test question expansion with pronouns and references."""

    def test_expands_pronoun_he(self):
        """Should replace 'he' with entity from conversation."""
        # Mock agent with conversation history
        # Test: "Who is he?" → "Who is [Person Name]?"
        pass

    def test_expands_pronoun_they(self):
        """Should replace 'they/them' with entity."""
        # Test: "Where can I buy them?" → "Where can I buy [product]?"
        pass

    def test_expands_implicit_reference(self):
        """Should make implicit references explicit."""
        # Test: "How much does it cost?" → "How much does [product] cost?"
        pass

    def test_no_expansion_needed(self):
        """Should return original if already standalone."""
        # Test: "What is the weather?" → "What is the weather?"
        pass

    def test_handles_empty_conversation(self):
        """Should work with no conversation history."""
        pass

    def test_expansion_failure_returns_original(self):
        """Should fallback to original on error."""
        pass

    def test_multiple_pronouns(self):
        """Should handle multiple pronoun replacements."""
        # Test: "Does he have a partner?" → "Does [Person] have a partner?"
        pass
```

### Update existing tests

**Files to update:**
- `tests/test_agent_core.py` - Add question_expansion to chain mocks
- `tests/test_agent_helpers.py` - Mock _expand_question calls
- `tests/test_chains.py` (if exists) - Add question_expansion chain test

---

## Phase 8: Configuration (Optional)

### File: `src/config.py`

**Add optional config flag:**

```python
@dataclass
class AgentConfig:
    # ... existing fields ...

    # Question expansion
    enable_question_expansion: bool = True  # Can disable if needed
```

**Then in agent.py:**

```python
if cfg.enable_question_expansion:
    expanded_question = self._expand_question(ctx, user_query, prior_responses_text)
else:
    expanded_question = user_query
```

---

## Implementation Order

### Step 1: Create Prompt
- [ ] Add `question_expansion_prompt` to `src/prompts.py`
- [ ] Test prompt manually with sample inputs

### Step 2: Add Chain
- [ ] Add chain to `src/chains.py`
- [ ] Verify chain builds correctly

### Step 3: Add Agent Method
- [ ] Add `_expand_question()` method to `src/agent.py`
- [ ] Add "question_expansion" to rebuild_counts
- [ ] Test method in isolation

### Step 4: Integrate Pipeline
- [ ] Modify `_handle_query_core()` to call expansion
- [ ] Replace `user_query` with `expanded_question` in search/decision logic
- [ ] Keep `user_query` for response generation

### Step 5: Test Integration
- [ ] Run existing tests - ensure nothing breaks
- [ ] Create new test file for question expansion
- [ ] Test with real conversations

### Step 6: Simplify Prompts (Optional)
- [ ] Remove pronoun logic from seed_prompt
- [ ] Remove pronoun logic from search_decision_prompt
- [ ] Test to ensure no regression

### Step 7: Documentation
- [ ] Update README if needed
- [ ] Add inline comments explaining the expansion step

---

## Expected Benefits

✅ **Single source of truth** for context resolution
✅ **Better search queries** - more explicit and specific
✅ **Reduced confusion** - downstream prompts work with clear questions
✅ **Simplified prompts** - remove scattered pronoun-handling logic
✅ **Easier debugging** - can see expanded question in logs
✅ **More reliable** - consistent pronoun resolution across all prompts

---

## Potential Issues & Solutions

### Issue 1: Extra LLM Call Adds Latency
**Solution:** Use robot model (small/fast). Cost is minimal (~50-100 tokens).

### Issue 2: Expansion Might Change Question Intent
**Solution:** Strict prompt rules - only resolve pronouns, don't add info.

### Issue 3: Tests Might Need Extensive Mocking
**Solution:** Mock at chain level, not LLM level. Keep tests simple.

### Issue 4: Backward Compatibility
**Solution:** Make it optional via config flag (enable_question_expansion).

---

## Rollback Plan

If issues arise:
1. Set `enable_question_expansion = False` in config
2. System falls back to original behavior
3. No data loss or breaking changes

---

## Success Criteria

✅ Question "Who is his VP?" correctly expands to "Who is [President]'s vice president?"
✅ All existing tests pass
✅ New expansion tests pass (100% coverage)
✅ No regression in search quality
✅ System handles edge cases gracefully (empty conversation, expansion failure)

---

## Timeline Estimate

- **Step 1-2:** 30 minutes (prompt + chain)
- **Step 3-4:** 1 hour (agent integration)
- **Step 5:** 1-2 hours (testing + fixes)
- **Step 6:** 30 minutes (cleanup prompts)
- **Total:** ~3-4 hours

---

## Files Changed Summary

| File | Type | Changes |
|------|------|---------|
| `src/prompts.py` | Modified | Add question_expansion_prompt |
| `src/chains.py` | Modified | Add question_expansion chain |
| `src/agent.py` | Modified | Add _expand_question() method, integrate into pipeline |
| `src/config.py` | Modified (optional) | Add enable_question_expansion flag |
| `tests/test_question_expansion.py` | New | Test suite for expansion logic |
| `tests/test_agent_core.py` | Modified | Update mocks for new chain |

---

## Next Steps

1. **Review this plan** - Any changes needed?
2. **Approve implementation** - Ready to proceed?
3. **Choose approach** - All at once or phase by phase?

Let me know when you're ready to start implementation!
