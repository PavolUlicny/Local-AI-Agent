## Conversation Management

The agent maintains a simple conversation history as a list of turns, where each turn contains:
- User query
- Assistant response
- Timestamp (UTC)
- Search usage flag

### Auto-Trim

When the conversation exceeds the configured character budget (`--max-conversation-chars`, default 64000 ≈ 16k tokens), the oldest turns are automatically removed to stay within budget.

The auto-trim mechanism:
1. Calculates total character count including formatting overhead
2. Removes oldest turns until under budget
3. Preserves at least the most recent turn

### Manual Control

Users can manually manage conversation history with slash commands:

- `/clear` - Clear all conversation history and start fresh
- `/compact` - Keep only the last N turns (configured via `--compact-keep-turns`, default 10)
- `/stats` - View current conversation statistics

### Conversation Statistics

The `/stats` command shows:
- Total number of turns
- Total character count (with formatting)
- Percentage of budget used
- Number of turns that used search

### Configuration

Key parameters:

- `--max-conversation-chars` (`--mcc`): Maximum character budget (default 64000)
  - ~16k tokens for most models
  - ~90 conversation turns with search enabled
  - ~180 turns for chat-only interactions

- `--compact-keep-turns` (`--ckt`): Number of recent turns to keep with `/compact` (default 10)

### Design Rationale

The conversation system is designed to:
1. **Simplicity**: Simple list-based history vs. complex topic tracking
2. **Transparency**: Users know exactly what context the LLM sees
3. **Control**: Manual commands for clearing/compacting history
4. **Scalability**: Works with 128k context models (phi4-mini:3.8b, llama3.1:8b)
5. **Efficiency**: Automatic trimming prevents context overflow

### Implementation

See `src/conversation.py` for the `ConversationManager` implementation and `src/commands.py` for command handling.
