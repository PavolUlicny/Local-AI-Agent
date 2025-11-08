from importlib import import_module
from typing import List, Tuple


def _resolve_prompt_template():
    """Return the PromptTemplate class from the installed LangChain package."""
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

try:
    from ollama._types import ResponseError
except ImportError:  # Older packages expose the error on the client module
    ResponseError = Exception


def main() -> None:
    """Run a local Ollama-backed model with DuckDuckGo search context."""
    max_context_turns = 6

    llm = OllamaLLM(
        model="llama3:8b",
        temperature=0.7,
        top_p=0.9,
        top_k=64,
        repeat_penalty=1.05,
        num_predict=4096,
        num_ctx=4096,
    )
    search = DuckDuckGoSearchRun()

    response_prompt = PromptTemplate(
        input_variables=["conversation_history", "search_results", "user_question"],
        template=(
            "You are an expert research assistant AI (Ollama 8b parameter model). Using the web search results below, "
            "craft the most comprehensive answer you can. Aim to fully satisfy the request, "
            "matching the user's desired level of depth and length. Do not claim a task is "
            "unreasonable; instead, deliver the best possible response within the available "
            "context.\n\n"
            "Conversation so far:\n{conversation_history}\n\n"
            "Search results:\n{search_results}\n\n"
            "User question: {user_question}"
        ),
    )

    planning_prompt = PromptTemplate(
        input_variables=["conversation_history", "user_question", "results_to_date"],
        template=(
            "You are assessing whether more web searches are needed to answer the question.\n"
            "Question: {user_question}\n\n"
            "Conversation so far:\n{conversation_history}\n\n"
            "Results collected so far:\n{results_to_date}\n\n"
            "Propose up to 2 high-value follow-up search queries that would meaningfully improve "
            "the final response, especially if the user requested exceptional depth. Return "
            "each query on its own line with no numbering. If no additional searches are "
            "required, respond with NONE."
        ),
    )

    conversation_history: List[Tuple[str, str]] = []

    while True:
        try:
            user_query = input("Enter your request (or 'exit' to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            return

        if not user_query:
            print("No input provided. Please try again or type 'exit' to quit.")
            continue

        if user_query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return

        recent_history = conversation_history[-max_context_turns:]
        conversation_text = (
            "None"
            if not recent_history
            else "\n\n".join(
                f"User: {user_msg}\nAssistant: {assistant_msg}"
                for user_msg, assistant_msg in recent_history
            )
        )

        aggregated_results: List[str] = []
        pending_queries: List[str] = [user_query]
        max_rounds = 5

        round_index = 0
        while round_index < len(pending_queries) and round_index < max_rounds:
            current_query = pending_queries[round_index]
            result = search.run(current_query)
            aggregated_results.append(
                f"Query {round_index + 1}: {current_query}\n{result}"
            )

            round_index += 1
            if round_index >= max_rounds:
                break

            results_to_date = "\n\n".join(aggregated_results) or "None"
            suggestions_raw = llm.invoke(
                planning_prompt.format(
                    conversation_history=conversation_text,
                    user_question=user_query,
                    results_to_date=results_to_date,
                )
            )

            new_queries: List[str] = []
            for line in str(suggestions_raw).splitlines():
                normalized = line.strip().strip("-*").strip()
                if not normalized:
                    continue
                if normalized.lower() == "none":
                    new_queries = []
                    break
                new_queries.append(normalized)

            for candidate in new_queries:
                if candidate not in pending_queries and len(pending_queries) < max_rounds:
                    pending_queries.append(candidate)

        formatted_prompt = response_prompt.format(
            conversation_history=conversation_text,
            search_results="\n\n".join(aggregated_results) or "No search results.",
            user_question=user_query,
        )

        try:
            response_stream = llm.stream(formatted_prompt)
        except ResponseError as exc:
            message = str(exc)
            if "not found" in message.lower():
                print(
                    "Model 'llama3:8b' not found. Run 'ollama pull llama3:8b' in a terminal "
                    "before retrying."
                )
                return
            raise

        print("\n[Ollama Answer]")
        response_chunks: List[str] = []
        for chunk in response_stream:
            text_chunk = str(chunk)
            response_chunks.append(text_chunk)
            print(text_chunk, end="", flush=True)

        if response_chunks and not response_chunks[-1].endswith("\n"):
            print()

        response_text = "".join(response_chunks)

        conversation_history.append((user_query, response_text))


if __name__ == "__main__":
    main()
