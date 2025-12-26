"""Response generation and streaming helpers extracted from `Agent`.

Provides a single entrypoint `generate_and_stream_response(agent, ...)` which
invokes the configured response chain, streams output via an injectable
`write_fn`, handles rebuild-on-context-length retries, and returns the
aggregated text or `None` on fatal errors.
"""

from __future__ import annotations

from typing import Any, List, Callable
import logging

try:
    from prompt_toolkit.formatted_text import ANSI as _ANSI
except Exception:
    ANSI = None
else:
    ANSI = _ANSI

from . import exceptions as _exceptions
from . import text_utils as _text_utils_mod
from . import model_utils as _model_utils_mod
from .constants import ChainName, RebuildKey, MAX_REBUILD_RETRIES


def generate_and_stream_response(
    agent: Any,
    resp_inputs: dict[str, Any],
    chain_name: ChainName | str,  # Accept both enums and strings
    one_shot: bool,
    write_fn: Callable[[str], None] | None = None,
) -> str | None:
    if write_fn is None:
        write_fn = agent._write
    cfg = agent.cfg

    # Convert string to ChainName if needed (allow arbitrary strings for test flexibility)
    if isinstance(chain_name, str):
        try:
            chain_name = ChainName(chain_name)
        except ValueError:
            # Not a valid enum value - keep as string (used by tests)
            pass

    chain = agent.chains[chain_name]
    try:
        response_stream = chain.stream(resp_inputs)
    except _exceptions.ResponseError as exc:
        if "not found" in str(exc).lower():
            _model_utils_mod.handle_missing_model(agent._mark_error, "Assistant", cfg.assistant_model)
            return None
        if _text_utils_mod.is_context_length_error(str(exc)):
            if agent.rebuild_counts[RebuildKey.ANSWER] < MAX_REBUILD_RETRIES:
                agent._reduce_context_and_rebuild(RebuildKey.ANSWER, "answer")
                try:
                    chain = agent.chains[
                        ChainName.RESPONSE if chain_name == ChainName.RESPONSE else ChainName.RESPONSE_NO_SEARCH
                    ]
                    response_stream = chain.stream(resp_inputs)
                except _exceptions.ResponseError as exc2:
                    logging.error(f"Answer generation failed after retry: {exc2}")
                    agent._mark_error("Answer generation failed after retry; see logs for details.")
                    return None
            else:
                logging.error("Reached answer generation rebuild cap; please shorten your query or reset session.")
                agent._mark_error(
                    "Answer generation failed: exceeded rebuild attempts; "
                    "please shorten your query or reset session."
                )
                return None
        else:
            logging.error(f"Answer generation failed: {exc}")
            agent._mark_error("Answer generation failed; see logs for details.")
            return None
    except (RuntimeError, ValueError, TypeError) as exc:
        # Runtime/programming errors (should be rare)
        logging.error(f"Answer generation failed with error: {exc}", exc_info=True)
        agent._mark_error("Answer generation failed; see logs for details.")
        return None
    except Exception as exc:
        # Truly unexpected errors
        logging.error(f"Answer generation failed unexpectedly: {exc}", exc_info=True)
        agent._mark_error("Answer generation failed unexpectedly; see logs for details.")
        return None

    if ANSI is not None and getattr(agent, "_is_tty", False):
        agent._writeln("\n\033[91m[Answer]\033[0m")
    else:
        agent._writeln("\n[Answer]")
    response_chunks: List[str] = []
    stream_error: Exception | None = None
    try:
        for chunk in response_stream:
            response_chunks.append(chunk)
            write_fn(chunk)
    except KeyboardInterrupt:
        logging.info("Streaming interrupted by user.")
    except (ConnectionError, TimeoutError, OSError) as exc:
        # Network/connection errors during streaming
        stream_error = exc
        logging.error(f"Streaming network error: {exc}")
    except (RuntimeError, ValueError) as exc:
        # Runtime errors during streaming
        stream_error = exc
        logging.error(f"Streaming runtime error: {exc}", exc_info=True)
    except Exception as exc:
        # Unexpected streaming errors
        stream_error = exc
        logging.error(f"Unexpected streaming error: {exc}", exc_info=True)
    if response_chunks and not response_chunks[-1].endswith("\n"):
        agent._writeln()
    if one_shot:
        agent._writeln()
    if stream_error:
        agent._mark_error("Answer streaming failed; please retry.")
        return None
    return "".join(response_chunks)


__all__ = ["generate_and_stream_response"]
