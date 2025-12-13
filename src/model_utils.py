"""Small helpers for consistent model-missing messages and handling.

This module centralizes the logging + optional agent error-marking used when
an Ollama model is not available locally (so messages are consistent).
"""

from __future__ import annotations

import logging
from typing import Callable


def handle_missing_model(mark_error: Callable[[str], str] | None, role: str, model_name: str) -> str:
    """Log and optionally mark a missing model message.

    mark_error: a callable (like Agent._mark_error) that accepts a string; may be None.
    role: human-friendly role name, e.g. "Robot" or "Assistant".
    model_name: the Ollama model name to suggest pulling.
    """
    msg = f"{role} model '{model_name}' not found. Run 'ollama pull {model_name}' and retry."
    logging.error(msg)
    if mark_error:
        try:
            mark_error(msg)
        except Exception:
            logging.debug("mark_error failed for missing model message", exc_info=True)
    return msg


__all__ = ["handle_missing_model"]
