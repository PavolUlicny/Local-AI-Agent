from __future__ import annotations

try:
    from ollama import ResponseError as OllamaResponseError
    ResponseError = OllamaResponseError  # type: ignore
except ImportError:
    try:
        from ollama._types import ResponseError as OllamaResponseError
        ResponseError = OllamaResponseError  # type: ignore
    except ImportError:  # pragma: no cover - fallback
        class ResponseError(Exception):
            pass

__all__ = ["ResponseError"]
