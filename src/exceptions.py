from __future__ import annotations

import importlib
from typing import Type


class _DefaultResponseError(Exception):
    pass


# Start with the safe fallback type, then override if a concrete
# implementation is discovered in one of the optional `ollama` modules.
_resp: Type[Exception] = _DefaultResponseError
for modname in ("ollama", "ollama._types"):
    try:
        mod = importlib.import_module(modname)
    except Exception:
        continue
    if hasattr(mod, "ResponseError"):
        _resp = getattr(mod, "ResponseError")
        break

ResponseError: Type[Exception] = _resp


__all__ = ["ResponseError"]
