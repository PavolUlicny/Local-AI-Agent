from __future__ import annotations

import importlib
from typing import Type, Optional

_found = None
_resp: Optional[Type[Exception]] = None
for modname in ("ollama", "ollama._types"):
    try:
        mod = importlib.import_module(modname)
    except Exception:
        continue
    if hasattr(mod, "ResponseError"):
        _found = getattr(mod, "ResponseError")
        break

if _found is None:  # pragma: no cover - fallback

    class _DefaultResponseError(Exception):
        pass

    _resp = _DefaultResponseError
else:
    _resp = _found
ResponseError: Type[Exception] = _resp  # type: ignore[assignment]


__all__ = ["ResponseError"]
