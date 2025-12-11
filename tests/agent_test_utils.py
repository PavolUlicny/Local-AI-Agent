from __future__ import annotations

from typing import Any, Iterable


class DummyChain:
    def __init__(self, *, outputs: Iterable[str] | None = None, stream_tokens: Iterable[str] | None = None):
        self.outputs = list(outputs or [])
        self.stream_tokens = list(stream_tokens or [])
        self.invocations: list[dict[str, Any]] = []

    def invoke(self, inputs: dict[str, Any]) -> str:
        self.invocations.append(inputs)
        if self.outputs:
            return self.outputs.pop(0)
        return ""

    def stream(self, inputs: dict[str, Any]):
        self.invocations.append(inputs)
        for token in self.stream_tokens:
            yield token


class _RepeatChain(DummyChain):
    def __init__(self, value: str):
        super().__init__()
        self.value = value

    def invoke(self, inputs: dict[str, Any]) -> str:  # noqa: D401
        self.invocations.append(inputs)
        return self.value


class _AlwaysYesChain(DummyChain):
    def invoke(self, inputs: dict[str, Any]) -> str:  # noqa: D401
        self.invocations.append(inputs)
        return "YES"


class _IncrementingQueryChain(DummyChain):
    def __init__(self, prefix: str = "query"):
        super().__init__()
        self.prefix = prefix
        self.counter = 0

    def invoke(self, inputs: dict[str, Any]) -> str:  # noqa: D401
        self.invocations.append(inputs)
        self.counter += 1
        return f"{self.prefix} {self.counter}"


__all__ = [
    "DummyChain",
    "_RepeatChain",
    "_AlwaysYesChain",
    "_IncrementingQueryChain",
]
