from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from qnavi.errors import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._entries:
                raise RegistryError(f"{self.kind} '{name}' is already registered")
            self._entries[name] = factory
            return factory

        return decorator

    def create(self, name: str, **kwargs: Any) -> T:
        if name not in self._entries:
            known = ", ".join(sorted(self._entries)) or "none"
            raise RegistryError(f"unknown {self.kind} '{name}'; available: {known}")
        return self._entries[name](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries
