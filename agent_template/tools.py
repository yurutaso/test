from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from langchain_core.tools import tool


class BaseToolset(ABC):
    """Interface for reusable tool collections."""

    @abstractmethod
    def list(self) -> Iterable[object]:
        """Return an iterable of LangChain tool callables."""


class MathToolset(BaseToolset):
    """Example toolset with simple numeric helpers."""

    @tool
    def add(x: float, y: float) -> float:
        """Add two numbers together."""
        return x + y

    @tool
    def multiply(x: float, y: float) -> float:
        """Multiply two numbers."""
        return x * y

    def list(self) -> Iterable[object]:
        return [self.add, self.multiply]


def create_default_tools() -> Iterable[object]:
    """Convenience helper to get a simple toolset."""

    return MathToolset().list()


__all__ = ["BaseToolset", "MathToolset", "create_default_tools"]
