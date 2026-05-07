"""Canonical agent interface for DIME evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Minimal inference-only interface required by the benchmark harness."""

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        """Reset per-episode agent state."""

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """Return an action for the current observation."""
        raise NotImplementedError
