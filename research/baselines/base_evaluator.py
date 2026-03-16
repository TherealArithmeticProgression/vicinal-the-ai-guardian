"""
research/baselines/base_evaluator.py

Abstract base class that every baseline evaluator must implement.

A BaselineEvaluator receives a raw prompt string and returns a
BaselineResult containing:
  - a binary label  (1 = threat, 0 = benign)
  - a confidence score ∈ [0, 1]
  - latency in milliseconds

All baselines share this interface so the experiment runner can treat them
uniformly regardless of implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BaselineResult:
    """Output of a single baseline evaluation."""
    baseline_name: str
    label: int          # 1 = threat, 0 = benign
    score: float        # confidence ∈ [0, 1]
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_threat(self) -> bool:
        return self.label == 1


class BaseEvaluator(ABC):
    """Abstract baseline evaluator."""

    name: str = "base"
    threshold: float = 0.5   # Score ≥ this → label = 1

    @abstractmethod
    def evaluate(self, prompt: str) -> BaselineResult:
        """Evaluate a single prompt. Must be synchronous."""
        ...

    def evaluate_batch(self, prompts: list[str]) -> list[BaselineResult]:
        """Default: evaluate one by one. Override for batch efficiency."""
        return [self.evaluate(p) for p in prompts]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, threshold={self.threshold})"
