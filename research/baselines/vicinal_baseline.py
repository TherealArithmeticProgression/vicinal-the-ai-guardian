"""
research/baselines/vicinal_baseline.py

Baseline 3: Vicinal (FAISS zero-resource or full pipeline).

This wraps VicinalGuard so it can be used in the research experiment
runner alongside the other baselines with a unified interface.

Two sub-modes accessible via 'mode' parameter:
  - "zero_resource"  : FAISS embedding distance only
  - "full"           : FAISS + OCR + RAG (context-driven)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from research.baselines.base_evaluator import BaseEvaluator, BaselineResult

# Add sdk/python to path if running from repo root
_SDK_PATH = Path(__file__).resolve().parents[3] / "sdk" / "python"
if str(_SDK_PATH) not in sys.path:
    sys.path.insert(0, str(_SDK_PATH))


class VicinalBaseline(BaseEvaluator):
    """
    Vicinal guardrail as a research baseline.

    Parameters
    ----------
    mode:
        "zero_resource" or "full"
    threshold:
        Score ≥ this → label = 1 (threat).
    config_overrides:
        Dict of VicinalConfig kwargs.
    """

    def __init__(
        self,
        mode: str = "zero_resource",
        threshold: float = 0.60,
        **config_overrides,
    ) -> None:
        self.name = f"vicinal_{mode}"
        self.threshold = threshold
        self._mode = mode
        self._overrides = config_overrides
        self._guard = None   # lazy init

    def evaluate(self, prompt: str) -> BaselineResult:
        guard = self._get_guard()
        t0 = time.perf_counter()

        result = guard.evaluate(prompt)
        elapsed = (time.perf_counter() - t0) * 1000

        label = 1 if result.composite_score >= self.threshold else 0

        return BaselineResult(
            baseline_name=self.name,
            label=label,
            score=result.composite_score,
            latency_ms=elapsed,
            metadata={
                "verdict": result.verdict.value,
                "top_category": result.top_category.value,
                "reason": result.reason,
            },
        )

    def _get_guard(self):
        if self._guard is None:
            from vicinal import VicinalGuard, VicinalConfig
            cfg = VicinalConfig(mode=self._mode, log_level="ERROR", **self._overrides)
            self._guard = VicinalGuard(cfg)
        return self._guard
