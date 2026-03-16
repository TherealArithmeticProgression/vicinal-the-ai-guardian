"""
core/types/verdict.py

VerdictResult — the final output of the Vicinal engine for a given prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from core.types.safety_score import SafetyScore, ThreatCategory


class Verdict(str, Enum):
    """
    Four-state verdict assigned by the DecisionMaker.

    ALLOW  — prompt is safe; forward to model unchanged.
    WARN   — prompt is suspicious; forward with a warning annotation
             so the caller can decide whether to proceed.
    BLOCK  — prompt is dangerous; do NOT forward to the model.
    REDACT — prompt contains sensitive content; caller should strip
             the flagged spans before forwarding.
    """
    ALLOW  = "ALLOW"
    WARN   = "WARN"
    BLOCK  = "BLOCK"
    REDACT = "REDACT"


@dataclass
class VerdictResult:
    """
    Complete evaluation result returned by VicinalGuard.evaluate().

    Attributes
    ----------
    verdict:
        The actionable decision.
    composite_score:
        Merged threat probability ∈ [0, 1].
    scores:
        Individual SafetyScore objects from each evaluator.
    session_id:
        Mirrors the session_id from the input PromptPayload.
    reason:
        Short human-readable explanation of the verdict.
    top_category:
        The dominant threat category that drove the verdict.
    redacted_text:
        If verdict is REDACT, this contains the sanitised prompt text.
        Otherwise None.
    metadata:
        Passed through from the input payload for traceability.
    created_at:
        UTC timestamp of the evaluation.
    """

    verdict: Verdict
    composite_score: float
    scores: list[SafetyScore] = field(default_factory=list)
    session_id: str = ""
    reason: str = ""
    top_category: ThreatCategory = ThreatCategory.UNKNOWN
    redacted_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_blocked(self) -> bool:
        return self.verdict == Verdict.BLOCK

    @property
    def is_safe(self) -> bool:
        return self.verdict == Verdict.ALLOW

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "composite_score": round(self.composite_score, 4),
            "reason": self.reason,
            "top_category": self.top_category.value,
            "session_id": self.session_id,
            "redacted_text": self.redacted_text,
            "evaluators": [
                {
                    "name": s.evaluator,
                    "score": round(s.composite, 4),
                    "latency_ms": round(s.latency_ms, 2),
                    "hits": [
                        {
                            "category": h.category.value,
                            "score": round(h.score, 4),
                            "evidence": h.evidence,
                        }
                        for h in s.hits
                    ],
                }
                for s in self.scores
            ],
            "created_at": self.created_at.isoformat(),
        }
