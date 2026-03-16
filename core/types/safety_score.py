"""
core/types/safety_score.py

SafetyScore — the intermediate scoring container produced by each evaluator.
Multiple SafetyScores are merged by the Scorer into a single composite
score before the DecisionMaker assigns a Verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ThreatCategory(str, Enum):
    """Known threat categories tracked by Vicinal."""
    PROMPT_INJECTION    = "prompt_injection"
    JAILBREAK           = "jailbreak"
    PII_EXTRACTION      = "pii_extraction"
    DATA_EXFILTRATION   = "data_exfiltration"
    INDIRECT_INJECTION  = "indirect_injection"
    SOCIAL_ENGINEERING  = "social_engineering"
    HARMFUL_CONTENT     = "harmful_content"
    POLICY_VIOLATION    = "policy_violation"
    UNKNOWN             = "unknown"


# Severity weight per category — higher = treated more seriously by scorer
CATEGORY_SEVERITY: dict[ThreatCategory, float] = {
    ThreatCategory.PROMPT_INJECTION:   1.0,
    ThreatCategory.JAILBREAK:          1.0,
    ThreatCategory.PII_EXTRACTION:     0.9,
    ThreatCategory.DATA_EXFILTRATION:  0.9,
    ThreatCategory.INDIRECT_INJECTION: 0.85,
    ThreatCategory.SOCIAL_ENGINEERING: 0.75,
    ThreatCategory.HARMFUL_CONTENT:    0.8,
    ThreatCategory.POLICY_VIOLATION:   0.7,
    ThreatCategory.UNKNOWN:            0.5,
}


@dataclass
class ThreatHit:
    """A single match between the input and a known threat pattern."""
    category: ThreatCategory
    score: float                  # ∈ [0, 1]; higher = more suspicious
    evidence: str = ""            # human-readable explanation
    source: str = ""              # evaluator that produced this hit
    distance: float | None = None # raw FAISS L2 distance if applicable
    matched_pattern: str | None = None  # nearest threat text in index


@dataclass
class SafetyScore:
    """
    Aggregate score produced by a single evaluator (zero-resource or
    context-driven).

    Attributes
    ----------
    composite:
        Overall threat probability ∈ [0, 1].  1 = definitely malicious.
    hits:
        Individual threat hits that contributed to this score.
    evaluator:
        Name of the evaluator that produced this score.
    latency_ms:
        Wall-clock time taken to produce the score.
    raw:
        Arbitrary evaluator-specific data for debugging / research.
    """

    composite: float                              # ∈ [0, 1]
    evaluator: str
    hits: list[ThreatHit] = field(default_factory=list)
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def top_category(self) -> ThreatCategory:
        if not self.hits:
            return ThreatCategory.UNKNOWN
        return max(self.hits, key=lambda h: h.score).category

    @property
    def is_clean(self) -> bool:
        return self.composite < 0.45
