"""
core/decisions/decision_maker.py

DecisionMaker — converts a list of SafetyScores into a final VerdictResult.

Thresholds (configurable via policy):
  ≥ BLOCK_THRESHOLD  → BLOCK
  ≥ WARN_THRESHOLD   → WARN
  ≥ REDACT_THRESHOLD → REDACT  (when PII/exfil is the top category)
  else               → ALLOW
"""

from __future__ import annotations

from typing import Any

from core.types.safety_score import SafetyScore, ThreatCategory, CATEGORY_SEVERITY
from core.types.verdict import Verdict, VerdictResult

# Default thresholds — overridden by policy YAML
DEFAULT_BLOCK_THRESHOLD  = 0.80
DEFAULT_WARN_THRESHOLD   = 0.60
DEFAULT_REDACT_THRESHOLD = 0.55

# Categories that trigger REDACT rather than WARN when in range
REDACT_CATEGORIES = {
    ThreatCategory.PII_EXTRACTION,
    ThreatCategory.DATA_EXFILTRATION,
}


class DecisionMaker:
    """Stateless composite-score → verdict mapping."""

    def __init__(self, policy: dict[str, Any] | None = None) -> None:
        p = policy or {}
        self.block_threshold  = float(p.get("block_threshold",  DEFAULT_BLOCK_THRESHOLD))
        self.warn_threshold   = float(p.get("warn_threshold",   DEFAULT_WARN_THRESHOLD))
        self.redact_threshold = float(p.get("redact_threshold", DEFAULT_REDACT_THRESHOLD))

    def decide(
        self,
        scores: list[SafetyScore],
        session_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> VerdictResult:
        if not scores:
            return VerdictResult(
                verdict=Verdict.ALLOW,
                composite_score=0.0,
                scores=[],
                session_id=session_id,
                reason="No evaluators ran.",
                metadata=metadata or {},
            )

        composite = self._merge(scores)
        top_category = self._dominant_category(scores)

        verdict, reason = self._threshold(composite, top_category)

        # Build redacted text if needed
        redacted_text: str | None = None
        if verdict == Verdict.REDACT:
            # Placeholder — real redaction happens in engine.py with spans
            redacted_text = "[REDACTED by Vicinal]"

        return VerdictResult(
            verdict=verdict,
            composite_score=composite,
            scores=scores,
            session_id=session_id,
            reason=reason,
            top_category=top_category,
            redacted_text=redacted_text,
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _merge(self, scores: list[SafetyScore]) -> float:
        """
        Weighted average of evaluator composite scores.
        Context-driven evaluator is up-weighted when present because it
        has richer signal than the zero-resource path.
        """
        weights = {"zero_resource": 1.0, "context_driven": 1.4}
        total_w, total_s = 0.0, 0.0
        for s in scores:
            w = weights.get(s.evaluator, 1.0)
            total_s += s.composite * w
            total_w += w
        return total_s / total_w if total_w else 0.0

    def _dominant_category(self, scores: list[SafetyScore]) -> ThreatCategory:
        """Return the highest-severity threat category across all hits."""
        best: tuple[float, ThreatCategory] = (0.0, ThreatCategory.UNKNOWN)
        for s in scores:
            for hit in s.hits:
                severity = CATEGORY_SEVERITY.get(hit.category, 0.5)
                weighted = hit.score * severity
                if weighted > best[0]:
                    best = (weighted, hit.category)
        return best[1]

    def _threshold(self, score: float, category: ThreatCategory) -> tuple[Verdict, str]:
        if score >= self.block_threshold:
            return (
                Verdict.BLOCK,
                f"Threat score {score:.2f} exceeds block threshold "
                f"({self.block_threshold}). Dominant category: {category.value}.",
            )
        if score >= self.redact_threshold and category in REDACT_CATEGORIES:
            return (
                Verdict.REDACT,
                f"Sensitive content detected (category: {category.value}, "
                f"score: {score:.2f}). Prompt should be redacted before forwarding.",
            )
        if score >= self.warn_threshold:
            return (
                Verdict.WARN,
                f"Suspicious prompt (score: {score:.2f}, "
                f"category: {category.value}). Proceeding with caution.",
            )
        return (
            Verdict.ALLOW,
            f"Prompt cleared all thresholds (score: {score:.2f}).",
        )
