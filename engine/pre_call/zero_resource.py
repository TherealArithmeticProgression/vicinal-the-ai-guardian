"""
engine/pre_call/zero_resource.py

Zero-Resource Pre-Call Evaluator.

No external API calls.  No internet.  No model inference beyond the
bundled sentence-transformer (which runs locally).

Pipeline
--------
1. Encode the active prompt text into a dense vector.
2. Search the FAISS threat index for top-k nearest neighbours.
3. Convert distances to a threat score using a calibrated sigmoid.
4. Map nearest-neighbour categories to ThreatHit objects.
5. Return a SafetyScore.

Distance calibration
--------------------
For L2-normalised vectors the L2 distance ∈ [0, 2]:
  - distance ≈ 0   → vectors are identical  (very suspicious)
  - distance ≈ 1   → orthogonal             (unrelated)
  - distance ≈ 2   → opposite               (impossible for unit vectors,
                                             practically very ~1.4)

We map this to threat_score ∈ [0, 1]:
    threat_score = max(0, 1 - distance / DISTANCE_SCALE)

where DISTANCE_SCALE is calibrated so that score → 0 around distance 1.2
(well past the decision boundary on identical normalised vectors).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from core.context.context_builder import EvaluationContext
from core.errors.exceptions import EvaluationError
from core.types.safety_score import SafetyScore, ThreatCategory, ThreatHit
from engine.evaluators.embedding_evaluator import EmbeddingEvaluator
from engine.evaluators.faiss_evaluator import FaissEvaluator

logger = logging.getLogger(__name__)

# L2 distance at which the threat score reaches 0
DISTANCE_SCALE = 1.2

# How many neighbours to pull from the index
TOP_K = 10

# Weight applied to the closest neighbour vs. the aggregate
CLOSEST_WEIGHT = 0.6


class ZeroResourceEvaluator:
    """
    Evaluates a prompt using only pre-computed FAISS threat embeddings.
    No network calls, no LLM inference.

    Parameters
    ----------
    embedding_evaluator:
        Shared EmbeddingEvaluator instance.
    faiss_evaluator:
        Shared FaissEvaluator instance.
    top_k:
        Number of nearest neighbours to retrieve.
    distance_scale:
        Calibration scale — see module docstring.
    """

    def __init__(
        self,
        embedding_evaluator: EmbeddingEvaluator | None = None,
        faiss_evaluator: FaissEvaluator | None = None,
        top_k: int = TOP_K,
        distance_scale: float = DISTANCE_SCALE,
    ) -> None:
        self._embedder = embedding_evaluator or EmbeddingEvaluator.get()
        self._faiss    = faiss_evaluator    or FaissEvaluator()
        self._top_k    = top_k
        self._scale    = distance_scale

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def evaluate(self, context: EvaluationContext) -> SafetyScore:
        """
        Run the zero-resource evaluation pass on this context.

        Returns
        -------
        SafetyScore
            Composite threat score + individual hits.
        """
        t0 = time.perf_counter()

        if not context.prompt_text.strip():
            return SafetyScore(composite=0.0, evaluator="zero_resource",
                               latency_ms=0.0, raw={"reason": "empty prompt"})

        try:
            vector = self._embedder.encode(context.prompt_text)
        except Exception as exc:
            raise EvaluationError(str(exc), evaluator="zero_resource") from exc

        try:
            results = self._faiss.search(vector, k=self._top_k)
        except Exception as exc:
            raise EvaluationError(str(exc), evaluator="zero_resource") from exc

        if not results:
            return SafetyScore(composite=0.0, evaluator="zero_resource",
                               latency_ms=(time.perf_counter() - t0) * 1000,
                               raw={"reason": "index empty or no results"})

        hits: list[ThreatHit] = []
        scores: list[float] = []

        for rank, result in enumerate(results):
            raw_score = max(0.0, 1.0 - result.distance / self._scale)
            # Apply a small rank penalty so the closest hit dominates
            rank_weight = CLOSEST_WEIGHT if rank == 0 else (1.0 - CLOSEST_WEIGHT) / (len(results) - 1)
            scores.append(raw_score * rank_weight if rank > 0 else raw_score)

            category = self._parse_category(result.category)

            if raw_score > 0.1:   # ignore near-zero hits to keep signal clean
                hits.append(
                    ThreatHit(
                        category=category,
                        score=raw_score,
                        evidence=(
                            f"Distance {result.distance:.4f} from known threat "
                            f"'{result.label}' in FAISS index."
                        ),
                        source="zero_resource",
                        distance=result.distance,
                        matched_pattern=result.text[:200] if result.text else None,
                    )
                )

        # Composite = weighted sum clipped to [0, 1]
        composite = min(1.0, sum(scores))

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(
            "ZeroResource: composite=%.3f top_dist=%.4f latency=%.1f ms",
            composite, results[0].distance, elapsed,
        )

        return SafetyScore(
            composite=composite,
            evaluator="zero_resource",
            hits=hits,
            latency_ms=elapsed,
            raw={
                "top_k_results": [
                    {
                        "rank": i,
                        "distance": r.distance,
                        "category": r.category,
                        "label": r.label,
                    }
                    for i, r in enumerate(results)
                ]
            },
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_category(raw: str) -> ThreatCategory:
        try:
            return ThreatCategory(raw)
        except ValueError:
            return ThreatCategory.UNKNOWN

    def get_config(self) -> dict[str, Any]:
        return {
            "evaluator": "zero_resource",
            "model": self._embedder.model_name,
            "top_k": self._top_k,
            "distance_scale": self._scale,
        }
