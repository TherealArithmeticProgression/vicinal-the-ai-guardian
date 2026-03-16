"""
engine/engine.py

VicinalEngine — the main pipeline orchestrator.

Execution order
---------------
1. Build EvaluationContext from the PromptPayload.
2. Run ZeroResourceEvaluator  (always — fast, no network).
3. Run ContextDrivenEvaluator (when: attachments present OR zero_resource
   score is above the "deep-check" threshold, or always-on mode is set).
4. Collect SafetyScores.
5. Pass to DecisionMaker → VerdictResult.

Thread safety
-------------
All evaluator instances are stateless after initialisation, so a single
VicinalEngine object can safely serve concurrent requests.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from core.context.context_builder import ContextBuilder
from core.decisions.decision_maker import DecisionMaker
from core.errors.exceptions import EvaluationError
from core.types.prompt import PromptPayload
from core.types.safety_score import SafetyScore
from core.types.verdict import VerdictResult
from engine.evaluators.embedding_evaluator import EmbeddingEvaluator
from engine.evaluators.faiss_evaluator import FaissEvaluator
from engine.evaluators.ocr_evaluator import OCREvaluator
from engine.evaluators.rag_evaluator import RAGEvaluator
from engine.pre_call.context_driven import ContextDrivenEvaluator
from engine.pre_call.zero_resource import ZeroResourceEvaluator

logger = logging.getLogger(__name__)

# Score above which context-driven pass is also triggered even without attachments
DEEP_CHECK_TRIGGER = 0.45


class VicinalEngine:
    """
    Orchestrates the full Vicinal evaluation pipeline.

    Parameters
    ----------
    policy:
        Dict of policy thresholds and settings from VicinalConfig.
    always_deep:
        If True, always run both evaluators regardless of zero-resource score.
    embedding_model:
        sentence-transformers model name.
    device:
        "cpu" | "cuda" | "mps"
    index_path / meta_path / kb_path:
        Override paths for FAISS index, metadata, and knowledge base.
    """

    def __init__(
        self,
        policy: dict[str, Any] | None = None,
        always_deep: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        index_path: str | None = None,
        meta_path: str | None = None,
        kb_path: str | None = None,
    ) -> None:
        self._policy     = policy or {}
        self._always_deep = always_deep

        # Shared, lazily-loaded evaluation components
        self._embedder = EmbeddingEvaluator.get(model_name=embedding_model, device=device)
        self._faiss    = FaissEvaluator(index_path=index_path, meta_path=meta_path)
        self._ocr      = OCREvaluator()
        self._rag      = RAGEvaluator(
            kb_path=kb_path,
            embedding_evaluator=self._embedder,
            faiss_evaluator=self._faiss,
        )

        self._zero_resource = ZeroResourceEvaluator(
            embedding_evaluator=self._embedder,
            faiss_evaluator=self._faiss,
        )
        self._context_driven = ContextDrivenEvaluator(
            ocr_evaluator=self._ocr,
            rag_evaluator=self._rag,
            include_conversation_history=bool(self._policy.get("full_history", False)),
        )

        self._decision_maker = DecisionMaker(policy=self._policy)

        logger.info(
            "VicinalEngine initialised (always_deep=%s, model=%s, device=%s)",
            always_deep, embedding_model, device,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def evaluate(self, payload: PromptPayload) -> VerdictResult:
        """
        Run the full evaluation pipeline on a prompt payload.

        Parameters
        ----------
        payload:
            Canonical PromptPayload.

        Returns
        -------
        VerdictResult
        """
        t_total = time.perf_counter()

        context = ContextBuilder.build(
            payload,
            policy_overrides=self._policy.get("overrides", {}),
        )

        scores: list[SafetyScore] = []

        # ----------------------------------------------------------------
        # Pass 1 — zero-resource (always)
        # ----------------------------------------------------------------
        try:
            zr_score = self._zero_resource.evaluate(context)
            scores.append(zr_score)
            logger.info(
                "[%s] ZeroResource score=%.3f",
                payload.session_id, zr_score.composite,
            )
        except EvaluationError as exc:
            logger.error("ZeroResource evaluator failed: %s", exc)
            zr_score = None

        # ----------------------------------------------------------------
        # Pass 2 — context-driven (conditional)
        # ----------------------------------------------------------------
        run_deep = (
            self._always_deep
            or context.has_attachments
            or (zr_score is not None and zr_score.composite >= DEEP_CHECK_TRIGGER)
        )

        if run_deep:
            try:
                cd_score = self._context_driven.evaluate(context, payload=payload)
                scores.append(cd_score)
                logger.info(
                    "[%s] ContextDriven score=%.3f",
                    payload.session_id, cd_score.composite,
                )
            except EvaluationError as exc:
                logger.error("ContextDriven evaluator failed: %s", exc)
        else:
            logger.debug("[%s] ContextDriven skipped (low ZR score).", payload.session_id)

        # ----------------------------------------------------------------
        # Decision
        # ----------------------------------------------------------------
        result = self._decision_maker.decide(
            scores=scores,
            session_id=payload.session_id,
            metadata=payload.metadata,
        )

        total_ms = (time.perf_counter() - t_total) * 1000
        logger.info(
            "[%s] VERDICT=%s score=%.3f total_latency=%.1f ms",
            payload.session_id,
            result.verdict.value,
            result.composite_score,
            total_ms,
        )

        return result

    def warmup(self) -> None:
        """
        Pre-load all lazy resources (model, index, BM25) so that the first
        real evaluation is not slow.  Call this at application startup.
        """
        logger.info("Warming up VicinalEngine …")
        dummy = PromptPayload.from_text("warmup")
        try:
            self.evaluate(dummy)
            logger.info("Warmup complete.")
        except Exception as exc:
            logger.warning("Warmup finished with non-fatal error: %s", exc)
