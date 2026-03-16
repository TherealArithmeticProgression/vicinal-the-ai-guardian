"""
engine/pre_call/context_driven.py

Context-Driven Pre-Call Evaluator.

Combines OCR (for attachments) + RAG retrieval to produce a SafetyScore
that is richer than the zero-resource path — especially for:

  - Image-based prompt injections (text embedded in screenshots/photos)
  - PDF-based indirect injections (malicious RAG documents)
  - Paraphrased or obfuscated attacks that evade embedding similarity

Pipeline
--------
1. If the payload has attachments: run OCR → extract text from each.
2. Concatenate [prompt_text + ocr_text] as the unified query.
3. Run RAG retrieval + scoring against the threat knowledge base.
4. Return a SafetyScore.
"""

from __future__ import annotations

import logging
import time

from core.context.context_builder import EvaluationContext
from core.errors.exceptions import EvaluationError
from core.types.prompt import PromptPayload
from core.types.safety_score import SafetyScore
from engine.evaluators.ocr_evaluator import OCREvaluator
from engine.evaluators.rag_evaluator import RAGEvaluator

logger = logging.getLogger(__name__)


class ContextDrivenEvaluator:
    """
    Evaluates a prompt using OCR + RAG.

    Parameters
    ----------
    ocr_evaluator:
        Shared OCREvaluator instance.
    rag_evaluator:
        Shared RAGEvaluator instance.
    include_conversation_history:
        If True, the full conversation (not just active message) is
        included in the RAG query.  Useful for catching multi-turn
        injection attacks.
    """

    def __init__(
        self,
        ocr_evaluator: OCREvaluator | None = None,
        rag_evaluator: RAGEvaluator | None = None,
        include_conversation_history: bool = False,
    ) -> None:
        self._ocr = ocr_evaluator or OCREvaluator()
        self._rag = rag_evaluator or RAGEvaluator()
        self._full_history = include_conversation_history

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def evaluate(
        self,
        context: EvaluationContext,
        payload: PromptPayload | None = None,
    ) -> SafetyScore:
        """
        Run the context-driven evaluation pass.

        Parameters
        ----------
        context:
            EvaluationContext built by ContextBuilder.
        payload:
            Original PromptPayload — required if OCR on attachments is needed.

        Returns
        -------
        SafetyScore
        """
        t0 = time.perf_counter()

        query_parts: list[str] = []

        # ---- Base text -------------------------------------------------
        if self._full_history:
            query_parts.append(context.full_conversation)
        else:
            query_parts.append(context.prompt_text)

        # ---- OCR pass --------------------------------------------------
        if context.has_attachments and payload is not None:
            try:
                ocr_text = self._ocr.extract_all(payload.all_attachments)
                if ocr_text.strip():
                    logger.debug(
                        "OCR extracted %d chars from %d attachment(s).",
                        len(ocr_text),
                        len(payload.all_attachments),
                    )
                    query_parts.append(f"[ATTACHMENT CONTENT]\n{ocr_text}")
            except Exception as exc:
                logger.warning("OCR stage failed (non-fatal): %s", exc)

        unified_query = "\n\n".join(filter(None, query_parts))

        if not unified_query.strip():
            return SafetyScore(
                composite=0.0,
                evaluator="context_driven",
                latency_ms=(time.perf_counter() - t0) * 1000,
                raw={"reason": "empty query after OCR"},
            )

        # ---- RAG scoring -----------------------------------------------
        try:
            rag_score = self._rag.score(unified_query)
        except Exception as exc:
            raise EvaluationError(str(exc), evaluator="context_driven") from exc

        # Relabel evaluator field
        rag_score.evaluator = "context_driven"
        rag_score.latency_ms = (time.perf_counter() - t0) * 1000
        rag_score.raw["ocr_included"] = context.has_attachments

        logger.debug(
            "ContextDriven: composite=%.3f latency=%.1f ms",
            rag_score.composite,
            rag_score.latency_ms,
        )

        return rag_score
