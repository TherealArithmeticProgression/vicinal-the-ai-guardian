"""
engine/evaluators/rag_evaluator.py

RAG Evaluator — retrieves threat-relevant context and scores the prompt
against a threat knowledge base using BM25 + semantic similarity.

Two-stage retrieval
-------------------
Stage 1 (BM25):   Fast lexical recall — catches keyword-heavy attacks
                  like "ignore previous instructions".
Stage 2 (FAISS):  Semantic recall — catches paraphrased/obfuscated attacks.

The two result sets are merged, deduplicated, and scored.  Each retrieved
document contributes to a category-weighted composite threat score.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.errors.exceptions import RAGError
from core.types.safety_score import SafetyScore, ThreatCategory, ThreatHit, CATEGORY_SEVERITY
from engine.evaluators.embedding_evaluator import EmbeddingEvaluator
from engine.evaluators.faiss_evaluator import FaissEvaluator

logger = logging.getLogger(__name__)

_DEFAULT_KB = Path(__file__).resolve().parents[2] / "data" / "threat_patterns.jsonl"

# Number of documents per retrieval stage
BM25_TOP_K   = 10
FAISS_TOP_K  = 10

# Minimum BM25 / similarity score to include a doc in scoring
BM25_MIN_SCORE   = 0.5
FAISS_MIN_SCORE  = 0.45   # corresponds to distance ≤ DISTANCE_SCALE * 0.55


class RAGEvaluator:
    """
    Retrieves relevant threat documents and produces a semantic safety score.

    Parameters
    ----------
    kb_path:
        Path to the threat knowledge base JSONL file.
    embedding_evaluator:
        Shared EmbeddingEvaluator instance.
    faiss_evaluator:
        Shared FaissEvaluator instance (same index as zero-resource path).
    """

    def __init__(
        self,
        kb_path: str | Path | None = None,
        embedding_evaluator: EmbeddingEvaluator | None = None,
        faiss_evaluator: FaissEvaluator | None = None,
    ) -> None:
        self._kb_path  = Path(kb_path or _DEFAULT_KB)
        self._embedder = embedding_evaluator or EmbeddingEvaluator.get()
        self._faiss    = faiss_evaluator    or FaissEvaluator()
        self._bm25     = None   # lazy load
        self._corpus   = None   # parallel list of threat documents

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def score(self, query: str) -> SafetyScore:
        """
        Score a query string against the threat knowledge base.

        Parameters
        ----------
        query:
            The combined prompt + OCR text to evaluate.

        Returns
        -------
        SafetyScore
        """
        t0 = time.perf_counter()

        if not query.strip():
            return SafetyScore(composite=0.0, evaluator="rag",
                               latency_ms=0.0, raw={"reason": "empty query"})

        # Stage 1: BM25
        bm25_hits = self._bm25_retrieve(query)

        # Stage 2: FAISS semantic
        try:
            vec = self._embedder.encode(query)
            faiss_results = self._faiss.search(vec, k=FAISS_TOP_K)
        except Exception as exc:
            logger.warning("FAISS retrieval failed in RAG: %s", exc)
            faiss_results = []

        # Merge + deduplicate
        all_docs = self._merge(bm25_hits, faiss_results)

        if not all_docs:
            return SafetyScore(composite=0.0, evaluator="rag",
                               latency_ms=(time.perf_counter() - t0) * 1000,
                               raw={"reason": "no relevant documents retrieved"})

        # Score
        hits, composite = self._compute_score(all_docs)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("RAG: composite=%.3f docs=%d latency=%.1f ms", composite, len(all_docs), elapsed)

        return SafetyScore(
            composite=composite,
            evaluator="rag",
            hits=hits,
            latency_ms=elapsed,
            raw={"retrieved_doc_count": len(all_docs)},
        )

    # ------------------------------------------------------------------
    # Private — BM25
    # ------------------------------------------------------------------

    def _bm25_retrieve(self, query: str) -> list[dict[str, Any]]:
        corpus, bm25 = self._load_bm25()
        if not corpus or bm25 is None:
            return []

        try:
            tokens = query.lower().split()
            scores = bm25.get_scores(tokens)
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:BM25_TOP_K]

            results = []
            for idx in top_indices:
                bm25_score = float(scores[idx])
                if bm25_score < BM25_MIN_SCORE:
                    continue
                doc = corpus[idx].copy()
                doc["retrieval_score"] = bm25_score
                doc["retrieval_method"] = "bm25"
                results.append(doc)
            return results

        except Exception as exc:
            logger.warning("BM25 retrieval error: %s", exc)
            return []

    def _load_bm25(self) -> tuple[list[dict], Any]:
        if self._corpus is not None:
            return self._corpus, self._bm25

        if not self._kb_path.exists():
            logger.warning("Threat KB not found at %s — RAG disabled.", self._kb_path)
            self._corpus = []
            return self._corpus, None

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank-bm25 not installed — BM25 stage skipped.")
            self._corpus = []
            return self._corpus, None

        self._corpus = []
        with open(self._kb_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._corpus.append(json.loads(line))

        tokenised = [doc["text"].lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenised)
        logger.info("BM25 index built from %d KB documents.", len(self._corpus))
        return self._corpus, self._bm25

    # ------------------------------------------------------------------
    # Private — merging & scoring
    # ------------------------------------------------------------------

    def _merge(self, bm25_hits: list[dict], faiss_results) -> list[dict]:
        """Deduplicate by index id and keep the highest score for each."""
        seen: dict[int, dict] = {}

        for doc in bm25_hits:
            idx = doc.get("id", id(doc))
            seen[idx] = doc

        for fa in faiss_results:
            sim_score = max(0.0, 1.0 - fa.distance / 1.2)
            if sim_score < FAISS_MIN_SCORE:
                continue
            if fa.index not in seen:
                seen[fa.index] = {
                    "id": fa.index,
                    "text": fa.text,
                    "category": fa.category,
                    "label": fa.label,
                    "retrieval_score": sim_score,
                    "retrieval_method": "faiss",
                }
            else:
                # Take max of both scores
                seen[fa.index]["retrieval_score"] = max(
                    seen[fa.index]["retrieval_score"], sim_score
                )

        return list(seen.values())

    def _compute_score(self, docs: list[dict]) -> tuple[list[ThreatHit], float]:
        from core.types.safety_score import CATEGORY_SEVERITY

        category_scores: dict[ThreatCategory, float] = {}
        hits: list[ThreatHit] = []

        for doc in docs:
            raw_score = float(doc.get("retrieval_score", 0.0))
            cat = self._parse_category(doc.get("category", "unknown"))
            severity = CATEGORY_SEVERITY.get(cat, 0.5)
            weighted = raw_score * severity

            # Accumulate per category (take max, not sum, to avoid inflation)
            category_scores[cat] = max(category_scores.get(cat, 0.0), weighted)

            hits.append(
                ThreatHit(
                    category=cat,
                    score=weighted,
                    evidence=(
                        f"Retrieved via {doc.get('retrieval_method', '?')} "
                        f"(score={raw_score:.3f}): {doc.get('label', '')}"
                    ),
                    source="rag",
                    matched_pattern=doc.get("text", "")[:200],
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)

        # Composite = weighted sum of top categories, clipped to [0, 1]
        composite = min(1.0, sum(category_scores.values()))
        return hits, composite

    @staticmethod
    def _parse_category(raw: str) -> ThreatCategory:
        try:
            return ThreatCategory(raw)
        except ValueError:
            return ThreatCategory.UNKNOWN
