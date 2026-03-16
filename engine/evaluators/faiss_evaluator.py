"""
engine/evaluators/faiss_evaluator.py

FAISS index wrapper for kNN threat-pattern retrieval.

The index ships pre-built inside the package (data/faiss_index.bin).
Callers can also supply a custom index path via VicinalConfig.

Index format
------------
  - Flat L2 index (exact search; suitable for ≤ 100k vectors)
  - Float32 vectors, dimension = embedding model output dim
  - Metadata (labels, category, text) stored in data/metadata.json

For larger deployments a FAISS IVF or HNSW index can be dropped in as
a direct replacement — the API is unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

from core.errors.exceptions import IndexCorruptedError, IndexNotFoundError

logger = logging.getLogger(__name__)

# Shipped index location (relative to repo root)
_DEFAULT_INDEX = Path(__file__).resolve().parents[2] / "data" / "faiss_index.bin"
_DEFAULT_META  = Path(__file__).resolve().parents[2] / "data" / "metadata.json"


class SearchResult(NamedTuple):
    index:    int    # position in the FAISS index
    distance: float  # L2 distance (lower = more similar)
    label:    str    # threat label from metadata
    category: str    # ThreatCategory value
    text:     str    # original threat pattern text


class FaissEvaluator:
    """
    Loads and queries the pre-built FAISS threat index.

    Parameters
    ----------
    index_path:
        Path to the .bin FAISS index file.
    meta_path:
        Path to the companion metadata.json file.
    """

    def __init__(
        self,
        index_path: str | Path | None = None,
        meta_path:  str | Path | None = None,
    ) -> None:
        self._index_path = Path(index_path or _DEFAULT_INDEX)
        self._meta_path  = Path(meta_path  or _DEFAULT_META)
        self._index = None
        self._meta: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, vector: np.ndarray, k: int = 5) -> list[SearchResult]:
        """
        Find the k nearest threat patterns to a given embedding vector.

        Parameters
        ----------
        vector:
            1-D float32 array (already L2-normalised by EmbeddingEvaluator).
        k:
            Number of nearest neighbours to return.

        Returns
        -------
        List of SearchResult sorted by ascending distance (closest first).
        """
        index = self._load_index()
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)

        t0 = time.perf_counter()
        distances, indices = index.search(vector, k)
        elapsed = (time.perf_counter() - t0) * 1000

        logger.debug("FAISS search returned in %.2f ms", elapsed)

        results: list[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:          # FAISS returns -1 for empty slots
                continue
            meta = self._meta[idx] if idx < len(self._meta) else {}
            results.append(
                SearchResult(
                    index=int(idx),
                    distance=float(dist),
                    label=meta.get("label", "unknown"),
                    category=meta.get("category", "unknown"),
                    text=meta.get("text", ""),
                )
            )
        return results

    def search_batch(self, vectors: np.ndarray, k: int = 5) -> list[list[SearchResult]]:
        """Batch version of search — one result list per input vector."""
        index = self._load_index()
        vectors = np.asarray(vectors, dtype=np.float32)
        distances, indices = index.search(vectors, k)

        batch: list[list[SearchResult]] = []
        for row_d, row_i in zip(distances, indices):
            row_results: list[SearchResult] = []
            for dist, idx in zip(row_d, row_i):
                if idx < 0:
                    continue
                meta = self._meta[idx] if idx < len(self._meta) else {}
                row_results.append(
                    SearchResult(
                        index=int(idx),
                        distance=float(dist),
                        label=meta.get("label", "unknown"),
                        category=meta.get("category", "unknown"),
                        text=meta.get("text", ""),
                    )
                )
            batch.append(row_results)
        return batch

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self._load_index().ntotal

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_index(self):
        if self._index is not None:
            return self._index

        try:
            import faiss
        except ImportError as exc:
            raise IndexNotFoundError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu"
            ) from exc

        if not self._index_path.exists():
            raise IndexNotFoundError(
                f"FAISS index not found at {self._index_path}. "
                "Run: python data/build_index.py to build it."
            )

        try:
            logger.info("Loading FAISS index from %s …", self._index_path)
            self._index = faiss.read_index(str(self._index_path))
            logger.info("Index loaded: %d vectors, dim=%d", self._index.ntotal, self._index.d)
        except Exception as exc:
            raise IndexCorruptedError(
                f"Failed to load FAISS index: {exc}"
            ) from exc

        # Load metadata
        if self._meta_path.exists():
            with open(self._meta_path, encoding="utf-8") as fh:
                self._meta = json.load(fh)
            logger.debug("Metadata loaded: %d entries", len(self._meta))
        else:
            logger.warning("Metadata file not found at %s — labels will be empty.", self._meta_path)

        return self._index
