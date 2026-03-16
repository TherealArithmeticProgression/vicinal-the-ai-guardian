"""
engine/evaluators/embedding_evaluator.py

Thin wrapper around sentence-transformers that encodes text into
fixed-dimensional dense vectors.  Loads the model once and caches it
for the process lifetime.
"""

from __future__ import annotations

import logging
import time
from typing import ClassVar

import numpy as np

from core.errors.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

# Default model: small, fast, good general-purpose semantic similarity.
# 384-dimensional embeddings, ~22M params, MIT license.
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingEvaluator:
    """
    Encodes text (or a list of texts) into L2-normalised numpy vectors.

    This class is intentionally stateless beyond the cached model so that
    it can be shared safely across the engine pipeline.

    Parameters
    ----------
    model_name:
        Any sentence-transformers compatible model identifier.
    device:
        "cpu" | "cuda" | "mps".  Falls back to CPU if the requested
        device is unavailable.
    """

    _instances: ClassVar[dict[str, "EmbeddingEvaluator"]] = {}

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None  # lazy load

    # ------------------------------------------------------------------
    # Singleton-per-model convenience
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> "EmbeddingEvaluator":
        key = f"{model_name}::{device}"
        if key not in cls._instances:
            cls._instances[key] = cls(model_name, device)
        return cls._instances[key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, texts: list[str] | str, batch_size: int = 32) -> np.ndarray:
        """
        Encode one or more strings into L2-normalised float32 vectors.

        Returns
        -------
        np.ndarray
            Shape (N, D) for a list of N texts, or (D,) for a single string.
        """
        if isinstance(texts, str):
            texts = [texts]
            squeeze = True
        else:
            squeeze = False

        try:
            model = self._load_model()
            t0 = time.perf_counter()
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,           # L2-normalise → cosine ≡ dot product
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            logger.debug(
                "Encoded %d text(s) in %.1f ms (model=%s)",
                len(texts), elapsed, self.model_name,
            )
            return embeddings[0] if squeeze else embeddings

        except Exception as exc:
            raise EmbeddingError(f"Failed to encode text: {exc}") from exc

    @property
    def dimension(self) -> int:
        """Embedding dimensionality (e.g. 384 for all-MiniLM-L6-v2)."""
        return self._load_model().get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc

            logger.info("Loading embedding model '%s' on %s …", self.model_name, self.device)
            t0 = time.perf_counter()
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(
                "Model '%s' ready in %.1f s.",
                self.model_name,
                time.perf_counter() - t0,
            )
        return self._model
