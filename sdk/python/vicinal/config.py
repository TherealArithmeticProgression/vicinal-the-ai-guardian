"""
sdk/python/vicinal/config.py

VicinalConfig — the single configuration object passed to VicinalGuard.
All fields have sensible defaults so the library can be used with zero
configuration for quick experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VicinalConfig:
    """
    Configuration for VicinalGuard.

    Parameters
    ----------
    mode : "zero_resource" | "context_driven" | "full"
        - zero_resource   : FAISS embedding distance only (fastest)
        - context_driven  : OCR + RAG only (richest signal, slower)
        - full            : both passes (recommended for production)

    embedding_model : str
        sentence-transformers model identifier.
        Default: "all-MiniLM-L6-v2" (384-dim, ~22M params, MIT licence).

    device : "cpu" | "cuda" | "mps"
        Device for the embedding model.

    always_deep : bool
        If True, the context-driven evaluator always runs even when the
        zero-resource score is low.  Increases safety coverage at the
        cost of latency.

    block_threshold : float   ∈ [0, 1]
    warn_threshold  : float   ∈ [0, 1]
    redact_threshold: float   ∈ [0, 1]
        Composite score thresholds for each verdict level.

    index_path : str | None
        Path to a custom FAISS index .bin file.
        If None, uses the index shipped with the package.

    meta_path : str | None
        Path to a custom metadata.json companion file.

    kb_path : str | None
        Path to a custom threat knowledge base JSONL file.

    policy_path : str | None
        Path to a YAML policy file that overrides the above thresholds.

    log_level : str
        Python logging level string ("DEBUG", "INFO", "WARNING", "ERROR").

    environment : "dev" | "staging" | "prod"
        Carries through to VerdictResult for audit logging.
    """

    mode: str = "full"
    embedding_model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    always_deep: bool = False

    # Verdict thresholds
    block_threshold:  float = 0.80
    warn_threshold:   float = 0.60
    redact_threshold: float = 0.55

    # Custom data paths
    index_path: str | None = None
    meta_path:  str | None = None
    kb_path:    str | None = None
    policy_path: str | None = None

    # Operational
    log_level:   str = "WARNING"
    environment: str = "prod"
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_policy_dict(self) -> dict[str, Any]:
        """Convert threshold fields to the dict the DecisionMaker expects."""
        overrides: dict[str, Any] = {}

        if self.policy_path:
            overrides = self._load_yaml_policy()

        # Config-level values override YAML (explicit always wins)
        overrides.setdefault("block_threshold",  self.block_threshold)
        overrides.setdefault("warn_threshold",   self.warn_threshold)
        overrides.setdefault("redact_threshold", self.redact_threshold)

        return overrides

    def _load_yaml_policy(self) -> dict[str, Any]:
        try:
            import yaml
        except ImportError:
            return {}

        path = Path(self.policy_path)
        if not path.exists():
            return {}

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data.get("thresholds", data)

    @classmethod
    def zero_resource(cls, **kwargs) -> "VicinalConfig":
        """Factory: zero-resource-only mode."""
        return cls(mode="zero_resource", **kwargs)

    @classmethod
    def context_driven(cls, **kwargs) -> "VicinalConfig":
        """Factory: context-driven-only mode."""
        return cls(mode="context_driven", **kwargs)
