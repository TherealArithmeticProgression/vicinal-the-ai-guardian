"""
core/errors/exceptions.py

Vicinal exception hierarchy.
All public-facing errors inherit from VicinalError so callers can catch
the whole family with a single except clause.
"""


class VicinalError(Exception):
    """Base exception for all Vicinal errors."""


# ------------------------------------------------------------------
# Configuration errors
# ------------------------------------------------------------------

class VicinalConfigError(VicinalError):
    """Raised when the VicinalConfig is invalid or incomplete."""


class PolicyNotFoundError(VicinalConfigError):
    """Raised when a referenced policy file cannot be located."""


class PolicyValidationError(VicinalConfigError):
    """Raised when a policy file fails schema validation."""


# ------------------------------------------------------------------
# Index / embedding errors
# ------------------------------------------------------------------

class IndexNotFoundError(VicinalError):
    """Raised when the FAISS index file is missing."""


class IndexCorruptedError(VicinalError):
    """Raised when the FAISS index file cannot be loaded or is corrupted."""


class EmbeddingError(VicinalError):
    """Raised when the embedding model fails to encode a prompt."""


# ------------------------------------------------------------------
# OCR errors
# ------------------------------------------------------------------

class OCRError(VicinalError):
    """Raised when OCR processing of an attachment fails."""


class UnsupportedAttachmentError(OCRError):
    """Raised when an attachment type is not supported by the OCR evaluator."""


# ------------------------------------------------------------------
# RAG errors
# ------------------------------------------------------------------

class RAGError(VicinalError):
    """Raised when the RAG retrieval pipeline encounters an error."""


# ------------------------------------------------------------------
# Adapter errors
# ------------------------------------------------------------------

class AdapterError(VicinalError):
    """Raised when a model adapter fails to normalise a request/response."""


class ModelUnavailableError(AdapterError):
    """Raised when the downstream model endpoint cannot be reached."""


# ------------------------------------------------------------------
# Engine errors
# ------------------------------------------------------------------

class EvaluationError(VicinalError):
    """Raised when the engine pipeline fails during evaluation."""

    def __init__(self, message: str, evaluator: str = "") -> None:
        super().__init__(message)
        self.evaluator = evaluator
