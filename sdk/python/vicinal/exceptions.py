"""
sdk/python/vicinal/exceptions.py

User-facing exception types for consumers of the Vicinal SDK.
Internal exceptions (VicinalError and subclasses) are in core/errors/.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.types.verdict import VerdictResult


class VicinalBlockedError(Exception):
    """
    Raised by the @guard.protect decorator when a prompt is BLOCKED.

    Attributes
    ----------
    result : VerdictResult
        The full evaluation result that triggered the block.
    """

    def __init__(self, result: "VerdictResult") -> None:
        self.result = result
        super().__init__(
            f"Prompt blocked by Vicinal (score={result.composite_score:.3f}): "
            f"{result.reason}"
        )


class VicinalWarnError(Exception):
    """
    Optional: can be raised on WARN verdicts if the caller wants strict mode.
    """

    def __init__(self, result: "VerdictResult") -> None:
        self.result = result
        super().__init__(
            f"Prompt flagged by Vicinal with WARN (score={result.composite_score:.3f}): "
            f"{result.reason}"
        )
