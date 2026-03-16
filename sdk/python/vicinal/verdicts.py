"""
sdk/python/vicinal/verdicts.py

Re-exports the Verdict enum and VerdictResult from core so that users
only need to import from 'vicinal' — not from the internal core package.
"""

from core.types.verdict import Verdict, VerdictResult
from core.types.safety_score import ThreatCategory, ThreatHit, SafetyScore

__all__ = [
    "Verdict",
    "VerdictResult",
    "ThreatCategory",
    "ThreatHit",
    "SafetyScore",
]
