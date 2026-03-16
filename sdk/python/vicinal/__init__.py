"""
sdk/python/vicinal/__init__.py

Public surface of the Vicinal SDK.

    from vicinal import VicinalGuard, VicinalConfig, Verdict
"""

from vicinal.config import VicinalConfig
from vicinal.guard import VicinalGuard
from vicinal.verdicts import Verdict, VerdictResult, ThreatCategory, SafetyScore
from vicinal.exceptions import VicinalBlockedError, VicinalWarnError

__version__ = "0.1.0"

__all__ = [
    "VicinalGuard",
    "VicinalConfig",
    "Verdict",
    "VerdictResult",
    "ThreatCategory",
    "SafetyScore",
    "VicinalBlockedError",
    "VicinalWarnError",
    "__version__",
]
