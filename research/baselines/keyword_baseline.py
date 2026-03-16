"""
research/baselines/keyword_baseline.py

Baseline 1: Keyword Matching.

Flags prompts that contain known attack keywords/phrases.
This is the simplest possible baseline — it catches obvious attacks but
is trivially bypassed with any paraphrase.

Expected behaviour in research:
  - High precision on exact-match attacks
  - Very low recall on paraphrased/obfuscated attacks
  - Near-zero latency
"""

from __future__ import annotations

import re
import time

from research.baselines.base_evaluator import BaseEvaluator, BaselineResult

# Curated set of high-signal attack keywords and phrases
ATTACK_KEYWORDS: list[str] = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard all prior",
    "forget everything",
    "your new instructions",
    "override.*instructions",
    "system prompt",
    "repeat your instructions",
    "reveal your prompt",
    r"\bDAN\b",
    r"\bSTAN\b",
    "developer mode",
    "no restrictions",
    "without restrictions",
    "bypass.*filter",
    "disable.*safety",
    "safety.*disabled",
    "ignore.*guidelines",
    "act as.*unrestricted",
    "pretend.*no.*restriction",
    "you are now",                       # common persona-injection opener
    "jailbreak",
    "sudo",
    "root access",
    "admin.*access",
    "grant.*access",
    "emergency protocol",
    "token limit",
    "dump.*system",
    "dump.*context",
    "dump.*memory",
    "exfiltrate",
    "api key",
    "secret key",
    "password.*list",
    "credit card",
    "social security",
    r"\bPII\b",
    "email address.*of.*user",
    "home address",
    "synthesise.*methamphetamine",
    "drug synthesis",
    "explosive device",
    "improvised explosive",
    "malware",
    "ransomware",
    "keylogger",
    "phishing",
    "sql injection",
    r"\bXSS\b",
    "man-in-the-middle",
    "bioweapon",
]

_COMPILED = [re.compile(kw, re.IGNORECASE) for kw in ATTACK_KEYWORDS]


class KeywordBaseline(BaseEvaluator):
    """
    Rule-based keyword matching baseline.

    Score = fraction of keyword groups matched (normalised to [0, 1]).
    Label = 1 if any keyword matches.
    """

    name = "keyword_matching"
    threshold = 0.0   # any match → threat

    def evaluate(self, prompt: str) -> BaselineResult:
        t0 = time.perf_counter()

        matches: list[str] = []
        for pattern in _COMPILED:
            if pattern.search(prompt):
                matches.append(pattern.pattern)

        label = 1 if matches else 0
        score = min(1.0, len(matches) / max(1, len(_COMPILED) / 10))

        return BaselineResult(
            baseline_name=self.name,
            label=label,
            score=score,
            latency_ms=(time.perf_counter() - t0) * 1000,
            metadata={"matched_patterns": matches[:5]},
        )
