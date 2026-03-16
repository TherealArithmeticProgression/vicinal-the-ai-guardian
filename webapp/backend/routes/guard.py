"""
webapp/backend/routes/guard.py

/api/guard/evaluate — standalone guard evaluation endpoint.

This endpoint runs the full Vicinal pipeline and returns the verdict
without making any model call. Useful for inspection, testing, and the
"guard only" mode in the frontend.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

SDK_PATH = Path(__file__).resolve().parents[4] / "sdk" / "python"
if str(SDK_PATH) not in sys.path:
    sys.path.insert(0, str(SDK_PATH))

router = APIRouter(tags=["guard"])


# ---------------------------------------------------------------------- #
# Request / Response schemas
# ---------------------------------------------------------------------- #

class EvaluateRequest(BaseModel):
    prompt: str
    user_id: str = "anonymous"
    session_id: str | None = None
    metadata: dict = {}


class HitSchema(BaseModel):
    category: str
    score: float
    evidence: str


class EvaluatorSchema(BaseModel):
    name: str
    score: float
    latency_ms: float
    hits: list[HitSchema]


class EvaluateResponse(BaseModel):
    verdict: str
    composite_score: float
    reason: str
    top_category: str
    session_id: str
    evaluators: list[EvaluatorSchema]


# ---------------------------------------------------------------------- #
# Guard singleton
# ---------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def get_guard():
    from vicinal import VicinalGuard, VicinalConfig

    mode = os.environ.get("VICINAL_MODE", "full")
    cfg = VicinalConfig(
        mode=mode,
        log_level="INFO",
        environment=os.environ.get("VICINAL_ENV", "prod"),
        index_path=os.environ.get("VICINAL_INDEX_PATH") or None,
        block_threshold=float(os.environ.get("VICINAL_BLOCK_THRESHOLD", "0.80")),
        warn_threshold=float(os.environ.get("VICINAL_WARN_THRESHOLD", "0.60")),
    )
    return VicinalGuard(cfg)


# ---------------------------------------------------------------------- #
# Route
# ---------------------------------------------------------------------- #

@router.post("/guard/evaluate", response_model=EvaluateResponse)
async def evaluate_prompt(req: EvaluateRequest):
    guard = get_guard()

    try:
        result = guard.evaluate(
            prompt=req.prompt,
            session_id=req.session_id,
            user_id=req.user_id,
            metadata=req.metadata,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Guard evaluation failed: {exc}")

    d = result.to_dict()
    return EvaluateResponse(
        verdict=d["verdict"],
        composite_score=d["composite_score"],
        reason=d["reason"],
        top_category=d["top_category"],
        session_id=d["session_id"],
        evaluators=[
            EvaluatorSchema(
                name=e["name"],
                score=e["score"],
                latency_ms=e["latency_ms"],
                hits=[HitSchema(**h) for h in e["hits"]],
            )
            for e in d["evaluators"]
        ],
    )


@router.get("/guard/config", tags=["guard"])
async def guard_config():
    guard = get_guard()
    return {
        "mode": guard.config.mode,
        "embedding_model": guard.config.embedding_model,
        "device": guard.config.device,
        "thresholds": {
            "block":  guard.config.block_threshold,
            "warn":   guard.config.warn_threshold,
            "redact": guard.config.redact_threshold,
        },
        "environment": guard.config.environment,
    }
