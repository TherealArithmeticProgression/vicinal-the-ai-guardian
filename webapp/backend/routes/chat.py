"""
webapp/backend/routes/chat.py

/api/chat — full chat endpoint with Vicinal guardrail.

Flow:
  1. Evaluate prompt with VicinalGuard
  2. If BLOCK → return 403 with verdict details (model never called)
  3. If WARN  → annotate the response but still call the model
  4. If ALLOW / REDACT → call the model, return response

Model backend: Ollama (default) — swap out via VICINAL_MODEL_BACKEND env var.
  - "ollama"   : local Ollama server (http://localhost:11434)
  - "echo"     : echo mode for testing (no model required)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

SDK_PATH = Path(__file__).resolve().parents[4] / "sdk" / "python"
if str(SDK_PATH) not in sys.path:
    sys.path.insert(0, str(SDK_PATH))

from webapp.backend.routes.guard import get_guard, EvaluateResponse, EvaluatorSchema, HitSchema

router = APIRouter(tags=["chat"])

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
MODEL_BACKEND = os.environ.get("VICINAL_MODEL_BACKEND", "ollama")


# ---------------------------------------------------------------------- #
# Schemas
# ---------------------------------------------------------------------- #

class ChatMessage(BaseModel):
    role: str   # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    user_id: str = "anonymous"
    session_id: str | None = None
    metadata: dict = {}


class GuardInfo(BaseModel):
    verdict: str
    composite_score: float
    reason: str
    top_category: str
    evaluators: list[EvaluatorSchema]


class ChatResponse(BaseModel):
    reply: str | None           # None if blocked
    blocked: bool
    warned: bool
    guard: GuardInfo


# ---------------------------------------------------------------------- #
# Route
# ---------------------------------------------------------------------- #

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    guard = get_guard()

    # Use the last user message as the prompt to evaluate
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found.")

    active_prompt = user_messages[-1].content

    # ---- Guard evaluation ----
    try:
        result = guard.evaluate(
            prompt=active_prompt,
            session_id=req.session_id,
            user_id=req.user_id,
            metadata=req.metadata,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Guard error: {exc}")

    d = result.to_dict()
    guard_info = GuardInfo(
        verdict=d["verdict"],
        composite_score=d["composite_score"],
        reason=d["reason"],
        top_category=d["top_category"],
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

    if result.is_blocked:
        return ChatResponse(
            reply=None,
            blocked=True,
            warned=False,
            guard=guard_info,
        )

    is_warn = (d["verdict"] == "WARN")

    # ---- Model call ----
    reply = await _call_model(req.messages, result.redacted_text or active_prompt)

    return ChatResponse(
        reply=reply,
        blocked=False,
        warned=is_warn,
        guard=guard_info,
    )


# ---------------------------------------------------------------------- #
# Model backends
# ---------------------------------------------------------------------- #

async def _call_model(messages: list[ChatMessage], active_prompt: str) -> str:
    if MODEL_BACKEND == "echo":
        return f"[Echo mode] You said: {active_prompt}"

    if MODEL_BACKEND == "ollama":
        return await _ollama(messages)

    return "[Unknown model backend configured.]"


async def _ollama(messages: list[ChatMessage]) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
    except httpx.ConnectError:
        return (
            "[Ollama is not running. Start it with `ollama serve` and pull a model: "
            f"`ollama pull {OLLAMA_MODEL}`]"
        )
    except Exception as exc:
        return f"[Model error: {exc}]"
