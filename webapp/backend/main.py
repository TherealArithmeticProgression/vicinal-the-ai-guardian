"""
webapp/backend/main.py

FastAPI demo application for Vicinal.

Endpoints
---------
POST /api/chat
    Evaluate a prompt with Vicinal, then (if ALLOW/WARN) forward to
    the configured model (Ollama by default) and return the response.

POST /api/guard/evaluate
    Evaluate a prompt with Vicinal only — no model call.
    Returns the full VerdictResult as JSON.

GET  /api/health
    Returns the guard's status and configuration.

GET  /api/config
    Returns human-readable engine configuration.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---- path bootstrap ---------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
SDK_PATH  = REPO_ROOT / "sdk" / "python"
for p in [str(REPO_ROOT), str(SDK_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)
# -----------------------------------------------------------------------

from webapp.backend.routes.chat   import router as chat_router
from webapp.backend.routes.guard  import router as guard_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vicinal.webapp")


# ---------------------------------------------------------------------- #
# Lifespan — warmup on startup
# ---------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up VicinalGuard …")
    try:
        from webapp.backend.routes.guard import get_guard
        guard = get_guard()
        guard.warmup()
        logger.info("VicinalGuard ready.")
    except Exception as exc:
        logger.warning("Warmup failed (non-fatal): %s", exc)
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------- #
# App
# ---------------------------------------------------------------------- #

app = FastAPI(
    title="Vicinal — AI Guardian Demo",
    description=(
        "Live demonstration of the Vicinal guardrail SDK. "
        "Prompts are evaluated for safety before being forwarded to the model."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router,  prefix="/api")
app.include_router(guard_router, prefix="/api")


@app.get("/api/health", tags=["meta"])
async def health():
    return {"status": "ok", "version": "0.1.0"}
