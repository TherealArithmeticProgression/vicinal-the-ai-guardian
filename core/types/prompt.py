"""
core/types/prompt.py

PromptPayload — the canonical input structure that Vicinal evaluates.
Every integration surface normalises its input into this form before
the engine ever touches it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class PromptRole(str, Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class AttachmentType(str, Enum):
    IMAGE = "image"
    PDF = "pdf"
    TEXT = "text"
    AUDIO = "audio"   # reserved for future OCR/ASR path
    UNKNOWN = "unknown"


@dataclass
class Attachment:
    """A file or blob attached alongside a prompt."""
    data: bytes                                   # raw bytes
    type: AttachmentType = AttachmentType.UNKNOWN
    filename: str = ""
    mime_type: str = ""
    extracted_text: str | None = None            # filled by OCR evaluator


@dataclass
class Message:
    """A single turn in a conversation."""
    role: PromptRole
    content: str
    attachments: list[Attachment] = field(default_factory=list)


@dataclass
class PromptPayload:
    """
    Canonical prompt representation passed through the Vicinal engine.

    Parameters
    ----------
    messages:
        One or more conversation turns.  The *last* user-role message is
        treated as the live input being evaluated.
    model_id:
        Identifier of the downstream model (e.g. "gpt-4o", "ollama/mistral").
    user_id:
        Opaque caller identity — used for audit trails only.
    session_id:
        Groups related calls together in telemetry.
    environment:
        "dev" | "staging" | "prod" — influences verdict thresholds.
    metadata:
        Arbitrary key-value pairs from the caller (endpoint purpose, user
        role, etc.).  Vicinal never modifies these; they are passed through
        to the context builder for policy evaluation.
    """

    messages: list[Message]
    model_id: str = "unknown"
    user_id: str = "anonymous"
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    environment: str = "prod"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def active_message(self) -> Message | None:
        """The last user-role message — the one being evaluated."""
        for msg in reversed(self.messages):
            if msg.role == PromptRole.USER:
                return msg
        return None

    @property
    def active_text(self) -> str:
        """Plain text of the active user message, or empty string."""
        msg = self.active_message
        return msg.content if msg else ""

    @property
    def all_attachments(self) -> list[Attachment]:
        """All attachments across every message turn."""
        return [a for msg in self.messages for a in msg.attachments]

    @classmethod
    def from_text(cls, text: str, **kwargs: Any) -> "PromptPayload":
        """Shortcut: create a single-turn user payload from a plain string."""
        return cls(
            messages=[Message(role=PromptRole.USER, content=text)],
            **kwargs,
        )
