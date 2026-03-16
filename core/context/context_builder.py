"""
core/context/context_builder.py

Builds an EvaluationContext from a PromptPayload and VicinalConfig.
The context is passed to every evaluator so it can make policy-aware
decisions without knowing about the raw payload structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.types.prompt import PromptPayload


@dataclass
class EvaluationContext:
    """
    Immutable context object injected into every evaluator.

    Attributes
    ----------
    prompt_text:
        Active user message text.
    full_conversation:
        All message contents concatenated for holistic analysis.
    has_attachments:
        True if any message contains file attachments.
    environment:
        "dev" | "staging" | "prod"
    model_id:
        Downstream model identifier.
    user_id:
        Opaque caller identity.
    session_id:
        Session identifier for audit.
    policy_overrides:
        Any caller-supplied threshold overrides.
    metadata:
        Pass-through metadata from the payload.
    """

    prompt_text: str
    full_conversation: str
    has_attachments: bool
    environment: str
    model_id: str
    user_id: str
    session_id: str
    policy_overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_production(self) -> bool:
        return self.environment == "prod"


class ContextBuilder:
    """Constructs an EvaluationContext from a PromptPayload."""

    @staticmethod
    def build(payload: "PromptPayload", policy_overrides: dict[str, Any] | None = None) -> EvaluationContext:
        full_conversation = "\n".join(
            f"[{msg.role.value}] {msg.content}"
            for msg in payload.messages
        )

        return EvaluationContext(
            prompt_text=payload.active_text,
            full_conversation=full_conversation,
            has_attachments=bool(payload.all_attachments),
            environment=payload.environment,
            model_id=payload.model_id,
            user_id=payload.user_id,
            session_id=payload.session_id,
            policy_overrides=policy_overrides or {},
            metadata=payload.metadata,
        )
