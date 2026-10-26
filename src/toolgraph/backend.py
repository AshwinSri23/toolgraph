"""The interface between the scheduler and whatever actually runs tools.

A backend answers one question per request: what happened? The scheduler only
needs to distinguish a handful of outcomes, and above all whether a failure
means "definitely not applied" or "might have been applied".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class Question:
    """Something a tool needs from a person (a 2FA code, a choice) before it can go on."""

    prompt: str
    schema: dict[str, Any] | None = None
    state: dict[str, Any] = field(default_factory=dict)
    """Opaque backend state needed to continue once answered."""


@dataclass
class Done:
    """The call finished and the tool succeeded."""

    value: Any


@dataclass
class ToolFailed:
    """The tool ran and reported an error."""

    message: str
    value: Any = None


@dataclass
class Started:
    """The server accepted the call as a long-running job (an MCP Task)."""

    task_id: str
    poll_interval: float | None = None


@dataclass
class NeedsInput:
    question: Question


CallOutcome = Done | ToolFailed | Started | NeedsInput


@dataclass
class TaskStatus:
    status: Literal["working", "input_required", "completed", "failed", "cancelled"]
    message: str | None = None
    question: Question | None = None
    poll_interval: float | None = None


class Rejected(Exception):
    """The request was refused before anything was applied. Always safe to retry."""


class Uncertain(Exception):
    """The request may or may not have been applied (lost response, timeout, crash)."""


class TaskLost(Exception):
    """The server no longer knows the task (expired or never created)."""


class Backend(Protocol):
    async def call(
        self,
        tool: str,
        args: dict[str, Any],
        *,
        idempotency_key: str | None,
        answer: Any = None,
        question: Question | None = None,
    ) -> CallOutcome:
        """Send one ``tools/call``. ``answer``/``question`` continue a call that asked for input."""
        ...

    async def get_task(self, tool: str, task_id: str) -> TaskStatus: ...

    async def task_result(self, tool: str, task_id: str) -> Done | ToolFailed: ...

    async def answer_task(self, tool: str, task_id: str, question: Question, answer: Any) -> None: ...

    async def cancel_task(self, tool: str, task_id: str) -> None: ...
