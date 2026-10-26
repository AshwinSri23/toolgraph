"""Side-effect policies: how dangerous each tool is and what to do when unsure."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import Enum


class SideEffect(str, Enum):
    READ = "read"
    """No side effects. Safe to repeat."""

    KEYED_WRITE = "keyed_write"
    """A write the server deduplicates by an idempotency key we send."""

    IDEMPOTENT_WRITE = "idempotent_write"
    """A write that has the same effect however many times it is applied."""

    UNKEYED_WRITE = "unkeyed_write"
    """A write that cannot be safely repeated. Never replayed on an unknown outcome."""

    @property
    def is_write(self) -> bool:
        return self is not SideEffect.READ

    @property
    def safe_to_replay(self) -> bool:
        """Can an attempt with an unknown outcome be sent again?"""
        return self is not SideEffect.UNKEYED_WRITE


@dataclass(frozen=True)
class ToolPolicy:
    side_effect: SideEffect = SideEffect.UNKEYED_WRITE
    max_attempts: int = 3
    """Total sends allowed, including the first."""

    backoff_base: float = 1.0
    backoff_max: float = 60.0
    timeout: float | None = 30.0
    """Seconds to wait for a single request before treating its outcome as unknown."""

    job_timeout: float | None = None
    """Seconds a long-running task may take before it is cancelled."""

    poll_interval: float = 1.0
    """Polling interval for tasks when the server suggests none."""

    concurrency_group: str | None = None
    key_arg: str | None = "idempotency_key"
    """For keyed writes: the argument the key is sent in (it is always also sent in _meta)."""

    error_may_have_applied: bool | None = None
    """Whether a tool error result may hide a partial write. Defaults to True for writes."""

    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    """Resource templates such as ``"file:{path}"``, formatted with the call's arguments."""

    def backoff(self, attempt: int) -> float:
        """Delay before the retry that follows send number ``attempt`` (1-based)."""
        return min(self.backoff_max, self.backoff_base * (2 ** (attempt - 1)))

    @property
    def tool_error_uncertain(self) -> bool:
        if self.error_may_have_applied is None:
            return self.side_effect.is_write
        return self.error_may_have_applied


# Anything we know nothing about is assumed to be the most dangerous kind of tool.
DEFAULT_POLICY = ToolPolicy(side_effect=SideEffect.UNKEYED_WRITE, max_attempts=1)


@dataclass
class PolicyRegistry:
    policies: dict[str, ToolPolicy] = field(default_factory=dict)
    group_limits: dict[str, int] = field(default_factory=dict)
    default: ToolPolicy = DEFAULT_POLICY

    def register(self, tool: str, policy: ToolPolicy | None = None, **kwargs: object) -> ToolPolicy:
        """Register a policy for a tool name or glob pattern such as ``"github.*"``."""
        if policy is None:
            policy = ToolPolicy(**kwargs)  # type: ignore[arg-type]
        self.policies[tool] = policy
        return policy

    def limit(self, group: str, n: int) -> None:
        self.group_limits[group] = n

    def get(self, tool: str) -> ToolPolicy:
        if tool in self.policies:
            return self.policies[tool]
        for pattern in sorted(self.policies, key=len, reverse=True):
            if fnmatch.fnmatchcase(tool, pattern):
                return self.policies[pattern]
        return self.default
