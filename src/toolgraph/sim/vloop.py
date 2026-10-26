"""An asyncio event loop that runs on virtual time.

The loop asks its selector to wait for I/O for as long as there is nothing
ready to run. This selector never blocks: it advances the virtual clock to the
next timer instead. Time therefore moves only when every coroutine is waiting,
so an hour of simulated work finishes in milliseconds and replays identically.
"""

from __future__ import annotations

import asyncio
import selectors
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


class SimulationDeadlock(RuntimeError):
    """Nothing is runnable and no timer is pending: the program would hang."""


class _AdvancingSelector(selectors.DefaultSelector):
    def __init__(self, loop: VirtualTimeLoop) -> None:
        super().__init__()
        self._loop = loop

    def select(self, timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        # Pick up anything real (the loop's self-pipe) without blocking.
        events = super().select(0)
        if events or timeout == 0:
            return events
        if timeout is None:
            raise SimulationDeadlock("every task is waiting and no timer is scheduled")
        self._loop._advance(timeout)
        return []


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    def __init__(self, start: float = 0.0) -> None:
        self._vtime = float(start)
        super().__init__(selector=_AdvancingSelector(self))

    def time(self) -> float:
        return self._vtime

    def _advance(self, dt: float) -> None:
        self._vtime += dt


def run_virtual(main: Callable[[], Awaitable[T]], start: float = 0.0) -> T:
    """Run ``main()`` to completion on a fresh virtual-time loop."""
    runner = asyncio.Runner(loop_factory=lambda: VirtualTimeLoop(start))
    with runner:
        return runner.run(main())  # type: ignore[arg-type]


def now() -> float:
    return asyncio.get_running_loop().time()


__all__: list[Any] = ["SimulationDeadlock", "VirtualTimeLoop", "run_virtual", "now"]
