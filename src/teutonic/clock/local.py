"""Local window clocks for testing without a blockchain."""

from __future__ import annotations

import asyncio
import time


class ManualClock:
    """Window advances only when the caller explicitly sets it."""

    def __init__(self, start: int = 0):
        self._window = start

    @property
    def current_window(self) -> int:
        return self._window

    def set_window(self, w: int) -> None:
        self._window = w

    async def wait_for_window(self, target: int) -> None:
        while self._window < target:
            await asyncio.sleep(0.01)


class TimedClock:
    """Window advances every *interval* seconds from construction time."""

    def __init__(self, interval: float = 10.0):
        self._t0 = time.monotonic()
        self._interval = interval

    @property
    def current_window(self) -> int:
        return int((time.monotonic() - self._t0) / self._interval)

    async def wait_for_window(self, target: int) -> None:
        while self.current_window < target:
            await asyncio.sleep(0.05)
