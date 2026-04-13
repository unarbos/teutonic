"""Local window clocks for testing without a blockchain.

All timestamps use ``time.time()`` (UTC epoch seconds) so they are
directly comparable with storage ``last_modified`` values (file mtime
or S3 LastModified).
"""

from __future__ import annotations

import asyncio
import hashlib
import time


class ManualClock:
    """Window advances only when the caller explicitly sets it.

    For manual clocks the ``window_duration`` is advisory (used by miners
    to budget training time).  ``window_end_time`` is computed relative to
    the wall-clock time when the window was set.
    """

    def __init__(self, start: int = 0, window_duration: float = 30.0):
        self._window = start
        self._window_duration = window_duration
        self._window_set_at: float = time.time()

    @property
    def current_window(self) -> int:
        return self._window

    @property
    def window_duration(self) -> float:
        return self._window_duration

    def window_end_time(self, window: int) -> float:
        offset = (window - self._window + 1) * self._window_duration
        return self._window_set_at + offset

    def window_block_hash(self, window: int) -> str:
        return hashlib.blake2b(
            f"local-block:{window}".encode(), digest_size=32
        ).hexdigest()

    def set_window(self, w: int) -> None:
        self._window = w
        self._window_set_at = time.time()

    async def wait_for_window(self, target: int) -> None:
        while self._window < target:
            await asyncio.sleep(0.01)


class TimedClock:
    """Window advances every *interval* seconds from construction time."""

    def __init__(self, interval: float = 10.0):
        self._t0 = time.time()
        self._interval = interval

    @property
    def current_window(self) -> int:
        return int((time.time() - self._t0) / self._interval)

    @property
    def window_duration(self) -> float:
        return self._interval

    def window_end_time(self, window: int) -> float:
        return self._t0 + (window + 1) * self._interval

    def window_block_hash(self, window: int) -> str:
        return hashlib.blake2b(
            f"local-block:{window}".encode(), digest_size=32
        ).hexdigest()

    async def wait_for_window(self, target: int) -> None:
        while self.current_window < target:
            await asyncio.sleep(0.05)
