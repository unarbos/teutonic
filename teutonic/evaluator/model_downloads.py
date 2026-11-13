"""CPU/network-only checkpoint preparation overlapping reference inference."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from time import monotonic


class ModelDownloads:
    """Resolve both snapshots without allocating GPUs or sharing partial paths.

    Keep this context inside the model-cache lock. On any exit, join the active
    resolver before releasing that lock; do not leave a downloader behind while
    cleanup or the next evaluation can touch its files.
    """

    def __init__(self, resolver, king, challenger):
        self.resolver = resolver
        self.artifacts = {"king": king, "challenger": challenger}
        self._lock = Lock()
        self._stop = Event()
        self._states = {role: {"state": "pending"} for role in self.artifacts}
        self._executor = None
        self._future = None
        self._started = None
        self._finished = None

    def __enter__(self):
        self._started = monotonic()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="duel-download")
        self._future = self._executor.submit(self._resolve)
        return self

    def _resolve(self):
        snapshots = {}
        try:
            # Each resolver already parallelizes file transfers. Keep the two
            # roles sequential, including when both refer to the same cache path.
            for role, artifact in self.artifacts.items():
                if self._stop.is_set():
                    break
                with self._lock:
                    self._states[role] = {"state": "downloading"}
                try:
                    snapshots[role] = self.resolver.resolve(artifact)
                except BaseException:
                    with self._lock:
                        self._states[role] = {"state": "failed"}
                    raise
                with self._lock:
                    self._states[role] = {"state": "verified"}
            return snapshots
        finally:
            with self._lock:
                self._finished = monotonic()

    def status(self):
        with self._lock:
            return {role: dict(state) for role, state in self._states.items()}

    def check(self):
        """Propagate download failures between reference batches without waiting."""
        if self._future.done():
            self._future.result()

    def result(self):
        """Wait for complete, verified snapshots; never return staging directories."""
        snapshots = self._future.result()
        if snapshots.keys() != self.artifacts.keys():
            raise RuntimeError("model preparation was cancelled")
        return snapshots

    @property
    def elapsed_seconds(self):
        with self._lock:
            return (self._finished or monotonic()) - self._started

    def __exit__(self, *_):
        self._stop.set()
        self._executor.shutdown(wait=True, cancel_futures=True)
