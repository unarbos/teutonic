"""Keep Horizon resident while idle; give MiMo exclusive GPU ownership in duels."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from teutonic.evaluator.horizon_pipeline_runtime import HorizonWorker
from teutonic.evaluator.vllm_runtime import wait_for_gpu_release

log = logging.getLogger(__name__)


class IdleHorizon:
    def __init__(self, downloader, gpu_ids, identity):
        self.downloader, self.gpu_ids, self.identity = downloader, list(gpu_ids), identity
        self.directory = Path(downloader.root) / "gpu-worker"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._operation = threading.RLock()
        self._state = threading.Lock()
        self._stopped = threading.Event()
        self._enabled = False
        self._worker = None
        self._snapshot = None
        # Linux PDEATHSIG follows the spawning thread, not just its process.
        # Keep that thread alive after loading, until its GPU worker is closed.
        self._loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="horizon-preload")
        self._preloads = []
        self._status = {"state": "unloaded"}

    def _update(self, **value):
        with self._state:
            self._status = value

    def status(self):
        with self._state:
            value = dict(self._status)
            if (value["state"] == "ready" and self._worker is not None
                    and self._worker.process.poll() is not None):
                value = {"state": "failed", "error_type": "WorkerExited"}
            return value

    def _progress(self, event):
        log.info("Idle Horizon: %s", event)

    def preload(self):
        """Nonblocking startup/post-duel load. A failed load can retry next duel."""
        with self._state:
            if self._stopped.is_set():
                return
            self._enabled = True
            self._preloads = [future for future in self._preloads if not future.done()]
            self._preloads.append(self._loader.submit(self._warm))

    def _warm(self):
        with self._operation:
            try:
                # release() can win the race before this thread starts. Never
                # allocate a late background worker while MiMo owns the GPUs.
                with self._state:
                    if not self._enabled or self._stopped.is_set():
                        return
                self._update(state="waiting_for_snapshot")
                snapshot = self.downloader.wait_ready(self._progress)
                self._ensure(snapshot, self.identity, self._progress)
            except Exception as exc:  # noqa: BLE001 -- background failure is observable and retryable
                if not self._stopped.is_set():
                    self._update(state="failed", error_type=type(exc).__name__)
                    log.error("Idle Horizon preload failed: %s", type(exc).__name__)

    def _discard(self):
        with self._state:
            worker = self._worker
        if worker is not None:
            worker.close()
            with self._state:
                self._worker = None
                self._snapshot = None

    def _ensure(self, snapshot, identity, on_phase):
        if self._stopped.is_set():
            raise RuntimeError("Horizon residency is stopping")
        worker = self._worker
        if worker is not None and (worker.process.poll() is not None
                                   or worker.identity != identity or self._snapshot != Path(snapshot)):
            self._discard()
            worker = None
        if worker is None:
            self._update(state="waiting_for_gpu_release")
            wait_for_gpu_release(self.gpu_ids)
            if self._stopped.is_set():
                raise RuntimeError("Horizon residency is stopping")
            self._update(state="loading")
            # Bound idle-worker log storage while retaining the preceding load.
            previous = self.directory / "horizon.log"
            if previous.exists():
                previous.replace(self.directory / "horizon.previous.log")
            worker = HorizonWorker("horizon", snapshot, self.gpu_ids, self.directory, identity, on_phase)
            with self._state:
                self._worker, self._snapshot = worker, Path(snapshot)
        try:
            if self._stopped.is_set():
                raise RuntimeError("Horizon residency is stopping")
            worker.on_phase = on_phase
            worker.wait_ready()
        except BaseException:
            self._discard()
            raise
        self._update(state="ready", gpu_ids=self.gpu_ids, load_seconds=worker.report.get("load_seconds"),
                     log_path=str(self.directory / "horizon.log"))
        return worker

    def acquire(self, snapshot, identity, on_phase):
        """The single admitted duel borrows the ready worker, or waits for loading."""
        with self._operation:
            try:
                worker = self._ensure(snapshot, identity, on_phase)
            except Exception as exc:
                if not self._stopped.is_set():
                    self._update(state="failed", error_type=type(exc).__name__)
                raise
            on_phase({"phase": "reference_ready", "load_seconds": worker.report.get("load_seconds")})
            return worker

    def release(self):
        """Finish/cancel preloading and unload before the MiMo phase, even on cache hits."""
        with self._state:
            self._enabled = False
        with self._operation:
            self._discard()
            self._update(state="unloaded")

    def close(self):
        """Prevent new loads and interrupt a worker still waiting for readiness."""
        self._stopped.set()
        with self._state:
            self._enabled = False
            worker = self._worker
        if worker is not None:
            worker.close()
        with self._operation:
            self._discard()
            self._update(state="stopped")
        self._loader.shutdown(wait=True, cancel_futures=True)
