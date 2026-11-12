import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from teutonic.evaluator import idle_horizon as idle


@pytest.fixture
def setup(tmp_path, monkeypatch):
    created = []
    waiting, proceed = threading.Event(), threading.Event()
    proceed.set()

    class Worker:
        def __init__(self, role, snapshot, ids, directory, identity, on_phase):
            self.identity, self.on_phase = identity, on_phase
            self.report = {"load_seconds": 3}
            self.closed = False
            self.process = SimpleNamespace(poll=lambda: 0 if self.closed else None)
            created.append(self)

        def wait_ready(self):
            waiting.set()
            assert proceed.wait(5)
            if self.closed:
                raise RuntimeError("worker closed")

        def close(self):
            self.closed = True
            proceed.set()

    def released(ids):
        assert all(worker.closed for worker in created), "overlapping GPU model loads"

    monkeypatch.setattr(idle, "HorizonWorker", Worker)
    monkeypatch.setattr(idle, "wait_for_gpu_release", released)
    download = SimpleNamespace(root=tmp_path, wait_ready=lambda callback: tmp_path)
    manager = idle.IdleHorizon(download, list(range(8)), {"version": 1})
    yield manager, created, waiting, proceed
    manager.close()


def finish_preload(manager):
    for future in list(manager._preloads):
        future.result(timeout=5)


def test_startup_reuse_release_and_post_duel_reload(setup):
    manager, workers, _, _ = setup
    manager.preload()
    finish_preload(manager)
    assert manager.status()["state"] == "ready"
    events = []
    assert manager.acquire(manager.downloader.root, manager.identity, events.append) is workers[0]
    assert len(workers) == 1
    assert events[0]["phase"] == "reference_ready"
    manager.release()
    assert workers[0].closed
    assert manager.status()["state"] == "unloaded"
    manager.preload()
    finish_preload(manager)
    assert len(workers) == 2 and not workers[1].closed


def test_duel_arriving_during_preload_waits_and_reuses_worker(setup):
    manager, workers, waiting, proceed = setup
    proceed.clear()
    manager.preload()
    assert waiting.wait(5)
    result = []
    thread = threading.Thread(target=lambda: result.append(
        manager.acquire(manager.downloader.root, manager.identity, lambda _: None)))
    thread.start()
    proceed.set()
    thread.join(5)
    assert not thread.is_alive()
    assert result == workers and len(workers) == 1


def test_release_cancels_preload_that_has_not_started(setup):
    manager, workers, _, _ = setup
    with manager._operation:
        manager.preload()
        manager.release()
    finish_preload(manager)
    assert workers == []
    assert manager.status()["state"] == "unloaded"


def test_failed_background_load_retries_on_acquire(setup, monkeypatch):
    manager, workers, _, _ = setup
    original = idle.HorizonWorker.wait_ready
    monkeypatch.setattr(idle.HorizonWorker, "wait_ready", lambda self: (_ for _ in ()).throw(ValueError("bad load")))
    manager.preload()
    finish_preload(manager)
    assert manager.status() == {"state": "failed", "error_type": "ValueError"}
    assert workers[0].closed
    monkeypatch.setattr(idle.HorizonWorker, "wait_ready", original)
    assert manager.acquire(manager.downloader.root, manager.identity, lambda _: None) is workers[1]


def test_dead_or_different_worker_is_replaced(setup):
    manager, workers, _, _ = setup
    manager.preload()
    finish_preload(manager)
    workers[0].closed = True
    assert manager.status()["error_type"] == "WorkerExited"
    manager.acquire(manager.downloader.root, manager.identity, lambda _: None)
    manager.acquire(manager.downloader.root, {"version": 2}, lambda _: None)
    assert len(workers) == 3
    assert workers[1].closed


def test_shutdown_interrupts_load_and_prevents_reload(setup):
    manager, workers, waiting, proceed = setup
    proceed.clear()
    manager.preload()
    assert waiting.wait(5)
    manager.close()
    finish_preload(manager)
    manager.preload()
    assert len(workers) == 1 and workers[0].closed
    assert manager.status()["state"] == "stopped"
    with pytest.raises(RuntimeError, match="stopping"):
        manager.acquire(manager.downloader.root, manager.identity, lambda _: None)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux parent-death signal semantics")
def test_real_parent_death_signal_survives_completed_preload(tmp_path, monkeypatch):
    child = """
import ctypes, signal, sys
assert ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) == 0
print('ready', flush=True)
for line in sys.stdin:
    print('pong', flush=True)
"""
    created = []

    class ProcessWorker:
        def __init__(self, role, snapshot, ids, directory, identity, on_phase):
            self.identity, self.on_phase, self.report = identity, on_phase, {}
            self.owner = threading.get_native_id()
            self.process = subprocess.Popen([sys.executable, '-u', '-c', child],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            created.append(self)

        def wait_ready(self):
            assert self.process.stdout.readline().strip() == 'ready'

        def close(self):
            if self.process.poll() is None:
                self.process.terminate()
            self.process.wait(timeout=5)
            self.process.stdin.close()
            self.process.stdout.close()

    # Negative control reproduces the bug without loading CUDA or model weights.
    def temporary_loader():
        worker = ProcessWorker(None, None, None, None, {}, None)
        worker.wait_ready()
    thread = threading.Thread(target=temporary_loader)
    thread.start()
    thread.join(5)
    assert not thread.is_alive()
    assert created[0].process.wait(timeout=5) == -15
    created[0].close()

    monkeypatch.setattr(idle, 'HorizonWorker', ProcessWorker)
    monkeypatch.setattr(idle, 'wait_for_gpu_release', lambda _: None)
    manager = idle.IdleHorizon(SimpleNamespace(root=tmp_path, wait_ready=lambda _: tmp_path), list(range(8)), {})
    try:
        manager.preload()
        finish_preload(manager)
        worker = created[1]
        # The spawning thread remains alive after the load task has returned.
        assert manager._loader.submit(threading.get_native_id).result(timeout=5) == worker.owner
        assert worker.process.poll() is None
        worker.process.stdin.write('ping\n')
        worker.process.stdin.flush()
        assert worker.process.stdout.readline().strip() == 'pong'
    finally:
        manager.close()
    assert worker.process.poll() is not None
