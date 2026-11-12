"""Bounded RPC and ownership shared by Horizon and MiMo worker process groups."""

import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
import weakref
from pathlib import Path

from teutonic.evaluator.vllm_score import code_identity, deterministic_env, input_hash, settings

_workers = weakref.WeakSet()
_worker_lock = threading.Lock()


def close_all_workers():
    with _worker_lock:
        workers = list(_workers)
    for worker in workers:
        worker.close()


def worker_status():
    with _worker_lock:
        workers = list(_workers)
    return {"running": any(worker.process.poll() is None for worker in workers),
            "workers": [{**worker.report, "ready": worker.ready,
                         "running": worker.process.poll() is None} for worker in workers]}


def worker_python():
    return os.environ.get("TEUTONIC_VLLM_PYTHON", sys.executable)


def worker_env(role, gpu_ids):
    # GPU scorers only need local checkpoints, not R2 or control-plane credentials.
    allowed = {"HOME", "USER", "LOGNAME", "PATH", "LANG", "LC_ALL", "TMPDIR",
               "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH", "CC", "CXX", "CUDA_HOME",
               "CUDA_PATH", "CUDA_CACHE_PATH", "HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update(deterministic_env(role))
    env["PATH"] = str(Path(worker_python()).absolute().parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    if root := os.environ.get("TEUTONIC_VLLM_CACHE_DIR"):
        workspace = Path(root) / ("horizon" if role == "horizon" else "mimo")
        workspace.mkdir(parents=True, exist_ok=True)
        env["FLASHINFER_WORKSPACE_BASE"] = str(workspace)
    return env


def runtime_identity():
    result = subprocess.run([worker_python(), "-m", "teutonic.evaluator.vllm_score", "--identity"],
                            env=worker_env("horizon", []), capture_output=True, text=True,
                            check=True, timeout=60)
    identity = json.loads(result.stdout)
    if identity["code"] != code_identity():
        raise RuntimeError("worker and evaluator scorer code differ")
    return identity


def wait_for_gpu_release(gpu_ids, timeout=120):
    """Wait for asynchronous teardown; never stop another job to free its GPUs."""
    deadline = time.monotonic() + timeout
    values = []
    while True:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--id=" + ",".join(map(str, gpu_ids)),
                 "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=15,
            )
        except subprocess.TimeoutExpired:
            # NVML can stall briefly during context teardown; never treat that as idle.
            pass
        else:
            values = [int(value.strip()) for value in result.stdout.splitlines() if value.strip()]
            if len(values) == len(gpu_ids) and all(value < 1000 for value in values):
                return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPUs did not release memory before the next phase: {values}")
        time.sleep(1)


class Worker:
    """One model engine, with bounded I/O timeouts and an independent reply pipe."""

    def _settings(self, role, gpu_ids, max_model_len):
        return settings(role, len(gpu_ids), max_model_len)

    def _command(self, role, snapshot, write_fd, gpu_ids):
        return [worker_python(), "-m", "teutonic.evaluator.vllm_score", "--role", role,
                "--snapshot", str(snapshot), "--response-fd", str(write_fd),
                "--tensor-parallel-size", str(len(gpu_ids)),
                "--max-model-len", str(self.configuration["max_model_len"])]

    def _environment(self, role, gpu_ids):
        return worker_env(role, gpu_ids)

    def __init__(self, role, snapshot, gpu_ids, directory, identity, on_phase,
                 *, load_timeout=900, score_timeout=1800, worker_id=None, max_model_len=None):
        self.role, self.identity, self.on_phase = role, identity, on_phase
        self.load_timeout, self.score_timeout = load_timeout, score_timeout
        self.batch, self.ready, self.closed = 0, False, False
        self.close_lock = threading.Lock()
        self.buffer = bytearray()
        self.configuration = self._settings(role, gpu_ids, max_model_len)
        self.worker_id = worker_id or role
        self.report = {"role": role, "worker_id": self.worker_id, "settings": self.configuration,
                       "gpu_ids": list(gpu_ids), "batches": 0,
                       "documents": 0, "scored_tokens": 0, "scoring_seconds": 0.0}
        self.log_path = Path(directory) / f"{self.worker_id}.log"
        self.log = self.log_path.open("wb")
        self.response_fd, write_fd = os.pipe()
        try:
            self.process = subprocess.Popen(
                self._command(role, snapshot, write_fd, gpu_ids),
                env=self._environment(role, gpu_ids), stdin=subprocess.PIPE,
                stdout=self.log, stderr=subprocess.STDOUT, pass_fds=(write_fd,),
                start_new_session=True,
            )
        except BaseException:
            os.close(self.response_fd)
            self.log.close()
            raise
        finally:
            os.close(write_fd)
        self.started = time.monotonic()
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.response_fd, False)
        with _worker_lock:
            _workers.add(self)
        try:
            on_phase({"phase": "model_loading", "role": role, "worker_id": self.worker_id,
                      "gpu_ids": list(gpu_ids), "settings": self.configuration})
        except BaseException:
            self.close()
            raise

    def _check(self, deadline):
        if self.process.poll() is not None:
            raise RuntimeError(f"{self.role} scoring worker exited ({self.process.returncode}); {self.log_path}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{self.role} scoring worker timed out; {self.log_path}")

    def _send(self, message, deadline):
        remaining = memoryview((json.dumps(message, allow_nan=False) + "\n").encode())
        fd = self.process.stdin.fileno()
        while remaining:
            self._check(deadline)
            if select.select([], [fd], [], min(1, max(0, deadline - time.monotonic())))[1]:
                try:
                    remaining = remaining[os.write(fd, remaining):]
                except BlockingIOError:
                    pass

    def _receive(self, deadline):
        while b"\n" not in self.buffer:
            self._check(deadline)
            if select.select([self.response_fd], [], [], min(1, max(0, deadline - time.monotonic())))[0]:
                chunk = os.read(self.response_fd, 65536)
                if not chunk:
                    raise RuntimeError(f"{self.role} worker closed reply pipe; {self.log_path}")
                self.buffer.extend(chunk)
                if len(self.buffer) > 1024 * 1024:
                    raise ValueError("oversized scorer reply")
        line, _, rest = self.buffer.partition(b"\n")
        self.buffer = bytearray(rest)
        return json.loads(line)

    def wait_ready(self):
        if not self.ready:
            response = self._receive(self.started + self.load_timeout)
            if response.get("type") != "ready" or response.get("identity") != self.identity:
                raise ValueError("worker ready identity differs from pinned runtime")
            if response.get("settings") != self.configuration:
                raise ValueError("worker topology/settings differ from requested layout")
            self.ready = True
            self.report["load_seconds"] = response["load_seconds"]
            self.on_phase({"phase": "model_loaded", "role": self.role,
                           "worker_id": self.worker_id,
                           "seconds": response["load_seconds"]})

    def score(self, documents, sequences):
        self.wait_ready()
        rows = [{"id": d.doc_id, "tokens": tokens} for d, tokens in zip(documents, sequences, strict=True)]
        self.batch += 1
        deadline = time.monotonic() + self.score_timeout
        self._send({"type": "score", "batch": self.batch, "rows": rows}, deadline)
        response = self._receive(deadline)
        if response.get("type") != "scores" or response.get("batch") != self.batch:
            raise ValueError("unexpected worker response")
        values = response["rows"]
        import math
        for expected, value in zip(rows, values, strict=True):
            if (value["id"] != expected["id"] or value["scored_tokens"] != len(expected["tokens"]) - 1
                    or value["input_sha256"] != input_hash(expected["tokens"])
                    or not math.isfinite(value["nll"]) or value["nll"] < 0):
                raise ValueError("scorer result does not match the selected document")
        self.report["batches"] += 1
        self.report["documents"] += len(values)
        self.report["scored_tokens"] += sum(value["scored_tokens"] for value in values)
        self.report["scoring_seconds"] += response["seconds"]
        self.on_phase({"phase": "scoring", **self.report})
        return values

    def close(self):
        with self.close_lock:
            self._close()

    def _close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.poll() is None and self.ready:
                try:
                    self._send({"type": "stop"}, time.monotonic() + 2)
                    self.process.wait(timeout=10)
                except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                    pass
            # Also reap children when the engine parent exited before its workers.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(self.process.pid, sig)
                except ProcessLookupError:
                    break
                if sig == signal.SIGTERM:
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            # A killed CUDA context can remain in driver teardown beyond 15 seconds.
            self.process.wait(timeout=120)
        finally:
            self.process.stdin.close()
            os.close(self.response_fd)
            self.log.close()
            with _worker_lock:
                _workers.discard(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
