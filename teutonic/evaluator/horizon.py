"""Persistent, immutable Horizon snapshot preparation (Hugging Face reads only)."""

import fcntl
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "IFM/K2-Horizon-375B-A23B"
REVISION = "7d22cc0ac26a91003d19482efa71070244239b11"
CONFIG_SHA256 = "0daf184082ec39c3b24c5da6fcd5739f3176f0ad2c9e21fddaaea24b7400a583"
INDEX_SHA256 = "e689097791a103adcfb36b20ee752813e8cd2f2d3d153bc22eea20f660baeafd"


def file_hash(path, algorithm="sha256", git_blob=False):
    h = hashlib.new(algorithm)
    if git_blob:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def stamp(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def verify_file(path, metadata):
    if path.stat().st_size != metadata["size"]:
        raise ValueError("Horizon file size mismatch")
    observed = file_hash(path, metadata["algorithm"], metadata["algorithm"] == "sha1")
    if observed != metadata["hash"]:
        raise ValueError(f"Horizon file hash mismatch: {path.name}")


class HorizonDownload:
    def __init__(self, root, *, api=None, download=None, workers=4):
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError("Horizon download workers must be in [1, 8]")
        self.workers = workers
        self.root = Path(root).expanduser().resolve()
        self.snapshot = self.root / REVISION
        self.api, self.download = api, download
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.thread = None
        self.progress = {"state": "pending", "repository": REPO, "revision": REVISION,
                         "directory": str(self.snapshot), "completed_files": 0, "total_files": 0}

    @classmethod
    def from_env(cls, model_cache):
        root = Path(os.environ.get("TEUTONIC_HORIZON_CACHE_DIR", "/tmp/teutonic/horizon"))
        if root.resolve().is_relative_to(Path(model_cache).resolve()):
            raise ValueError("Horizon cache must be outside transient duel model cache")
        return cls(root)

    def status(self):
        with self._lock:
            return dict(self.progress)

    def update(self, **fields):
        with self._lock:
            self.progress.update(fields)

    def prepare(self):
        from huggingface_hub import HfApi, hf_hub_download

        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "download.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            api, download = self.api or HfApi(), self.download or hf_hub_download
            info = api.model_info(REPO, revision=REVISION, files_metadata=True)
            if info.sha != REVISION:
                raise ValueError("Horizon revision changed")
            files = {}
            for entry in info.siblings:
                name = entry.rfilename
                if name.startswith(".") or not name.endswith((".safetensors", ".json", ".py", ".jinja", ".txt", ".model")):
                    continue
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("invalid Horizon snapshot path")
                lfs = entry.lfs
                files[name] = {"size": entry.size, "algorithm": "sha256" if lfs else "sha1",
                               "hash": lfs.sha256 if lfs else entry.blob_id}
            if not {"config.json", "model.safetensors.index.json", "tokenizer.json"} <= files.keys():
                raise ValueError("incomplete Horizon snapshot inventory")
            self.snapshot.mkdir(parents=True, exist_ok=True)
            receipt_path = self.snapshot / "verified.json"
            try:
                receipt = json.loads(receipt_path.read_text())
            except (OSError, ValueError):
                receipt = {}
            verified = {}
            self.update(state="downloading", total_files=len(files))
            failed = threading.Event()
            def prepare_file(name, metadata):
                if self._stop.is_set() or failed.is_set():
                    return None
                path = self.snapshot / name
                saved = receipt.get("files", {}).get(name, {})
                if not (path.is_file() and saved.get("metadata") == metadata and saved.get("stat") == stamp(path)):
                    for attempt in range(2):
                        download(repo_id=REPO, filename=name, revision=REVISION,
                                 local_dir=str(self.snapshot), force_download=bool(attempt))
                        try:
                            verify_file(path, metadata)
                            break
                        except ValueError:
                            if attempt:
                                raise
                return {"metadata": metadata, "stat": stamp(path)}

            # Transfer independent shards concurrently; only this thread updates
            # the receipt, so interruption leaves an atomic, reusable checkpoint.
            with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="horizon-file") as pool:
                pending = {pool.submit(prepare_file, name, metadata): name
                           for name, metadata in sorted(files.items())}
                try:
                    for future in as_completed(pending):
                        saved = future.result()
                        if saved is None:
                            continue
                        name = pending[future]
                        verified[name] = saved
                        self.update(completed_files=len(verified), current_file=name)
                        temporary = receipt_path.with_suffix(".pending")
                        temporary.write_text(json.dumps({"revision": REVISION, "files": verified}, sort_keys=True))
                        temporary.replace(receipt_path)
                except BaseException:
                    failed.set()
                    for future in pending:
                        future.cancel()
                    raise
            if self._stop.is_set():
                self.update(state="stopped")
                return None
            if file_hash(self.snapshot / "config.json") != CONFIG_SHA256:
                raise ValueError("pinned Horizon config hash mismatch")
            if file_hash(self.snapshot / "model.safetensors.index.json") != INDEX_SHA256:
                raise ValueError("pinned Horizon weight-index hash mismatch")
            index = json.loads((self.snapshot / "model.safetensors.index.json").read_text())
            if not set(index["weight_map"].values()) <= verified.keys():
                raise ValueError("Horizon weight shards missing")
            self.update(state="ready", current_file=None)
            return self.snapshot

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        def run():
            try:
                self.prepare()
            except Exception as exc:  # noqa: BLE001 -- background status must report failures
                # Do not include authentication-bearing HTTP exception URLs.
                self.update(state="failed", error_type=type(exc).__name__)
        self.thread = threading.Thread(target=run, name="horizon-download", daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=2)

    def require_ready(self):
        if self.status()["state"] != "ready":
            raise RuntimeError("Horizon snapshot is not verified and ready; inspect horizon_download health")
        receipt = json.loads((self.snapshot / "verified.json").read_text())
        if receipt.get("revision") != REVISION or any(
            saved["stat"] != stamp(self.snapshot / name)
            for name, saved in receipt["files"].items()
        ):
            self.update(state="failed", error_type="SnapshotChanged")
            raise RuntimeError("Horizon snapshot changed after verification")
        return self.snapshot

    def wait_ready(self, on_progress):
        last_report = 0.0
        while self.status()["state"] != "ready":
            state = self.status()
            if state["state"] in {"failed", "stopped"} or not self.thread or not self.thread.is_alive():
                raise RuntimeError("Horizon preparation is not running or failed; inspect horizon_download health")
            if time.monotonic() - last_report >= 30:
                on_progress({"phase": "reference_download_wait", "download": state})
                last_report = time.monotonic()
            if self._stop.wait(1):
                raise RuntimeError("Horizon preparation stopped")
        return self.require_ready()
