"""Persistent, ETag-pinned document indexes; idle-time, resumable prefetch.

Receipts attest to a completed pinned transfer, not a whole-file content hash.
The store must live on trusted local storage outside the disposable caches.
"""

import fcntl
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request

from teutonic.evaluation.index_manifest import fetch_index_manifest, validate_index_manifest_pin
from teutonic.evaluation.long_documents import validate_index_file
from teutonic.evaluator.range_http import pooled_urlopen

log = logging.getLogger(__name__)
CHUNK_BYTES = 8 * 1024 * 1024
STREAM_BYTES = 256 * 1024


def store_dir():
    value = os.environ.get("TEUTONIC_DOC_INDEX_STORE_DIR", "").strip()
    return Path(value).expanduser() if value else None


def identity(descriptor):
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()


def _stamp(stat):
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json(path, value):
    temporary = path.with_suffix(".pending")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class LocalIndexReader:
    def __init__(self, stream, stamp):
        self.stream, self.stamp = stream, stamp

    def read(self, offset, size):
        # pread uses the OS page cache without loading the entire index into RAM.
        if _stamp(os.fstat(self.stream.fileno())) != self.stamp:
            raise OSError("local index metadata changed")
        data = os.pread(self.stream.fileno(), size, offset)
        if len(data) != size:
            raise OSError("short local index read")
        return data

    def close(self):
        self.stream.close()

    def __del__(self):
        self.close()


class LocalIndexStore:
    def __init__(self, root):
        self.root = Path(root)

    def directory(self, descriptor):
        return self.root / identity(descriptor)

    def open(self, descriptor):
        directory = self.directory(descriptor)
        receipt = _json(directory / "complete.json")
        if receipt.get("descriptor") != descriptor:
            return None
        try:
            stream = (directory / "index.bin").open("rb")
        except OSError:
            return None
        stamp = _stamp(os.fstat(stream.fileno()))
        if receipt.get("stat") != stamp or stamp[2] != descriptor["size_bytes"]:
            stream.close()
            return None
        return LocalIndexReader(stream, stamp)

    def download(self, descriptor, interrupted, opener=pooled_urlopen):
        """Return True when complete, False when paused or owned by another process."""
        validate_index_file(descriptor)
        directory = self.directory(descriptor)
        directory.mkdir(parents=True, exist_ok=True)
        (self.root / ".teutonic-index-store").touch(exist_ok=True)
        with (directory / "download.lock").open("a+b") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            reader = self.open(descriptor)
            if reader is not None:
                reader.close()
                return True
            partial = directory / "download.part"
            checkpoint = directory / "partial.json"
            receipt = _json(checkpoint)
            with partial.open("a+b") as stream:
                stat = os.fstat(stream.fileno())
                offset = receipt.get("offset", 0)
                if not (
                    receipt.get("descriptor") == descriptor
                    and receipt.get("file") == [stat.st_dev, stat.st_ino]
                    and type(offset) is int and 0 <= offset <= stat.st_size
                    and offset <= descriptor["size_bytes"]
                    and (stat.st_size > offset or receipt.get("stat") == _stamp(stat))
                ):
                    offset = 0
                # A crash can leave bytes beyond the last durable checkpoint.
                if stat.st_size != offset:
                    stream.truncate(offset)

                def save():
                    stream.flush()
                    os.fsync(stream.fileno())
                    stat = os.fstat(stream.fileno())
                    _write_json(checkpoint, {
                        "descriptor": descriptor, "offset": offset,
                        "file": [stat.st_dev, stat.st_ino],
                        "stat": _stamp(stat),
                    })

                while offset < descriptor["size_bytes"]:
                    if interrupted():
                        save()
                        return False
                    start = offset
                    end = min(start + CHUNK_BYTES, descriptor["size_bytes"]) - 1
                    request = Request(descriptor["url"], headers={
                        "Range": f"bytes={start}-{end}", "If-Match": descriptor["etag"],
                        "Accept-Encoding": "identity", "User-Agent": "teutonic-eval/1.0",
                    })
                    try:
                        with opener(request, timeout=10) as response:
                            if (response.status != 206
                                or response.headers.get("Content-Range") != f"bytes {start}-{end}/{descriptor['size_bytes']}"
                                or response.headers.get("ETag") != descriptor["etag"]
                                or response.headers.get("Content-Encoding", "identity") != "identity"):
                                raise ValueError("index prefetch response differs from pinned range")
                            for chunk in response.iter_chunks(STREAM_BYTES):
                                if offset + len(chunk) > end + 1:
                                    raise ValueError("index prefetch returned extra bytes")
                                stream.write(chunk)
                                offset += len(chunk)
                                if interrupted():
                                    save()
                                    return False
                            if offset != end + 1:
                                raise OSError("index prefetch returned a short range")
                    except Exception:
                        # Only committed ranges survive an unsuccessful request.
                        stream.flush()
                        stream.truncate(start)
                        offset = start
                        save()
                        raise
                    save()
                stream.flush()
                os.fsync(stream.fileno())
            target = directory / "index.bin"
            partial.replace(target)
            _write_json(directory / "complete.json", {
                "descriptor": descriptor, "stat": _stamp(target.stat()),
                "verification": "etag-and-size",
            })
            checkpoint.unlink(missing_ok=True)
            log.info("Local document index ready: %s", descriptor["url"])
            return True


class IndexPrefetcher:
    def __init__(self, root, manifest_url, manifest_sha256, workers=4, *, opener=pooled_urlopen):
        if not 1 <= workers <= 16:
            raise ValueError("document index download workers must be between 1 and 16")
        self.store = LocalIndexStore(root)
        self.url, self.sha256, self.workers, self.opener = manifest_url, manifest_sha256, workers, opener
        self.condition = threading.Condition()
        self.paused = False
        self.pause_count = 0
        self.stopped = False
        self.jobs = {}
        self.threads = []
        self.manifest_ready = False
        self.errors = 0

    @classmethod
    def from_env(cls):
        root = store_dir()
        if root is None:
            return None
        model_cache = Path(os.environ.get("TEUTONIC_MODEL_CACHE_DIR", "/tmp/teutonic/pair_models"))
        if root.resolve().is_relative_to(model_cache.resolve()):
            raise ValueError("persistent document indexes must be outside the model cache")
        if os.environ.get("TEUTONIC_DOC_INDEX_PREFETCH", "1") == "0":
            return None
        url = os.environ.get("TEUTONIC_DOC_INDEX_PREFETCH_MANIFEST_URL", "")
        digest = os.environ.get("TEUTONIC_DOC_INDEX_PREFETCH_MANIFEST_SHA256", "")
        if not url.startswith("https://") or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("document index prefetch requires an HTTPS manifest URL and SHA-256 pin")
        return cls(root, url, digest, int(os.environ.get("TEUTONIC_DOC_INDEX_DOWNLOAD_WORKERS", "4")))

    def enqueue_pin(self, pin):
        manifest = validate_index_manifest_pin(pin)
        with self.condition:
            for path, metadata in manifest["files"].items():
                descriptor = {"url": urljoin(pin["url"], path), **metadata}
                self.jobs.setdefault(identity(descriptor), {
                    "descriptor": descriptor, "state": "pending", "retry_at": 0,
                })
            self.condition.notify_all()

    def start(self):
        for target in [self._bootstrap] + [self._worker] * self.workers:
            thread = threading.Thread(target=target, name="document-index-prefetch", daemon=True)
            self.threads.append(thread)
            thread.start()

    def pause(self):
        with self.condition:
            self.pause_count += 1
            self.paused = True

    def resume(self):
        with self.condition:
            self.pause_count = max(0, self.pause_count - 1)
            self.paused = self.pause_count > 0
            self.condition.notify_all()

    def interrupted(self):
        with self.condition:
            return self.paused or self.stopped

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        deadline = time.monotonic() + 2
        for thread in self.threads:
            thread.join(timeout=max(0, deadline - time.monotonic()))

    def status(self):
        with self.condition:
            return {
                "enabled": True, "directory": str(self.store.root),
                "paused": self.paused, "manifest_ready": self.manifest_ready,
                "files_total": len(self.jobs),
                "files_ready": sum(j["state"] == "complete" for j in self.jobs.values()),
                "active_downloads": sum(j["state"] == "active" for j in self.jobs.values()),
                "bytes_ready": sum(j["descriptor"]["size_bytes"] for j in self.jobs.values() if j["state"] == "complete"),
                "errors": self.errors,
            }

    def _bootstrap(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopped or not self.paused)
                if self.stopped:
                    return
            try:
                # Bound this small startup fetch too; it never blocks /health.
                pin = fetch_index_manifest(self.url, self.sha256,
                    opener=lambda request, timeout: self.opener(request, timeout=10))
                self.enqueue_pin(pin)
                with self.condition:
                    self.manifest_ready = True
                return
            except Exception as exc:  # noqa: BLE001 - prefetch must not bring down evaluation
                detail = f"{type(exc).__name__}: {exc}" if isinstance(exc, ValueError) else type(exc).__name__
                log.warning("Document index inventory prefetch failed: %s", detail)
                with self.condition:
                    self.errors += 1
                    self.condition.wait_for(lambda: self.stopped, timeout=60)
                    if self.stopped:
                        return

    def _worker(self):
        while True:
            with self.condition:
                while True:
                    if self.stopped:
                        return
                    job = next((j for j in self.jobs.values() if j["state"] == "pending"
                                and j["retry_at"] <= time.monotonic()), None)
                    if not self.paused and job is not None:
                        job["state"] = "active"
                        break
                    self.condition.wait(timeout=1)
            delay = 1
            complete = False
            try:
                complete = self.store.download(job["descriptor"], self.interrupted, self.opener)
            except Exception as exc:  # noqa: BLE001 - prefetch must not bring down evaluation
                delay = 60
                headers = getattr(exc, "headers", None)
                try:
                    delay = max(delay, float(headers.get("Retry-After", "0"))) if headers else delay
                except (TypeError, ValueError):
                    pass
                log.warning("Document index prefetch deferred: %s", type(exc).__name__)
                with self.condition:
                    self.errors += 1
            with self.condition:
                job["state"] = "complete" if complete else "pending"
                job["retry_at"] = time.monotonic() + delay
                self.condition.notify_all()
