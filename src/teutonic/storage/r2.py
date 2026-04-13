"""Cloudflare R2 storage backend with persistent connections and zstd compression."""

from __future__ import annotations

import asyncio
import io
import time
from contextlib import asynccontextmanager
from typing import Any

import structlog
import torch

try:
    import zstandard as zstd

    _HAS_ZSTD = True
    _CCTX = zstd.ZstdCompressor(level=3)
    _DCTX = zstd.ZstdDecompressor()
except ImportError:
    _HAS_ZSTD = False
    _CCTX = None
    _DCTX = None

import aioboto3
from botocore.config import Config as BotoConfig

logger = structlog.get_logger(__name__)

_RETRIES = 3
_BACKOFF_BASE = 0.1


def _serialize(data: dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    torch.save(data, buf)
    raw = buf.getvalue()
    if _HAS_ZSTD:
        return _CCTX.compress(raw)
    return raw


def _deserialize(blob: bytes) -> dict[str, Any]:
    if _HAS_ZSTD:
        try:
            raw = _DCTX.decompress(blob)
        except zstd.ZstdError:
            raw = blob
    else:
        raw = blob
    buf = io.BytesIO(raw)
    return torch.load(buf, weights_only=True)


class R2Storage:
    """S3-compatible storage for Cloudflare R2.

    Performance optimizations over the previous version:
    - Persistent S3 client (single TLS handshake, connection reuse)
    - Serialization/compression offloaded to thread pool
    - Module-level compressor/decompressor instances (no per-call allocation)
    - Batch delete (up to 1000 keys per S3 DeleteObjects call)
    """

    def __init__(
        self,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
        prefix: str = "teutonic/",
        max_concurrent: int = 32,
        put_timeout: float = 60.0,
        get_timeout: float = 30.0,
        list_timeout: float = 30.0,
    ):
        self._endpoint = endpoint_url
        self._access_key = access_key_id
        self._secret_key = secret_access_key
        self._bucket = bucket_name
        self._prefix = prefix
        self._sem = asyncio.Semaphore(max_concurrent)
        self._put_timeout = put_timeout
        self._get_timeout = get_timeout
        self._list_timeout = list_timeout
        self._session = aioboto3.Session()
        self._boto_cfg = BotoConfig(
            retries={"max_attempts": 0},
            max_pool_connections=max_concurrent,
            connect_timeout=5,
            read_timeout=15,
        )
        self._client_cm = None
        self._client = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self):
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is not None:
                return self._client
            self._client_cm = self._session.client(
                "s3",
                endpoint_url=self._endpoint,
                aws_access_key_id=self._access_key,
                aws_secret_access_key=self._secret_key,
                region_name="auto",
                config=self._boto_cfg,
            )
            self._client = await self._client_cm.__aenter__()
            logger.info("storage.r2.connected", endpoint=self._endpoint, bucket=self._bucket)
            return self._client

    def _s3_key(self, key: str) -> str:
        suffix = ".pt.zst" if _HAS_ZSTD else ".pt"
        return f"{self._prefix}{key}{suffix}"

    def _key_from_s3(self, s3_key: str) -> str:
        k = s3_key
        if k.startswith(self._prefix):
            k = k[len(self._prefix) :]
        for suffix in (".pt.zst", ".pt"):
            if k.endswith(suffix):
                k = k[: -len(suffix)]
                break
        return k

    async def _retry(self, fn, *args, op: str = "unknown", key: str = "", **kwargs):
        last_exc = None
        for attempt in range(_RETRIES):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                wait = _BACKOFF_BASE * (2**attempt)
                logger.warning(
                    "storage.r2.retry",
                    op=op, key=key,
                    attempt=attempt + 1, max_retries=_RETRIES,
                    error=str(exc),
                )
                await asyncio.sleep(wait)
        logger.error("storage.r2.failed", op=op, key=key, retries=_RETRIES, error=str(last_exc))
        raise last_exc

    async def put(self, key: str, data: dict[str, Any]) -> None:
        s3_key = self._s3_key(key)
        t0 = time.monotonic()
        body = await asyncio.to_thread(_serialize, data)
        size_bytes = len(body)
        client = await self._get_client()
        try:
            async with self._sem:
                await asyncio.wait_for(
                    self._retry(
                        client.put_object, Bucket=self._bucket, Key=s3_key, Body=body,
                        op="put", key=key,
                    ),
                    timeout=self._put_timeout,
                )
        except asyncio.TimeoutError:
            logger.error(
                "storage.r2.put.timeout",
                key=key, timeout_s=self._put_timeout, size_bytes=size_bytes,
            )
            raise
        logger.debug(
            "storage.r2.put",
            key=key, size_bytes=size_bytes,
            duration_s=round(time.monotonic() - t0, 3),
        )

    async def get(self, key: str) -> dict[str, Any] | None:
        s3_key = self._s3_key(key)
        client = await self._get_client()
        t0 = time.monotonic()

        async def _do_get():
            try:
                resp = await client.get_object(Bucket=self._bucket, Key=s3_key)
                return await resp["Body"].read()
            except client.exceptions.NoSuchKey:
                return None
            except Exception as exc:
                if "NoSuchKey" in str(exc) or "404" in str(exc):
                    return None
                raise

        try:
            async with self._sem:
                blob = None
                last_exc = None
                for attempt in range(_RETRIES):
                    try:
                        blob = await asyncio.wait_for(
                            _do_get(), timeout=self._get_timeout
                        )
                        break
                    except asyncio.TimeoutError:
                        last_exc = asyncio.TimeoutError(
                            f"R2 get timed out after {self._get_timeout}s for {key}"
                        )
                        logger.warning(
                            "storage.r2.get.timeout",
                            key=key, attempt=attempt + 1, max_retries=_RETRIES,
                        )
                    except Exception as exc:
                        last_exc = exc
                        wait = _BACKOFF_BASE * (2**attempt)
                        logger.warning(
                            "storage.r2.get.retry",
                            key=key, attempt=attempt + 1, max_retries=_RETRIES,
                            error=str(exc),
                        )
                        await asyncio.sleep(wait)
                else:
                    if last_exc is not None:
                        logger.error(
                            "storage.r2.get.failed",
                            key=key, retries=_RETRIES, error=str(last_exc),
                        )
                        return None

                if blob is None:
                    return None
        except asyncio.TimeoutError:
            logger.error("storage.r2.get.timeout", key=key)
            return None

        try:
            result = await asyncio.to_thread(_deserialize, blob)
            logger.debug(
                "storage.r2.get",
                key=key, size_bytes=len(blob),
                duration_s=round(time.monotonic() - t0, 3),
            )
            return result
        except Exception:
            logger.warning("storage.r2.get.corrupt", key=key, s3_key=s3_key)
            return None

    async def list_keys(self, prefix: str) -> list[str]:
        items = await self.list_keys_with_metadata(prefix)
        return [item["key"] for item in items]

    async def list_keys_with_metadata(self, prefix: str) -> list[dict[str, Any]]:
        s3_prefix = f"{self._prefix}{prefix}"
        items: list[dict[str, Any]] = []
        client = await self._get_client()

        async def _do_list():
            paginator = client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(
                Bucket=self._bucket, Prefix=s3_prefix
            ):
                for obj in page.get("Contents", []):
                    lm = obj.get("LastModified")
                    items.append({
                        "key": self._key_from_s3(obj["Key"]),
                        "last_modified": lm.timestamp() if lm else 0.0,
                    })

        try:
            async with self._sem:
                await asyncio.wait_for(_do_list(), timeout=self._list_timeout)
        except asyncio.TimeoutError:
            logger.error(
                "storage.r2.list.timeout",
                prefix=prefix, timeout_s=self._list_timeout,
            )
        logger.debug("storage.r2.list", prefix=prefix, n_keys=len(items))
        return sorted(items, key=lambda x: x["key"])

    async def delete(self, key: str) -> None:
        s3_key = self._s3_key(key)
        client = await self._get_client()
        async with self._sem:
            await client.delete_object(Bucket=self._bucket, Key=s3_key)

    async def delete_prefix(self, prefix: str) -> int:
        """Delete all objects under *prefix* using batch delete."""
        keys = await self.list_keys(prefix)
        if not keys:
            return 0
        client = await self._get_client()
        s3_keys = [self._s3_key(k) for k in keys]
        # S3 batch delete: up to 1000 per call
        deleted = 0
        for i in range(0, len(s3_keys), 1000):
            batch = s3_keys[i : i + 1000]
            async with self._sem:
                await client.delete_objects(
                    Bucket=self._bucket,
                    Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
                )
            deleted += len(batch)
        return deleted

    async def close(self) -> None:
        if self._client_cm is not None:
            await self._client_cm.__aexit__(None, None, None)
            self._client = None
            self._client_cm = None
            logger.info("storage.r2.closed")
