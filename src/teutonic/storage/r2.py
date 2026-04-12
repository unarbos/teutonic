"""Cloudflare R2 storage backend with persistent connections and zstd compression."""

from __future__ import annotations

import asyncio
import io
import logging
from contextlib import asynccontextmanager
from typing import Any

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

logger = logging.getLogger(__name__)

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

    async def _retry(self, fn, *args, **kwargs):
        last_exc = None
        for attempt in range(_RETRIES):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                wait = _BACKOFF_BASE * (2**attempt)
                logger.warning("R2 retry %d/%d: %s", attempt + 1, _RETRIES, exc)
                await asyncio.sleep(wait)
        raise last_exc

    async def put(self, key: str, data: dict[str, Any]) -> None:
        s3_key = self._s3_key(key)
        body = await asyncio.to_thread(_serialize, data)
        client = await self._get_client()
        try:
            async with self._sem:
                await asyncio.wait_for(
                    self._retry(
                        client.put_object, Bucket=self._bucket, Key=s3_key, Body=body
                    ),
                    timeout=self._put_timeout,
                )
        except asyncio.TimeoutError:
            logger.error("R2 put timed out after %.0fs for %s", self._put_timeout, key)
            raise

    async def get(self, key: str) -> dict[str, Any] | None:
        s3_key = self._s3_key(key)
        client = await self._get_client()

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
                        logger.warning("R2 get timeout %d/%d for %s", attempt + 1, _RETRIES, key)
                    except Exception as exc:
                        last_exc = exc
                        wait = _BACKOFF_BASE * (2**attempt)
                        logger.warning("R2 get retry %d/%d: %s", attempt + 1, _RETRIES, exc)
                        await asyncio.sleep(wait)
                else:
                    if last_exc is not None:
                        logger.error("R2 get failed after %d retries for %s", _RETRIES, key)
                        return None

                if blob is None:
                    return None
        except asyncio.TimeoutError:
            logger.error("R2 get timed out for %s", key)
            return None

        try:
            return await asyncio.to_thread(_deserialize, blob)
        except Exception:
            logger.warning("Corrupt R2 object at %s", s3_key)
            return None

    async def list_keys(self, prefix: str) -> list[str]:
        s3_prefix = f"{self._prefix}{prefix}"
        keys: list[str] = []
        client = await self._get_client()

        async def _do_list():
            paginator = client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(
                Bucket=self._bucket, Prefix=s3_prefix
            ):
                for obj in page.get("Contents", []):
                    keys.append(self._key_from_s3(obj["Key"]))

        try:
            async with self._sem:
                await asyncio.wait_for(_do_list(), timeout=self._list_timeout)
        except asyncio.TimeoutError:
            logger.error("R2 list_keys timed out after %.0fs for prefix %s", self._list_timeout, prefix)
        return sorted(keys)

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
