from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from botocore.exceptions import ClientError

DASHBOARD_KEY = "dashboard.json"
DATASET_MANIFEST_KEY = "datasets/manifest.json"


@dataclass(frozen=True, slots=True)
class PublicationResult:
    state: str
    sha256: str
    size_bytes: int


class DashboardObjectStore:
    def __init__(
        self,
        client,
        *,
        bucket: str,
        maximum_bytes: int = 10 * 1024 * 1024,
        cache_control: str = "public, max-age=15, must-revalidate",
        prefix: str = "",
    ) -> None:
        if not bucket:
            raise ValueError("dashboard bucket is required")
        if maximum_bytes < 1024:
            raise ValueError("dashboard maximum object size is too small")
        if prefix not in {"", "competitions/math/", "competitions/code/", "competitions/text/"}:
            raise ValueError("invalid competition publication prefix")
        self.prefix = prefix
        self.client = client
        self.bucket = bucket
        self.maximum_bytes = maximum_bytes
        self.cache_control = cache_control

    def previous_payload(self) -> dict[str, Any] | None:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.prefix + DASHBOARD_KEY)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise
        body = response["Body"].read(self.maximum_bytes + 1)
        if len(body) > self.maximum_bytes:
            return None
        try:
            payload = json.loads(body.decode("utf-8", errors="strict"))
        except (UnicodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) and payload.get("schema_version") == 1 else None

    def publish(self, body: bytes, *, source_watermark: int) -> PublicationResult:
        if len(body) > self.maximum_bytes:
            raise ValueError(
                f"complete dashboard object is {len(body)} bytes; limit is {self.maximum_bytes}"
            )
        digest = hashlib.sha256(body).hexdigest()
        existing = self._head()
        if existing is not None:
            metadata = existing.get("Metadata", {})
            try:
                previous_watermark = int(metadata.get("source-watermark", "-1"))
            except ValueError:
                previous_watermark = -1
            if previous_watermark > source_watermark:
                return PublicationResult("stale_skipped", digest, len(body))
            if (
                previous_watermark == source_watermark
                and metadata.get("content-sha256") == digest
            ):
                return PublicationResult("unchanged", digest, len(body))
        self.client.put_object(
            Bucket=self.bucket,
            Key=self.prefix + DASHBOARD_KEY,
            Body=body,
            ContentType="application/json; charset=utf-8",
            CacheControl=self.cache_control,
            Metadata={
                "schema-version": "1",
                "source-watermark": str(source_watermark),
                "content-sha256": digest,
            },
        )
        confirmed = self._head()
        if confirmed is None or int(confirmed.get("ContentLength", -1)) != len(body):
            raise RuntimeError("dashboard object verification failed")
        metadata = confirmed.get("Metadata", {})
        if metadata.get("content-sha256") != digest:
            raise RuntimeError("dashboard content hash metadata verification failed")
        return PublicationResult("published", digest, len(body))

    def publish_dataset_manifest(
        self, body: bytes, *, config_version: str
    ) -> PublicationResult:
        if len(body) > self.maximum_bytes:
            raise ValueError(
                f"global dataset manifest is {len(body)} bytes; limit is {self.maximum_bytes}"
            )
        digest = hashlib.sha256(body).hexdigest()
        existing = self._head(DATASET_MANIFEST_KEY)
        if existing is not None and existing.get("Metadata", {}).get(
            "content-sha256"
        ) == digest:
            return PublicationResult("unchanged", digest, len(body))
        self.client.put_object(
            Bucket=self.bucket,
            Key=self.prefix + DATASET_MANIFEST_KEY,
            Body=body,
            ContentType="application/json; charset=utf-8",
            CacheControl=self.cache_control,
            Metadata={
                "schema-version": "1",
                "config-version": config_version,
                "content-sha256": digest,
            },
        )
        confirmed = self._head(DATASET_MANIFEST_KEY)
        if confirmed is None or int(confirmed.get("ContentLength", -1)) != len(body):
            raise RuntimeError("global dataset manifest verification failed")
        if confirmed.get("Metadata", {}).get("content-sha256") != digest:
            raise RuntimeError("global dataset manifest hash metadata verification failed")
        return PublicationResult("published", digest, len(body))

    def _head(self, key: str = DASHBOARD_KEY) -> Mapping[str, Any] | None:
        try:
            return self.client.head_object(Bucket=self.bucket, Key=self.prefix + key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise
