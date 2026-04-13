"""R2 bucket operations for eval results, state, and audit trail."""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import boto3
from botocore.config import Config as BotoConfig

from .config import R2Config

logger = logging.getLogger(__name__)


class R2Client:
    """Thin wrapper around boto3 S3 client for Cloudflare R2."""

    def __init__(self, cfg: R2Config):
        self.cfg = cfg
        self._client = boto3.client(
            "s3",
            endpoint_url=cfg.endpoint_url,
            aws_access_key_id=cfg.access_key_id,
            aws_secret_access_key=cfg.secret_access_key,
            region_name=cfg.region,
            config=BotoConfig(
                retries={"max_attempts": 3, "mode": "adaptive"},
                connect_timeout=10,
                read_timeout=30,
            ),
        )
        self.bucket = cfg.bucket_name

    def put_json(self, key: str, data: Any) -> None:
        body = json.dumps(data, default=str).encode()
        self._client.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType="application/json")
        logger.debug("R2 PUT %s (%d bytes)", key, len(body))

    def get_json(self, key: str) -> Any | None:
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
            return json.loads(resp["Body"].read())
        except self._client.exceptions.NoSuchKey:
            return None
        except Exception:
            logger.exception("R2 GET %s failed", key)
            return None

    def append_jsonl(self, key: str, record: dict) -> None:
        """Append a single JSON line to an existing JSONL file, or create it."""
        line = json.dumps(record, default=str) + "\n"
        existing = b""
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
            existing = resp["Body"].read()
        except self._client.exceptions.NoSuchKey:
            pass
        except Exception:
            logger.warning("R2 append_jsonl: could not read existing %s", key)

        new_body = existing + line.encode()
        self._client.put_object(Bucket=self.bucket, Key=key, Body=new_body, ContentType="application/x-ndjson")

    def get_jsonl(self, key: str) -> list[dict]:
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
            lines = resp["Body"].read().decode().strip().split("\n")
            return [json.loads(line) for line in lines if line.strip()]
        except self._client.exceptions.NoSuchKey:
            return []
        except Exception:
            logger.exception("R2 GET JSONL %s failed", key)
            return []

    def download_file(self, key: str, local_path: str) -> None:
        self._client.download_file(self.bucket, key, local_path)

    def upload_file(self, local_path: str, key: str) -> None:
        self._client.upload_file(local_path, self.bucket, key)

    def list_keys(self, prefix: str) -> list[str]:
        keys = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def key_exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False
