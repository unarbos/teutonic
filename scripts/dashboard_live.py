#!/usr/bin/env python3
"""Publish queue and progress independently of slow history/market refreshes."""
from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
from time import monotonic

import boto3
import psycopg
from botocore.config import Config

from teutonic.config import BucketNames
from teutonic.dashboard.live import LiveDashboardRepository, publish_live

log = logging.getLogger("teutonic.dashboard-live")


def required(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    stopping = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.set())
    client = boto3.client(
        "s3", endpoint_url=required("TEUTONIC_DASHBOARD_R2_ENDPOINT"),
        aws_access_key_id=required("TEUTONIC_DASHBOARD_R2_ACCESS_KEY_ID"),
        aws_secret_access_key=required("TEUTONIC_DASHBOARD_R2_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("TEUTONIC_DASHBOARD_R2_SESSION_TOKEN") or None,
        region_name="auto",
        config=Config(signature_version="s3v4", connect_timeout=2, read_timeout=3,
                      retries={"total_max_attempts": 1}),
    )
    bucket = BucketNames.from_env().dashboard
    while not stopping.is_set():
        try:
            with psycopg.connect(required("TEUTONIC_DATABASE_URL"), autocommit=True,
                                 connect_timeout=5, application_name="teutonic-dashboard-live") as c:
                repository = LiveDashboardRepository(
                    c, netuid=int(required("TEUTONIC_NETUID")),
                    chain_generation=required("TEUTONIC_CHAIN_GENERATION"),
                    competition=required("TEUTONIC_COMPETITION"),
                )
                if not repository.acquire_lock():
                    raise RuntimeError("another live publisher holds the advisory lock")
                try:
                    while not stopping.is_set():
                        started = monotonic()
                        payload = repository.project()
                        size = publish_live(client, bucket, payload)
                        log.info("published bytes=%d projection_and_upload_seconds=%.3f watermark=%d",
                                 size, monotonic() - started, payload["source_watermark"])
                        if args.once:
                            return 0
                        stopping.wait(max(0, 5 - (monotonic() - started)))
                finally:
                    repository.release_lock()
        except Exception as exc:
            # Keep the last good object with its original timestamp. Never turn
            # a database/network error into a fresh, empty queue publication.
            log.error("live publication failed type=%s", type(exc).__name__)
            if args.once:
                raise
            stopping.wait(5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
