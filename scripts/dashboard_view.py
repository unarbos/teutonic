#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import time
from datetime import timedelta

import boto3
import psycopg
from botocore.config import Config

import chain_config
from teutonic.config import BucketNames
from teutonic.dashboard import (
    DashboardObjectStore,
    DashboardProjectionRepository,
    DashboardViewService,
    KeylessMarketClient,
    MarketClient,
)

log = logging.getLogger("teutonic.dashboard-view")
stopping = False


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return value


def stop(_signum, _frame) -> None:
    global stopping
    stopping = True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish dashboard-v1 from sanitized PostgreSQL views"
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    endpoint = required("TEUTONIC_DASHBOARD_R2_ENDPOINT")
    session = boto3.session.Session()
    client = session.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=required("TEUTONIC_DASHBOARD_R2_ACCESS_KEY_ID"),
        aws_secret_access_key=required("TEUTONIC_DASHBOARD_R2_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("TEUTONIC_DASHBOARD_R2_SESSION_TOKEN") or None,
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
    )
    netuid = int(required("TEUTONIC_NETUID"))
    market_refresh_seconds = int(os.environ.get("TEUTONIC_MARKET_REFRESH_SECONDS", "60"))
    market_url = os.environ.get("TEUTONIC_MARKET_URL", "").strip()
    if market_url:
        market = MarketClient(market_url, source=required("TEUTONIC_MARKET_SOURCE"))
    else:
        market = KeylessMarketClient(
            netuid=netuid,
            network=os.environ.get("TEUTONIC_NETWORK", "finney"),
            refresh_interval=timedelta(seconds=market_refresh_seconds),
        )
    maximum_market_stale = timedelta(
        seconds=int(os.environ.get("TEUTONIC_MARKET_MAX_STALE_SECONDS", "3600"))
    )
    store = DashboardObjectStore(
        client,
        bucket=BucketNames.from_env().dashboard,
        maximum_bytes=int(os.environ.get("TEUTONIC_DASHBOARD_MAX_BYTES", 10 * 1024 * 1024)),
    )
    market_service = DashboardViewService(
        None,
        store,
        market_client=market,
        maximum_market_stale=maximum_market_stale,
    )
    instance = os.environ.get("TEUTONIC_INSTANCE_ID", f"{socket.gethostname()}-{os.getpid()}")
    del instance  # Reserved for service heartbeat integration; never published.
    failures = 0
    database_url = required("TEUTONIC_DATABASE_URL")
    while not stopping:
        repository = None
        try:
            with psycopg.connect(database_url, autocommit=True) as connection:
                repository = DashboardProjectionRepository(
                    connection,
                    netuid=netuid,
                    chain_generation=required("TEUTONIC_CHAIN_GENERATION"),
                    competition=required("TEUTONIC_COMPETITION"),
                    chain_name=os.environ.get("TEUTONIC_CHAIN_NAME", "Teutonic"),
                    seed_repo=chain_config.SEED_REPO,
                    seed_digest=chain_config.SEED_DIGEST,
                    seed_repo_backend=chain_config.SEED_REPO_BACKEND,
                )
                if not repository.acquire_lock():
                    raise RuntimeError("another dashboard-view publisher holds the advisory lock")
                service = DashboardViewService(
                    repository,
                    store,
                    market_client=market,
                    maximum_market_stale=maximum_market_stale,
                )
                try:
                    while not stopping:
                        try:
                            result, dataset_result, active = service.publish_once()
                            rows = connection.execute(
                                "SELECT competition, competition_key, has_king, eval_n, delta_threshold FROM control_plane.dashboard_competitions WHERE netuid=%s AND chain_generation=%s AND main_competition=%s ORDER BY competition_key",
                                (netuid, repository.chain_generation, repository.competition),
                            ).fetchall()
                            index = []
                            for name, key, has_king, eval_n, delta in rows:
                                prefix = "" if key == "main" else f"competitions/{key}/"
                                index.append(
                                    {
                                        "competition": key,
                                        "name": name,
                                        "has_king": has_king,
                                        "eval_n": eval_n,
                                        "delta_threshold": delta,
                                        "dashboard": prefix + "dashboard.json",
                                    }
                                )
                                if key == "main":
                                    continue
                                split_repository = repository.competition_scope(name)
                                split_store = DashboardObjectStore(
                                    client,
                                    bucket=store.bucket,
                                    maximum_bytes=store.maximum_bytes,
                                    prefix=prefix,
                                )
                                _, _, split_active = DashboardViewService(
                                    split_repository,
                                    split_store,
                                    market_client=market,
                                    maximum_market_stale=maximum_market_stale,
                                ).publish_once()
                                active = active or split_active
                            client.put_object(
                                Bucket=store.bucket,
                                Key="competitions.json",
                                Body=json.dumps({"competitions": index}).encode(),
                                ContentType="application/json",
                                CacheControl="public, max-age=15, must-revalidate",
                            )
                            failures = 0
                            log.info(
                                "dashboard %s bytes=%d sha256=%s",
                                result.state,
                                result.size_bytes,
                                result.sha256,
                            )
                            log.info(
                                "dataset manifest %s bytes=%d sha256=%s",
                                dataset_result.state,
                                dataset_result.size_bytes,
                                dataset_result.sha256,
                            )
                            if args.once:
                                return 0
                            time.sleep(5 if active else 20)
                        except psycopg.Error:
                            raise
                        except Exception as exc:
                            failures += 1
                            log.error("dashboard publication failed type=%s", type(exc).__name__)
                            overlay = market_service.publish_market_only()
                            log.info(
                                "market overlay %s bytes=%d sha256=%s",
                                overlay.state,
                                overlay.size_bytes,
                                overlay.sha256,
                            )
                            if args.once:
                                raise
                            time.sleep(min(60, 2 ** min(failures, 5)))
                finally:
                    repository.release_lock()
        except psycopg.Error as exc:
            failures += 1
            log.error("dashboard database unavailable type=%s", type(exc).__name__)
            try:
                overlay = market_service.publish_market_only()
                log.info(
                    "market overlay %s bytes=%d sha256=%s",
                    overlay.state,
                    overlay.size_bytes,
                    overlay.sha256,
                )
            except Exception as market_exc:
                log.error("market overlay failed type=%s", type(market_exc).__name__)
            if args.once:
                raise
            time.sleep(market_refresh_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
