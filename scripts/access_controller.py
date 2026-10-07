#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import time
from datetime import datetime, timedelta, timezone

import boto3
import bittensor as bt
import httpx
import psycopg
from botocore.config import Config
from nacl.signing import SigningKey

import chain_config

from teutonic.access import (
    AccessControllerJobRunner,
    AccessControllerRepository,
    ControllerLockUnavailable,
    FinalizedChainScanner,
    MailboxCipher,
    MailboxStore,
    R2UploadController,
)
from teutonic.access.cloudflare import CloudflareR2TokenGateway
from teutonic.access.crypto import SecretCipher
from teutonic.config import BucketNames


log = logging.getLogger("teutonic.access-controller")
stopping = False
STATUS_LOG_SECONDS = 60.0
UPLOAD_QUOTA_SCAN_SECONDS = 5.0
FAILED_MODEL_CLEANUP_SCAN_SECONDS = 5.0
CREDENTIAL_RENEWAL_SCAN_SECONDS = 60.0


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return value


def secret_key(name: str) -> bytes:
    try:
        value = bytes.fromhex(required(name))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be hexadecimal") from exc
    if len(value) != 32:
        raise RuntimeError(f"{name} must contain exactly 32 bytes")
    return value


def stop(_signum, _frame) -> None:
    global stopping
    stopping = True


def r2_endpoint() -> str:
    configured = os.environ.get("TEUTONIC_R2_ENDPOINT", "").strip()
    if configured:
        return configured
    return f"https://{required('CLOUDFLARE_ACCOUNT_ID')}.r2.cloudflarestorage.com"


def r2_client():
    return boto3.client(
        "s3",
        endpoint_url=r2_endpoint(),
        aws_access_key_id=required("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=required("R2_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("R2_SESSION_TOKEN") or None,
        region_name=os.environ.get("TEUTONIC_R2_REGION", "auto"),
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
            max_pool_connections=64,
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run durable Cloudflare credential and private-upload controller jobs"
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    poll_seconds = float(os.environ.get("TEUTONIC_ACCESS_CONTROLLER_POLL_SECONDS", "2"))
    if poll_seconds <= 0:
        raise ValueError("access-controller poll interval must be positive")
    instance = os.environ.get(
        "TEUTONIC_ACCESS_CONTROLLER_INSTANCE_ID",
        f"{socket.gethostname()}-{os.getpid()}",
    )
    buckets = BucketNames.from_env()
    endpoint = r2_endpoint()
    log.info(
        "access controller initializing instance=%s network=%s netuid=%s",
        instance,
        required("TEUTONIC_NETWORK"),
        required("TEUTONIC_NETUID"),
    )
    s3 = r2_client()

    with (
        psycopg.connect(required("TEUTONIC_DATABASE_URL"), autocommit=True) as connection,
        httpx.Client(timeout=30.0) as http,
        bt.Subtensor(network=required("TEUTONIC_NETWORK")) as subtensor,
    ):
        repository = AccessControllerRepository(
            connection,
            finalized_start_block=int(
                os.environ.get("TEUTONIC_FINALIZED_START_BLOCK", "0")
            ),
        )
        scanner = FinalizedChainScanner(
            subtensor,
            netuid=int(required("TEUTONIC_NETUID")),
            chain_generation=required("TEUTONIC_CHAIN_GENERATION"),
        )
        runner = AccessControllerJobRunner(
            repository,
            token_gateway=CloudflareR2TokenGateway(
                http,
                account_id=required("CLOUDFLARE_ACCOUNT_ID"),
                management_token=required("CLOUDFLARE_API_TOKEN"),
                bucket=buckets.private_models,
            ),
            upload_controller=R2UploadController(
                s3,
                private_model_bucket=buckets.private_models,
                genesis_contract_files=chain_config.GENESIS_CONTRACT_FILES,
                chunk_size=8 * 1024 * 1024,
            ),
            mailbox_store=MailboxStore(s3, bucket=buckets.dashboard),
            secret_cipher=SecretCipher(secret_key("TEUTONIC_CONTROLLER_SECRET_KEY")),
            mailbox_cipher=MailboxCipher(
                SigningKey(secret_key("TEUTONIC_MAILBOX_SIGNING_KEY"))
            ),
            account_id=required("CLOUDFLARE_ACCOUNT_ID"),
            r2_endpoint=endpoint,
            private_model_bucket=buckets.private_models,
            instance_id=instance,
            lease=timedelta(
                seconds=int(os.environ.get("TEUTONIC_ACCESS_CONTROLLER_LEASE_SECONDS", "120"))
            ),
            retry_delay=timedelta(
                seconds=int(os.environ.get("TEUTONIC_ACCESS_CONTROLLER_RETRY_SECONDS", "5"))
            ),
        )
        log.info(
            "access controller active instance=%s network=%s netuid=%s "
            "private_bucket=%s mailbox_bucket=%s poll_seconds=%s upload_limit_bytes=%d",
            instance,
            required("TEUTONIC_NETWORK"),
            required("TEUTONIC_NETUID"),
            buckets.private_models,
            buckets.dashboard,
            poll_seconds,
            runner.upload_controller.max_upload_bytes,
        )
        next_chain_scan = 0.0
        next_upload_quota_scan = 0.0
        next_failed_model_cleanup_scan = 0.0
        next_credential_renewal_scan = 0.0
        next_status_log = 0.0
        mailboxes_reconciled = False
        while not stopping:
            acquired = False
            try:
                repository.acquire_lock()
                acquired = True
                if not mailboxes_reconciled:
                    removed = runner.reconcile_revoked_mailboxes()
                    mailboxes_reconciled = True
                    if removed:
                        log.info(
                            "removed %d revoked mailbox credential objects",
                            removed,
                        )
                scanned = accepted = 0
                if time.monotonic() >= next_chain_scan:
                    next_chain_scan = time.monotonic() + 6.0
                    scanned, accepted = scanner.scan(repository)
                quota_revocations = 0
                if time.monotonic() >= next_upload_quota_scan:
                    next_upload_quota_scan = (
                        time.monotonic() + UPLOAD_QUOTA_SCAN_SECONDS
                    )
                    quota_revocations = runner.enforce_upload_quotas()
                reuse_limit_cleanups = 0
                if time.monotonic() >= next_failed_model_cleanup_scan:
                    next_failed_model_cleanup_scan = (
                        time.monotonic() + FAILED_MODEL_CLEANUP_SCAN_SECONDS
                    )
                    reuse_limit_cleanups = runner.schedule_reuse_limit_cleanups()
                recovered = repository.recover_expired_jobs(
                    now=datetime.now(timezone.utc)
                )
                if time.monotonic() >= next_credential_renewal_scan:
                    renewed = repository.schedule_credential_renewals(
                        now=datetime.now(timezone.utc),
                        netuid=int(required("TEUTONIC_NETUID")),
                        chain_generation=required("TEUTONIC_CHAIN_GENERATION"),
                        private_model_bucket=buckets.private_models,
                    )
                    next_credential_renewal_scan = (
                        time.monotonic() + CREDENTIAL_RENEWAL_SCAN_SECONDS
                    )
                    if renewed:
                        log.info("scheduled %d credential renewals", renewed)
                processed = runner.run_until_idle(maximum_jobs=1000)
                if (
                    scanned
                    or accepted
                    or quota_revocations
                    or reuse_limit_cleanups
                    or recovered
                    or processed
                ):
                    log.info(
                        "controller chain_blocks=%d signals=%d quota_revocations=%d "
                        "reuse_limit_cleanups=%d jobs_recovered=%d jobs_processed=%d",
                        scanned,
                        accepted,
                        quota_revocations,
                        reuse_limit_cleanups,
                        recovered,
                        processed,
                    )
                elif time.monotonic() >= next_status_log:
                    log.info("controller heartbeat status=idle instance=%s", instance)
                if time.monotonic() >= next_status_log:
                    next_status_log = time.monotonic() + STATUS_LOG_SECONDS
            except ControllerLockUnavailable:
                log.info("controller lock busy; retrying")
            except Exception:
                log.exception("access-controller cycle failed")
                if args.once:
                    raise
            finally:
                if acquired:
                    repository.release_lock()
            if args.once:
                return 0
            time.sleep(poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
