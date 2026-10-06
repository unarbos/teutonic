#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import socket
from datetime import datetime, timezone

import psycopg

from teutonic.config import BucketNames
from teutonic.evaluation import EvaluationRequestV2, HttpEvaluatorClient
from teutonic.validator import (
    BittensorFinalizedMetagraphReader,
    CrownCoordinator,
    ValidatorRepository,
    ValidatorScheduler,
    evaluation_policy_from_env,
)


SOFTWARE_VERSION = "postgres-validator-splits-v1"
log = logging.getLogger("teutonic.validator")
stopping = False
STATUS_LOG_SECONDS = 60.0


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return value


def stop(_signum, _frame) -> None:
    global stopping
    stopping = True


async def contract_preflight(request):
    EvaluationRequestV2.from_mapping(request)
    return None


async def heartbeat_loop(
    repository, phase: dict[str, str], interval: float, instance: str
) -> None:
    next_status_log = 0.0
    while not stopping:
        repository.heartbeat_service(
            now=datetime.now(timezone.utc),
            phase=phase["value"],
            software_version=SOFTWARE_VERSION,
        )
        now = asyncio.get_running_loop().time()
        if now >= next_status_log:
            log.info(
                "validator heartbeat status=%s instance=%s",
                phase["value"],
                instance,
            )
            next_status_log = now + STATUS_LOG_SECONDS
        await asyncio.sleep(interval)


def refresh_weight_plan(coordinator) -> bool:
    try:
        refreshed = coordinator.reconcile_current_weight_plan()
        if refreshed:
            log.info("refreshed current weight plan after finalized UID remap")
        return refreshed
    except Exception:
        log.exception("current weight plan refresh failed")
        return False


async def weight_plan_refresh_loop(coordinator, interval: float) -> None:
    while not stopping:
        refresh_weight_plan(coordinator)
        await asyncio.sleep(interval)


async def run(*, once: bool) -> int:
    database_url = required("TEUTONIC_DATABASE_URL")
    network = required("TEUTONIC_NETWORK")
    netuid = int(required("TEUTONIC_NETUID"))
    generation = required("TEUTONIC_CHAIN_GENERATION")
    competition = required("TEUTONIC_COMPETITION")
    evaluator_url = required("TEUTONIC_EVAL_SERVER")
    instance = os.environ.get(
        "TEUTONIC_VALIDATOR_INSTANCE_ID",
        os.environ.get("TEUTONIC_INSTANCE_ID", f"{socket.gethostname()}-{os.getpid()}"),
    )
    buckets = BucketNames.from_env()
    poll_seconds = float(os.environ.get("TEUTONIC_VALIDATOR_POLL_SECONDS", "12"))
    heartbeat_seconds = float(os.environ.get("TEUTONIC_HEARTBEAT_SECONDS", "30"))
    weight_refresh_seconds = float(
        os.environ.get("TEUTONIC_WEIGHT_PLAN_REFRESH_SECONDS", "12")
    )
    if poll_seconds <= 0 or heartbeat_seconds <= 0 or weight_refresh_seconds <= 0:
        raise ValueError("validator poll, heartbeat, and weight refresh intervals must be positive")
    log.info(
        "validator initializing instance=%s network=%s netuid=%d competition=%s",
        instance,
        network,
        netuid,
        competition,
    )
    with psycopg.connect(database_url, autocommit=True) as connection:
        repository = ValidatorRepository(
            connection,
            netuid=netuid,
            chain_generation=generation,
            competition=competition,
            instance_id=instance,
            public_model_bucket=buckets.public_models,
        )
        repository.acquire_lock()
        policy = evaluation_policy_from_env(
            settings=repository.load_evaluation_settings(),
            early_stopping=repository.load_early_stopping_policy(),
        )
        log.info(
            "loaded evaluation config dataset=%s n=%d delta=%s manifests=%d "
            "early_stop=%s min_fraction=%s advantage_quantile=%s margin=%s "
            "check_interval=%d",
            policy.dataset_version,
            policy.n,
            policy.delta_threshold,
            len(policy.dataset_manifests),
            policy.early_stopping.enabled,
            policy.early_stopping.min_fraction,
            policy.early_stopping.advantage_quantile,
            policy.early_stopping.margin,
            policy.early_stopping.check_interval,
        )
        chain = BittensorFinalizedMetagraphReader(network=network, netuid=netuid)
        coordinator = CrownCoordinator(
            repository,
            chain,
            burn_uid=int(os.environ.get("TEUTONIC_BURN_UID", "0")),
            king_chain_size=int(os.environ.get("TEUTONIC_KING_CHAIN_SIZE", "5")),
        )

        phase = {"value": "starting"}

        async with HttpEvaluatorClient(evaluator_url) as evaluator:
            scheduler = ValidatorScheduler(
                repository,
                evaluator,
                policy=policy,
                policy_loader=lambda: evaluation_policy_from_env(
                    settings=repository.load_evaluation_settings(),
                    early_stopping=repository.load_early_stopping_policy(),
                ),
                preflight=contract_preflight,
            )
            heartbeat_task = asyncio.create_task(
                heartbeat_loop(repository, phase, heartbeat_seconds, instance)
            )
            weight_refresh_task = asyncio.create_task(
                weight_plan_refresh_loop(coordinator, weight_refresh_seconds)
            )
            try:
                phase["value"] = "reconciling_evaluations"
                recovered = await scheduler.reconcile()
                log.info(
                    "validator active instance=%s network=%s netuid=%d competition=%s recovered=%d poll_seconds=%s",
                    instance,
                    network,
                    netuid,
                    competition,
                    recovered,
                    poll_seconds,
                )
                while not stopping:
                    phase["value"] = "reconciling_evaluations"
                    cycle_recovered = await scheduler.reconcile()
                    if cycle_recovered:
                        log.info(
                            "validator cycle recovered_evaluations=%d",
                            cycle_recovered,
                        )
                    phase["value"] = "evaluating"
                    evaluated = await scheduler.run_once()
                    if evaluated:
                        log.info("validator cycle completed evaluation_work=true")
                    phase["value"] = "refreshing_weight_plan"
                    weights_refreshed = refresh_weight_plan(coordinator)
                    if once:
                        return (
                            0
                            if recovered or cycle_recovered or evaluated or weights_refreshed
                            else 3
                        )
                    if not cycle_recovered and not evaluated and not weights_refreshed:
                        phase["value"] = "idle"
                        await asyncio.sleep(poll_seconds)
            finally:
                phase["value"] = "stopping"
                repository.heartbeat_service(
                    now=datetime.now(timezone.utc),
                    phase="stopping",
                    state="stopping",
                    software_version=SOFTWARE_VERSION,
                )
                heartbeat_task.cancel()
                weight_refresh_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task
                with contextlib.suppress(asyncio.CancelledError):
                    await weight_refresh_task
                repository.release_lock()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PostgreSQL-backed evaluator scheduler"
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    return asyncio.run(run(once=args.once))


if __name__ == "__main__":
    raise SystemExit(main())
