"""Optional metrics reporting for experiment tracking (e.g. wandb).

Provides a ``MetricsReporter`` protocol and two implementations:

- ``NullReporter``: no-op, used when experiment tracking is disabled.
- ``WandbReporter``: forwards numeric metrics to Weights & Biases.

Use ``create_reporter()`` to build the right one based on CLI flags.
"""

from __future__ import annotations

import dataclasses
import structlog
from typing import Any, Protocol, runtime_checkable

logger = structlog.get_logger(__name__)


@runtime_checkable
class MetricsReporter(Protocol):
    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None: ...
    def close(self) -> None: ...


class NullReporter:
    """No-op reporter for when experiment tracking is disabled."""

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        pass

    def close(self) -> None:
        pass


class WandbReporter:
    """Forwards metrics to Weights & Biases."""

    def __init__(
        self,
        *,
        project: str = "teutonic",
        entity: str | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        job_type: str | None = None,
    ):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is not installed. Install it with: "
                "uv pip install wandb  (or add teutonic[wandb])"
            )

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            tags=tags,
            group=group,
            job_type=job_type,
            settings=wandb.Settings(
                init_timeout=120,
                _disable_stats=True,
            ),
        )
        logger.info(
            "metrics.wandb.init",
            project=project,
            entity=entity,
            run_name=self._run.name,
            run_id=self._run.id,
            run_url=self._run.url,
        )

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        self._run.log(metrics, step=step)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
            logger.info("metrics.wandb.closed")
            self._run = None


def create_reporter(
    *,
    use_wandb: bool = False,
    role: str = "unknown",
    uid: int = 0,
    hparams: Any = None,
    project: str = "teutonic",
    entity: str | None = None,
) -> MetricsReporter:
    """Factory: build a reporter based on flags."""
    if not use_wandb:
        return NullReporter()

    config = None
    if hparams is not None:
        config = dataclasses.asdict(hparams) if dataclasses.is_dataclass(hparams) else dict(hparams)
    config = config or {}
    config["role"] = role
    config["uid"] = uid

    return WandbReporter(
        project=project,
        entity=entity,
        name=f"{role[0].upper()}{uid}",
        config=config,
        tags=[role],
        group=role,
        job_type=role,
    )
