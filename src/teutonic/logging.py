"""Structured logging configuration for teutonic.

Call ``setup_logging()`` once at process startup.  All modules that use
``structlog.get_logger()`` or stdlib ``logging.getLogger()`` will produce
structured output (JSON in production, colored console in dev).

Context variables (role, uid, window) are automatically injected into every
log record when bound via ``structlog.contextvars.bind_contextvars()``.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

import structlog


def setup_logging(
    *,
    level: str | int = "INFO",
    json_output: bool = False,
    log_file: str | Path | None = None,
    log_file_max_bytes: int = 50 * 1024 * 1024,
    log_file_backup_count: int = 5,
) -> None:
    """Configure structlog + stdlib logging for the entire process.

    Parameters
    ----------
    level:
        Root log level (e.g. ``"DEBUG"``, ``"INFO"``).
    json_output:
        If *True*, render as JSON lines (production).
        If *False*, use colored console output (dev).
    log_file:
        Optional path to a log file.  Uses ``RotatingFileHandler``.
    log_file_max_bytes:
        Max size per log file before rotation (default 50 MB).
    log_file_backup_count:
        Number of rotated backups to keep (default 5).
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Clear any existing handlers to avoid duplicate output
    root.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        fh = logging.handlers.RotatingFileHandler(
            str(log_file),
            maxBytes=log_file_max_bytes,
            backupCount=log_file_backup_count,
        )
        json_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
            foreign_pre_chain=shared_processors,
        )
        fh.setFormatter(json_formatter)
        root.addHandler(fh)

    # Quiet noisy third-party loggers
    for name in ("botocore", "aiobotocore", "urllib3", "s3transfer"):
        logging.getLogger(name).setLevel(logging.WARNING)
