from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker

from teutonic.schemas import SCHEMA_DIR

SCHEMA_PATH = SCHEMA_DIR / "dashboard-v1.schema.json"


class DashboardContractError(ValueError):
    pass


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _reject_non_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise DashboardContractError(f"non-finite number at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")


def validate_dashboard(payload: Mapping[str, Any]) -> None:
    _reject_non_finite(payload)
    errors = sorted(_validator().iter_errors(payload), key=lambda item: list(item.absolute_path))
    if errors:
        error = errors[0]
        location = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.absolute_path
        )
        raise DashboardContractError(f"dashboard-v1 invalid at {location}: {error.message}")


def canonical_dashboard_json(payload: Mapping[str, Any]) -> bytes:
    validate_dashboard(payload)
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return text.encode("utf-8", errors="strict")
    except (UnicodeError, ValueError) as exc:
        raise DashboardContractError("dashboard cannot be serialized as strict UTF-8 JSON") from exc


def canonical_dataset_manifest_json(payload: Mapping[str, Any]) -> bytes:
    _reject_non_finite(payload)
    required = {
        "schema_version",
        "generated_at",
        "chain",
        "config_version",
        "dataset_label",
        "eval_n",
        "delta_threshold",
        "sampling",
        "sources",
    }
    if set(payload) != required:
        raise DashboardContractError("global dataset manifest has invalid top-level fields")
    if payload.get("schema_version") != 1:
        raise DashboardContractError("global dataset manifest schema_version must be 1")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise DashboardContractError("global dataset manifest needs at least one source")
    source_fields = {
        "name",
        "proportion",
        "manifest_url",
        "manifest_sha256",
        "source_repo",
        "tokenizer",
        "dtype",
        "tokenization_mode",
        "sequence_length",
        "total_tokens",
        "total_shards",
        "estimated_sequences",
    }
    for source in sources:
        if not isinstance(source, Mapping) or set(source) not in (source_fields, source_fields | {"dataset"}):
            raise DashboardContractError(
                "global dataset manifest source has invalid fields"
            )
        if "dataset" in source and (not isinstance(source["dataset"], str) or not source["dataset"]):
            raise DashboardContractError("global dataset manifest source needs a dataset name")
        if not isinstance(source["name"], str) or not source["name"]:
            raise DashboardContractError("global dataset manifest source needs a name")
        proportion = source["proportion"]
        if not isinstance(proportion, (int, float)) or isinstance(proportion, bool):
            raise DashboardContractError(
                "global dataset manifest source proportion must be numeric"
            )
        if not 0 < float(proportion) <= 1:
            raise DashboardContractError(
                "global dataset manifest source proportion must be in (0, 1]"
            )
        if not isinstance(source["manifest_url"], str) or not source[
            "manifest_url"
        ].startswith("https://"):
            raise DashboardContractError(
                "global dataset manifest source URL must use HTTPS"
            )
        digest = source["manifest_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise DashboardContractError(
                "global dataset manifest source digest must be lowercase SHA-256"
            )
        for field in ("source_repo", "tokenizer", "dtype", "tokenization_mode"):
            value = source[field]
            if value is not None and (not isinstance(value, str) or not value):
                raise DashboardContractError(
                    f"global dataset manifest source {field} must be text or null"
                )
        for field in ("sequence_length", "estimated_sequences"):
            value = source[field]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise DashboardContractError(
                    f"global dataset manifest source {field} must be positive or null"
                )
        for field in ("total_tokens", "total_shards"):
            value = source[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise DashboardContractError(
                    f"global dataset manifest source {field} must be positive"
                )
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    except (UnicodeError, ValueError) as exc:
        raise DashboardContractError(
            "global dataset manifest cannot be serialized as strict UTF-8 JSON"
        ) from exc
