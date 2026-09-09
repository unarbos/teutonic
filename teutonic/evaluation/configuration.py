from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class EvaluationConfigurationError(RuntimeError):
    pass


def canonical_manifest_bytes(manifest: object) -> bytes:
    return json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def validate_dataset_manifest(manifest: object) -> Mapping[str, Any]:
    if isinstance(manifest, Mapping):
        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise EvaluationConfigurationError("dataset manifest must contain non-empty shards")
        value = dict(manifest)
    else:
        raise EvaluationConfigurationError("dataset manifest must be an object")

    for index, shard in enumerate(shards):
        if isinstance(shard, Mapping):
            reference = next(
                (
                    str(shard[key]).strip()
                    for key in ("url", "href", "uri", "key", "path", "name")
                    if shard.get(key)
                ),
                "",
            )
            digest = shard.get("sha256")
            if (
                not isinstance(digest, str) or not _DIGEST.fullmatch(digest.lower())
            ):
                raise EvaluationConfigurationError(
                    f"dataset manifest shard {index} has an invalid SHA-256"
                )
            for field in ("n_tokens", "size_bytes"):
                number = shard.get(field)
                if (
                    isinstance(number, bool) or not isinstance(number, int) or number <= 0
                ):
                    raise EvaluationConfigurationError(
                        f"dataset manifest shard {index} has invalid {field}"
                    )
        else:
            raise EvaluationConfigurationError(f"dataset manifest shard {index} must be an object")
        if not reference.endswith(".npy"):
            raise EvaluationConfigurationError(
                f"dataset manifest shard {index} does not reference an .npy file"
            )
    return value


@dataclass(frozen=True, slots=True)
class DatasetManifestSnapshot:
    name: str
    manifest_url: str
    manifest_sha256: str
    proportion: float
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError("dataset manifest name is invalid")
        parsed = urlparse(self.manifest_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("dataset manifest URL must be public HTTPS")
        if not _DIGEST.fullmatch(self.manifest_sha256):
            raise ValueError("dataset manifest SHA-256 is invalid")
        if not math.isfinite(self.proportion) or not 0 < self.proportion <= 1:
            raise ValueError("dataset sample proportion must be in (0, 1]")
        validated = validate_dataset_manifest(self.manifest)
        observed = hashlib.sha256(canonical_manifest_bytes(validated)).hexdigest()
        if observed != self.manifest_sha256:
            raise ValueError("dataset manifest content does not match its SHA-256")

    def request_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "manifest_url": self.manifest_url,
            "manifest_sha256": self.manifest_sha256,
            "proportion": self.proportion,
            "manifest": self.manifest,
        }


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    config_version: str
    dataset_label: str
    n: int
    delta_threshold: float
    manifests: tuple[DatasetManifestSnapshot, ...]
    shards_per_dataset: int

    def __post_init__(self) -> None:
        if not _DIGEST.fullmatch(self.config_version):
            raise ValueError("evaluation config version must be a SHA-256 digest")
        if not self.dataset_label:
            raise ValueError("dataset label is required")
        if self.n < 1:
            raise ValueError("evaluation sample count must be positive")
        if not math.isfinite(self.delta_threshold):
            raise ValueError("evaluation delta threshold must be finite")
        if not self.manifests:
            raise ValueError("evaluation configuration needs dataset manifests")
        if self.shards_per_dataset < 1:
            raise ValueError("evaluation shards_per_dataset must be positive")
        total = sum(item.proportion for item in self.manifests)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("dataset sample proportions must sum to 1")


def _source_targets(total: int, proportions: Sequence[float]) -> list[int]:
    raw = [total * value for value in proportions]
    targets = [int(value) for value in raw]
    remainder = total - sum(targets)
    order = sorted(range(len(raw)), key=lambda index: (-(raw[index] - targets[index]), index))
    for index in order[:remainder]:
        targets[index] += 1
    return targets


def _dataset_seed(*, block_hash: str, hotkey: str) -> int:
    material = f"block_hash={block_hash}|hotkey={hotkey}"
    digest = hashlib.blake2b(material.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _shard_url(manifest_url: str, reference: str) -> str:
    parsed = urlparse(reference)
    if parsed.scheme:
        if parsed.scheme != "https" or not parsed.netloc:
            raise EvaluationConfigurationError("dataset shard URL must be public HTTPS")
        return reference
    manifest = urlparse(manifest_url)
    key = reference.lstrip("/")
    parent = manifest.path.rsplit("/", 1)[0].strip("/")
    if parent and key.startswith(f"{parent}/"):
        return f"{manifest.scheme}://{manifest.netloc}/{key}"
    return urljoin(manifest_url, key)


def pretokenized_dataset_request(
    settings: EvaluationSettings,
    *,
    block_hash: str,
    hotkey: str,
    seq_len: int,
) -> dict[str, Any]:
    if seq_len < 2:
        raise ValueError("evaluation sequence length must be at least 2")
    targets = _source_targets(
        settings.n, [item.proportion for item in settings.manifests]
    )
    seed = _dataset_seed(block_hash=block_hash, hotkey=hotkey)
    sources: list[dict[str, Any]] = []
    for target, snapshot in zip(targets, settings.manifests, strict=True):
        shards = list(snapshot.manifest["shards"])
        source_digest = hashlib.blake2b(
            f"{seed}:{snapshot.name}".encode("utf-8"), digest_size=8
        ).digest()
        random.Random(int.from_bytes(source_digest, "little")).shuffle(shards)
        shards = shards[: settings.shards_per_dataset]
        selected: list[dict[str, Any]] = []
        available_sequences = 0
        required_sequences = target + max(16, math.ceil(target * 0.5))
        for shard in shards:
            reference = next(
                str(shard[key]).strip()
                for key in ("url", "href", "uri", "key", "path", "name")
                if shard.get(key)
            )
            n_tokens = int(shard["n_tokens"])
            selected.append(
                {
                    "url": _shard_url(snapshot.manifest_url, reference),
                    "sha256": str(shard["sha256"]).lower(),
                    "size_bytes": int(shard["size_bytes"]),
                    "n_tokens": n_tokens,
                }
            )
            available_sequences += n_tokens // seq_len
            if available_sequences >= required_sequences:
                break
        if available_sequences < target:
            raise EvaluationConfigurationError(
                f"dataset {snapshot.name!r} cannot provide {target} sequences of length {seq_len}"
            )
        sources.append(
            {
                "name": snapshot.name,
                "proportion": snapshot.proportion,
                "target_sequences": target,
                "shards": selected,
            }
        )
    return {
        "source": "pretokenized_npy",
        "label": settings.dataset_label,
        "sources": sources,
    }


def fetch_dataset_manifest(
    *,
    name: str,
    manifest_url: str,
    proportion: float,
    opener: Callable[..., Any] = urlopen,
) -> DatasetManifestSnapshot:
    parsed = urlparse(manifest_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise EvaluationConfigurationError("dataset manifest URL must be public HTTPS")
    request = Request(manifest_url, headers={"User-Agent": "teutonic-config/1.0"})
    try:
        with opener(request, timeout=120) as response:
            raw = response.read()
        manifest = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise EvaluationConfigurationError(
            f"failed to fetch dataset manifest {manifest_url}"
        ) from exc
    validated = validate_dataset_manifest(manifest)
    digest = hashlib.sha256(canonical_manifest_bytes(validated)).hexdigest()
    return DatasetManifestSnapshot(
        name=name,
        manifest_url=manifest_url,
        manifest_sha256=digest,
        proportion=float(proportion),
        manifest=validated,
    )


def evaluation_config_version(
    *,
    dataset_label: str,
    n: int,
    delta_threshold: float,
    manifests: Sequence[DatasetManifestSnapshot],
    shards_per_dataset: int,
) -> str:
    value = {
        "dataset_label": dataset_label,
        "delta_threshold": float(delta_threshold),
        "manifests": [
            {
                "manifest_sha256": item.manifest_sha256,
                "manifest_url": item.manifest_url,
                "name": item.name,
                "proportion": item.proportion,
            }
            for item in manifests
        ],
        "n": int(n),
        "shards_per_dataset": int(shards_per_dataset),
        "protocol_version": 2,
    }
    return hashlib.sha256(canonical_manifest_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class StoredEvaluationConfiguration:
    evaluation_config_id: str
    config_version: str
    created: bool


def store_evaluation_configuration(
    connection: Any,
    *,
    netuid: int,
    chain_generation: str,
    competition: str,
    dataset_label: str,
    n: int,
    delta_threshold: float,
    manifests: Sequence[DatasetManifestSnapshot],
    shards_per_dataset: int,
) -> StoredEvaluationConfiguration:
    snapshots = tuple(manifests)
    config_version = evaluation_config_version(
        dataset_label=dataset_label,
        n=n,
        delta_threshold=delta_threshold,
        manifests=snapshots,
        shards_per_dataset=shards_per_dataset,
    )
    settings = EvaluationSettings(
        config_version=config_version,
        dataset_label=dataset_label,
        n=n,
        delta_threshold=delta_threshold,
        manifests=snapshots,
        shards_per_dataset=shards_per_dataset,
    )
    with connection.transaction():
        row = connection.execute(
            """
            SELECT competition_id
              FROM control_plane.competitions
             WHERE netuid = %s AND chain_generation = %s AND name = %s
             FOR UPDATE
            """,
            (netuid, chain_generation, competition),
        ).fetchone()
        if row is None:
            raise EvaluationConfigurationError("competition must exist before evaluation setup")
        competition_id = row["competition_id"]
        existing = connection.execute(
            """
            SELECT evaluation_config_id
              FROM control_plane.evaluation_configs
             WHERE competition_id = %s AND config_version = %s
            """,
            (competition_id, settings.config_version),
        ).fetchone()
        created = existing is None
        if created:
            evaluation_config_id = connection.execute(
                """
                INSERT INTO control_plane.evaluation_configs (
                    competition_id, config_version, dataset_label, eval_n,
                    delta_threshold, shards_per_dataset, active
                ) VALUES (%s, %s, %s, %s, %s, %s, false)
                RETURNING evaluation_config_id
                """,
                (
                    competition_id,
                    settings.config_version,
                    settings.dataset_label,
                    settings.n,
                    settings.delta_threshold,
                    settings.shards_per_dataset,
                ),
            ).fetchone()["evaluation_config_id"]
            for position, snapshot in enumerate(settings.manifests):
                connection.execute(
                    """
                    INSERT INTO control_plane.dataset_manifests (
                        evaluation_config_id, position, name, manifest_url,
                        manifest_sha256, manifest_json, sample_proportion
                    ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                    """,
                    (
                        evaluation_config_id,
                        position,
                        snapshot.name,
                        snapshot.manifest_url,
                        snapshot.manifest_sha256,
                        json.dumps(snapshot.manifest, sort_keys=True, separators=(",", ":")),
                        snapshot.proportion,
                    ),
                )
        else:
            evaluation_config_id = existing["evaluation_config_id"]
        connection.execute(
            """
            UPDATE control_plane.evaluation_configs
               SET active = false
             WHERE competition_id = %s AND active
            """,
            (competition_id,),
        )
        connection.execute(
            """
            UPDATE control_plane.evaluation_configs
               SET active = true
             WHERE evaluation_config_id = %s AND NOT active
            """,
            (evaluation_config_id,),
        )
    return StoredEvaluationConfiguration(
        str(evaluation_config_id), settings.config_version, created
    )
