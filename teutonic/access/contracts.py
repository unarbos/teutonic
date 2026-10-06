from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Mapping

from teutonic.storage.artifacts import ArtifactIntegrityError, model_digest_from_inventory


_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_READY_SIGNAL = re.compile(
    r"^r2ready:v1\|(?P<registration_id>[0-9a-f]{64})\|(?P<manifest_sha256>[0-9a-f]{64})$"
)
_COMPACT_READY_SIGNAL = re.compile(r"^r2ready:v1:(?P<identity>[A-Za-z0-9_-]{86})(?::(?P<competition>main|math|code|text))?$")


def ready_signal_payload(registration_id: str, manifest_sha256: str, competition: str = "main") -> str:
    """Encode two SHA-256 identities into Bittensor-sized commitment text."""
    if not _HEX_DIGEST.fullmatch(registration_id):
        raise ValueError("ready signal registration ID must be a lowercase SHA-256 digest")
    if not _HEX_DIGEST.fullmatch(manifest_sha256):
        raise ValueError("ready signal manifest SHA-256 must be a lowercase digest")
    if competition not in {"main", "math", "code", "text"}:
        raise ValueError("unknown competition")
    identity = bytes.fromhex(registration_id) + bytes.fromhex(manifest_sha256)
    encoded = base64.urlsafe_b64encode(identity).rstrip(b"=").decode("ascii")
    suffix = "" if competition == "main" else f":{competition}"
    return f"r2ready:v1:{encoded}{suffix}"


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True, order=True)
class UidAssignment:
    uid: int
    hotkey: str | None
    coldkey: str | None
    registration_block: int | None = None

    def __post_init__(self) -> None:
        if self.uid < 0:
            raise ValueError("UID must be non-negative")
        if (self.hotkey is None) != (self.coldkey is None):
            raise ValueError("hotkey and coldkey must both be present or absent")
        if self.hotkey is None and self.registration_block is not None:
            raise ValueError("an empty UID cannot have a registration block")
        if self.registration_block is not None and self.registration_block < 0:
            raise ValueError("registration block must be non-negative")


@dataclass(frozen=True, slots=True)
class MetagraphSnapshot:
    netuid: int
    chain_generation: str
    finalized_block: int
    finalized_block_hash: str
    assignments: tuple[UidAssignment, ...]
    observed_at: datetime
    complete: bool = True

    def __post_init__(self) -> None:
        if self.netuid < 0 or self.finalized_block < 0:
            raise ValueError("netuid and finalized block must be non-negative")
        if not self.chain_generation or not self.finalized_block_hash:
            raise ValueError("chain generation and finalized block hash are required")
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        uids = [assignment.uid for assignment in self.assignments]
        if len(uids) != len(set(uids)):
            raise ValueError("metagraph snapshot contains duplicate UIDs")

    @property
    def checksum(self) -> str:
        payload = {
            "assignments": [
                {
                    "coldkey": assignment.coldkey,
                    "hotkey": assignment.hotkey,
                    "registration_block": assignment.registration_block,
                    "uid": assignment.uid,
                }
                for assignment in sorted(self.assignments)
            ],
            "chain_generation": self.chain_generation,
            "finalized_block": self.finalized_block,
            "finalized_block_hash": self.finalized_block_hash,
            "netuid": self.netuid,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(b"teutonic-metagraph-v1\0" + canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class ReadySignal:
    registration_id: str
    manifest_sha256: str
    signalling_hotkey: str
    block_number: int
    extrinsic_index: int
    event_index: int
    raw_payload: str
    competition: str = "main"

    @classmethod
    def parse(
        cls,
        payload: str,
        *,
        signalling_hotkey: str,
        block_number: int,
        extrinsic_index: int,
        event_index: int,
    ) -> ReadySignal:
        compact = _COMPACT_READY_SIGNAL.fullmatch(payload)
        legacy = _LEGACY_READY_SIGNAL.fullmatch(payload)
        if compact is not None:
            try:
                identity = base64.b64decode(
                    compact.group("identity") + "==", altchars=b"-_", validate=True
                )
            except ValueError as exc:
                raise ValueError("ready signal contains invalid base64url") from exc
            if len(identity) != 64:
                raise ValueError("ready signal identity must contain two SHA-256 digests")
            registration_id = identity[:32].hex()
            manifest_sha256 = identity[32:].hex()
        elif legacy is not None:
            registration_id = legacy.group("registration_id")
            manifest_sha256 = legacy.group("manifest_sha256")
        else:
            raise ValueError("ready signal is not a valid r2ready:v1 commitment")
        if min(block_number, extrinsic_index, event_index) < 0:
            raise ValueError("ready signal chain position must be non-negative")
        if not signalling_hotkey:
            raise ValueError("ready signal signalling hotkey is required")
        return cls(
            registration_id=registration_id,
            manifest_sha256=manifest_sha256,
            signalling_hotkey=signalling_hotkey,
            block_number=block_number,
            extrinsic_index=extrinsic_index,
            event_index=event_index,
            raw_payload=payload,
            competition=(compact.group("competition") or "main") if compact else "main",
        )


@dataclass(frozen=True, slots=True, order=True)
class ManifestFile:
    path: str
    size: int
    sha256: str

    @classmethod
    def from_mapping(cls, value: object) -> ManifestFile:
        if not isinstance(value, Mapping) or set(value) != {"path", "size", "sha256"}:
            raise ValueError("manifest file entries require exactly path, size, and sha256")
        path = value["path"]
        size = value["size"]
        sha256 = value["sha256"]
        if not isinstance(path, str) or not path or path == "manifest.json":
            raise ValueError("manifest file path is invalid")
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != path:
            raise ValueError(f"unsafe manifest path: {path!r}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"manifest size is invalid for {path!r}")
        if not isinstance(sha256, str) or not _HEX_DIGEST.fullmatch(sha256):
            raise ValueError(f"manifest SHA-256 is invalid for {path!r}")
        return cls(path=path, size=size, sha256=sha256)


@dataclass(frozen=True, slots=True)
class Manifest:
    registration_id: str
    hotkey: str
    model_name: str
    files: tuple[ManifestFile, ...]
    model_digest: str
    signature: str
    protocol_version: int = 1
    signature_scheme: str = "ed25519"

    @classmethod
    def from_bytes(cls, raw: bytes) -> Manifest:
        try:
            value = json.loads(raw)
        except Exception as exc:
            raise ValueError("manifest is not valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("manifest must be an object")
        required = {
            "protocol_version",
            "signature_scheme",
            "registration_id",
            "hotkey",
            "model_name",
            "files",
            "model_digest",
            "signature",
        }
        if set(value) != required:
            raise ValueError(f"manifest fields differ from v1 contract: {sorted(set(value) - required)}")
        if value["protocol_version"] != 1 or value["signature_scheme"] != "ed25519":
            raise ValueError("manifest requires protocol v1 with Ed25519")
        if not isinstance(value["registration_id"], str) or not _HEX_DIGEST.fullmatch(
            value["registration_id"]
        ):
            raise ValueError("manifest registration_id is invalid")
        if not isinstance(value["hotkey"], str) or not value["hotkey"]:
            raise ValueError("manifest hotkey is required")
        if not isinstance(value["model_name"], str) or not value["model_name"].strip():
            raise ValueError("manifest model_name is required")
        if not isinstance(value["files"], list):
            raise ValueError("manifest files must be an array")
        files = tuple(ManifestFile.from_mapping(item) for item in value["files"])
        if not files or len({item.path for item in files}) != len(files):
            raise ValueError("manifest must contain a non-empty unique file inventory")
        if not isinstance(value["model_digest"], str) or not _HEX_DIGEST.fullmatch(
            value["model_digest"]
        ):
            raise ValueError("manifest model_digest is invalid")
        try:
            observed = model_digest_from_inventory(
                [(item.path, item.size, item.sha256) for item in files]
            )
        except ArtifactIntegrityError as exc:
            raise ValueError(str(exc)) from exc
        if observed != value["model_digest"]:
            raise ValueError("manifest model_digest does not match its file inventory")
        if not isinstance(value["signature"], str) or not value["signature"]:
            raise ValueError("manifest signature is required")
        return cls(
            registration_id=value["registration_id"],
            hotkey=value["hotkey"],
            model_name=value["model_name"].strip(),
            files=files,
            model_digest=value["model_digest"],
            signature=value["signature"],
        )

    def signing_payload(self) -> bytes:
        value: dict[str, Any] = {
            "files": [
                {"path": item.path, "sha256": item.sha256, "size": item.size}
                for item in self.files
            ],
            "hotkey": self.hotkey,
            "model_digest": self.model_digest,
            "model_name": self.model_name,
            "protocol_version": self.protocol_version,
            "registration_id": self.registration_id,
            "signature_scheme": self.signature_scheme,
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    def as_dict(self) -> dict[str, Any]:
        return {**json.loads(self.signing_payload()), "signature": self.signature}

    def as_bytes(self) -> bytes:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.as_bytes()).hexdigest()
