from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass


PROTOCOL_VERSION = 1
ED25519_SCHEME = "ed25519"
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{40,64}$")
_ACTIVATION = re.compile(r"^r2activate:v1:(?P<signature>[A-Za-z0-9_-]{86})$")


def _require_hash(value: str, field: str) -> str:
    if not _HEX_SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _require_ss58(value: str, field: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{field} is not a canonical SS58-like identifier")
    return value


def _require_chain_generation(value: str) -> str:
    if not value or "|" in value or len(value) > 128:
        raise ValueError("chain_generation must be non-empty and delimiter-safe")
    return value


def registration_id(
    *,
    netuid: int,
    uid: int,
    hotkey: str,
    registration_block: int,
    chain_generation: str,
) -> str:
    """Derive one registration identity entirely from finalized public chain data."""
    if min(netuid, uid, registration_block) < 0:
        raise ValueError("netuid, uid, and registration_block must be non-negative")
    _require_ss58(hotkey, "hotkey")
    _require_chain_generation(chain_generation)
    body = json.dumps(
        {
            "chain_generation": chain_generation,
            "hotkey": hotkey,
            "netuid": netuid,
            "registration_block": registration_block,
            "uid": uid,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(b"teutonic-registration-v2\0" + body).hexdigest()


def activation_message(
    *,
    netuid: int,
    uid: int,
    hotkey: str,
    registration_id: str,
    registration_block: int,
    chain_generation: str,
) -> str:
    """Canonical Ed25519 proof committed by the registered hotkey on-chain."""
    _require_ss58(hotkey, "hotkey")
    _require_hash(registration_id, "registration_id")
    _require_chain_generation(chain_generation)
    if min(netuid, uid, registration_block) < 0:
        raise ValueError("netuid, uid, and registration_block must be non-negative")
    return "|".join(
        (
            "activate",
            "v2",
            chain_generation,
            str(netuid),
            str(uid),
            hotkey,
            registration_id,
            str(registration_block),
        )
    )


def activation_signal_payload(signature: bytes) -> str:
    if len(signature) != 64:
        raise ValueError("Ed25519 signature must contain 64 bytes")
    encoded = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
    return f"r2activate:v1:{encoded}"


@dataclass(frozen=True, slots=True)
class ActivationSignal:
    signature: str
    netuid: int
    chain_generation: str
    signalling_hotkey: str
    block_number: int
    extrinsic_index: int
    event_index: int
    raw_payload: str

    @classmethod
    def parse(
        cls,
        payload: str,
        *,
        netuid: int,
        chain_generation: str,
        signalling_hotkey: str,
        block_number: int,
        extrinsic_index: int,
        event_index: int,
    ) -> "ActivationSignal":
        match = _ACTIVATION.fullmatch(payload)
        if match is None:
            raise ValueError("activation signal is not a valid r2activate:v1 commitment")
        try:
            signature = base64.b64decode(
                match.group("signature") + "==", altchars=b"-_", validate=True
            )
        except ValueError as exc:
            raise ValueError("activation signal contains invalid base64url") from exc
        if len(signature) != 64:
            raise ValueError("activation signal must contain an Ed25519 signature")
        if netuid < 0:
            raise ValueError("activation signal netuid must be non-negative")
        _require_chain_generation(chain_generation)
        _require_ss58(signalling_hotkey, "signalling_hotkey")
        if min(block_number, extrinsic_index, event_index) < 0:
            raise ValueError("activation signal chain position must be non-negative")
        return cls(
            signature=base64.b64encode(signature).decode("ascii"),
            netuid=netuid,
            chain_generation=chain_generation,
            signalling_hotkey=signalling_hotkey,
            block_number=block_number,
            extrinsic_index=extrinsic_index,
            event_index=event_index,
            raw_payload=payload,
        )


def mailbox_object_key(registration_id: str, generation: int) -> str:
    """Return the immutable public mailbox key for one credential generation."""
    _require_hash(registration_id, "registration_id")
    if generation < 1:
        raise ValueError("credential generation must start at one")
    return f"mailbox/v1/{registration_id}/generations/{generation:020d}.bin"


def latest_mailbox_object_key(registration_id: str) -> str:
    """Return the mutable alias containing the latest encrypted generation."""
    _require_hash(registration_id, "registration_id")
    return f"mailbox/v1/{registration_id}/latest.bin"
