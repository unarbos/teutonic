"""Behavioral-copy probes and a bounded private fingerprint bank.

The public deterministic probe is retained for experiments and regression
tests.  The verdict path uses a secret-derived, fixed probe and stores each
model's float16 logits in a local SQLite bank.  A newly loaded model can then be
compared with recent queue entries without loading their checkpoints again.

This is an identity/copy gate, not a quality metric.  A very small divergence
means that two checkpoints implement effectively the same function on the
probe.  Larger divergences do not prove independent training.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import sqlite3
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PUBLIC_COPY_PROBE_VERSION = "copy-probe-v1"
COPY_PROBE_VERSION = "copy-probe-private-v1"
COPY_PROBE_SEED = 0x544555544F4E4943  # ASCII-ish "TEUTONIC", fixed and public.
COPY_PROBE_BATCH = 16
COPY_PROBE_SEQ_LEN = 128
COPY_PROBE_POSITIONS = (31, 63, 95, 127)
COPY_PROBE_N_POSITIONS = len(COPY_PROBE_POSITIONS)
MIN_PRIVATE_SECRET_BYTES = 32


@dataclass(frozen=True)
class CopyProbeSpec:
    version: str = PUBLIC_COPY_PROBE_VERSION
    seed: int = COPY_PROBE_SEED
    batch_size: int = COPY_PROBE_BATCH
    seq_len: int = COPY_PROBE_SEQ_LEN
    positions: tuple[int, ...] = COPY_PROBE_POSITIONS


DEFAULT_SPEC = CopyProbeSpec()


@dataclass(frozen=True)
class PrivateCopyProbeSpec:
    version: str = COPY_PROBE_VERSION
    batch_size: int = COPY_PROBE_BATCH
    seq_len: int = COPY_PROBE_SEQ_LEN
    n_positions: int = COPY_PROBE_N_POSITIONS


DEFAULT_PRIVATE_SPEC = PrivateCopyProbeSpec()


def public_probe_tokens(vocab_size: int, spec: CopyProbeSpec = DEFAULT_SPEC) -> np.ndarray:
    """Return the public constant probe tensor as ``int64[batch, seq]``.

    NumPy PCG64 is named explicitly so the sample definition is portable and
    does not depend on PyTorch/CUDA RNG implementations.
    """
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
    if spec.batch_size < 1 or spec.seq_len < 2:
        raise ValueError("copy probe needs a non-empty batch and seq_len >= 2")
    if not spec.positions or min(spec.positions) < 0 or max(spec.positions) >= spec.seq_len:
        raise ValueError(f"invalid probe positions {spec.positions} for seq_len={spec.seq_len}")
    rng = np.random.Generator(np.random.PCG64(spec.seed))
    return rng.integers(
        0,
        vocab_size,
        size=(spec.batch_size, spec.seq_len),
        dtype=np.int64,
    )


def probe_input_digest(tokens: np.ndarray) -> str:
    canonical = np.asarray(tokens, dtype="<i8", order="C")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _secret_bytes(secret: str | bytes) -> bytes:
    value = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
    if len(value) < MIN_PRIVATE_SECRET_BYTES:
        raise ValueError(
            f"private copy-probe secret must be at least {MIN_PRIVATE_SECRET_BYTES} bytes"
        )
    return value


def load_private_probe_secret(path: str | os.PathLike[str]) -> str:
    """Read a private probe secret from an operator-only local file."""
    secret_path = Path(path)
    if not secret_path.is_file():
        raise RuntimeError(f"private copy-probe secret file is missing: {secret_path}")
    mode = stat.S_IMODE(secret_path.stat().st_mode)
    if mode & 0o077:
        raise RuntimeError(
            f"private copy-probe secret file must not be accessible by group/other: "
            f"{secret_path} mode={mode:o}"
        )
    secret = secret_path.read_text(encoding="utf-8").strip()
    _secret_bytes(secret)
    return secret


def _private_context(vocab_size: int, spec: PrivateCopyProbeSpec) -> bytes:
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
    if spec.batch_size < 1 or spec.seq_len < 2:
        raise ValueError("copy probe needs a non-empty batch and seq_len >= 2")
    if not 1 <= spec.n_positions < spec.seq_len:
        raise ValueError(
            f"n_positions must be between 1 and seq_len-1, got {spec.n_positions}"
        )
    return json.dumps(
        {
            "version": spec.version,
            "vocab_size": int(vocab_size),
            "batch_size": spec.batch_size,
            "seq_len": spec.seq_len,
            "n_positions": spec.n_positions,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _secret_seed(secret: bytes, label: bytes, context: bytes) -> int:
    digest = hmac.new(secret, label + b"\0" + context, hashlib.sha256).digest()
    return int.from_bytes(digest[:16], "little")


def private_probe_plan(
    vocab_size: int,
    secret: str | bytes,
    spec: PrivateCopyProbeSpec = DEFAULT_PRIVATE_SPEC,
) -> tuple[np.ndarray, tuple[int, ...], str]:
    """Derive fixed private tokens, positions, and a non-secret key identifier.

    The secret itself, generated token IDs, positions, and input digest must
    remain inside the evaluator.  ``key_id`` only prevents accidentally mixing
    fingerprints after a secret rotation; it is not used as key material.
    """
    key = _secret_bytes(secret)
    context = _private_context(vocab_size, spec)
    token_rng = np.random.Generator(np.random.PCG64(_secret_seed(key, b"tokens", context)))
    position_rng = np.random.Generator(
        np.random.PCG64(_secret_seed(key, b"positions", context))
    )
    tokens = token_rng.integers(
        0,
        vocab_size,
        size=(spec.batch_size, spec.seq_len),
        dtype=np.int64,
    )
    positions = tuple(
        sorted(
            int(value)
            for value in position_rng.choice(
                np.arange(1, spec.seq_len),
                size=spec.n_positions,
                replace=False,
            )
        )
    )
    key_id = hmac.new(key, b"copy-probe-key-id\0" + context, hashlib.sha256).hexdigest()[:16]
    return tokens, positions, key_id


def _model_input_device(model) -> torch.device:
    embed = getattr(getattr(model, "model", None), "embed_tokens", None)
    if embed is not None:
        try:
            return next(embed.parameters()).device
        except StopIteration:
            pass
    return next(model.parameters()).device


def _lm_head_device(model) -> torch.device:
    return next(model.lm_head.parameters()).device


@torch.no_grad()
def probe_logits(
    model,
    tokens: np.ndarray,
    positions: tuple[int, ...] = COPY_PROBE_POSITIONS,
) -> torch.Tensor:
    """Return float32 CPU logits shaped ``[batch * positions, vocab]``."""
    input_ids = torch.as_tensor(tokens, dtype=torch.long, device=_model_input_device(model))
    if hasattr(model, "reset_state"):
        model.reset_state()
    try:
        hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
        selected = hidden[:, list(positions), :]
        head_device = _lm_head_device(model)
        if selected.device != head_device:
            selected = selected.to(head_device)
        logits = model.lm_head(selected)
        return logits.reshape(-1, logits.shape[-1]).float().cpu()
    finally:
        # Stateful architectures must not carry the public probe into the real
        # evaluation or into the second model comparison.
        if hasattr(model, "reset_state"):
            model.reset_state()


def _summary(values: torch.Tensor, prefix: str) -> dict[str, float]:
    values = values.detach().float().cpu()
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_p50": float(torch.quantile(values, 0.50)),
        f"{prefix}_p95": float(torch.quantile(values, 0.95)),
        f"{prefix}_max": float(values.max()),
    }


@torch.no_grad()
def distribution_metrics(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    row_chunk: int = 8,
) -> dict[str, float]:
    """Compare aligned full-vocabulary logits without retaining probabilities.

    Computation is chunked by probe row to cap temporary CPU memory.  JS is
    bounded by ln(2); symmetric KL is included as a more sensitive diagnostic.
    """
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError(
            f"copy-probe logit shape mismatch: {tuple(reference_logits.shape)} != "
            f"{tuple(candidate_logits.shape)}"
        )
    if reference_logits.ndim != 2 or reference_logits.shape[0] < 1:
        raise ValueError(
            f"copy-probe logits must be non-empty 2D tensors, got {reference_logits.shape}"
        )

    js_rows: list[torch.Tensor] = []
    skl_rows: list[torch.Tensor] = []
    tv_rows: list[torch.Tensor] = []
    top1_matches = 0
    rows = reference_logits.shape[0]
    chunk_size = max(1, int(row_chunk))

    for start in range(0, rows, chunk_size):
        # float64 matters near the copy boundary. In float32 the standard JS
        # expression can report ~1e-7 for two exactly equal distributions due
        # solely to cancellation, which is large enough to pollute calibration.
        ref = reference_logits[start : start + chunk_size].double()
        cand = candidate_logits[start : start + chunk_size].double()
        log_p = F.log_softmax(ref, dim=-1)
        log_q = F.log_softmax(cand, dim=-1)
        p = log_p.exp()
        q = log_q.exp()
        log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)

        js = 0.5 * (
            (p * (log_p - log_m)).sum(dim=-1)
            + (q * (log_q - log_m)).sum(dim=-1)
        )
        symmetric_kl = 0.5 * (
            (p * (log_p - log_q)).sum(dim=-1)
            + (q * (log_q - log_p)).sum(dim=-1)
        )
        tv = 0.5 * (p - q).abs().sum(dim=-1)

        # Tiny negative values can arise from floating-point cancellation even though
        # both divergences are non-negative mathematically.
        js_rows.append(js.clamp_min_(0).cpu())
        skl_rows.append(symmetric_kl.clamp_min_(0).cpu())
        tv_rows.append(tv.cpu())
        top1_matches += int((ref.argmax(dim=-1) == cand.argmax(dim=-1)).sum())

    js_all = torch.cat(js_rows)
    skl_all = torch.cat(skl_rows)
    tv_all = torch.cat(tv_rows)
    metrics = {
        **_summary(js_all, "js"),
        **_summary(skl_all, "symmetric_kl"),
        **_summary(tv_all, "total_variation"),
        "top1_agreement": top1_matches / rows,
        "n_distributions": rows,
        "vocab_size": int(reference_logits.shape[1]),
    }
    metrics["js_similarity"] = 1.0 - metrics["js_mean"] / math.log(2.0)
    return metrics


@torch.no_grad()
def compare_models(reference_model, candidate_model, spec: CopyProbeSpec = DEFAULT_SPEC) -> dict:
    ref_vocab = int(getattr(getattr(reference_model, "config", None), "vocab_size", 0) or 0)
    cand_vocab = int(getattr(getattr(candidate_model, "config", None), "vocab_size", 0) or 0)
    if not ref_vocab or ref_vocab != cand_vocab:
        raise ValueError(
            f"copy-probe vocab mismatch: reference={ref_vocab}, candidate={cand_vocab}"
        )

    tokens = public_probe_tokens(ref_vocab, spec)
    ref_logits = probe_logits(reference_model, tokens, spec.positions)
    candidate_logits = probe_logits(candidate_model, tokens, spec.positions)
    metrics: dict[str, Any] = distribution_metrics(ref_logits, candidate_logits)
    metrics.update({
        "version": spec.version,
        "seed": spec.seed,
        "batch_size": spec.batch_size,
        "seq_len": spec.seq_len,
        "positions": list(spec.positions),
        "input_sha256": probe_input_digest(tokens),
    })
    return metrics


def model_compatibility_key(model) -> str:
    """Return the fields that must agree before fingerprints are comparable."""
    config = getattr(model, "config", None)
    vocab_size = int(getattr(config, "vocab_size", 0) or 0)
    if vocab_size < 2:
        raise ValueError(f"model has invalid vocab_size={vocab_size}")
    architectures = getattr(config, "architectures", None) or []
    if isinstance(architectures, str):
        architectures = [architectures]
    payload = {
        "model_type": str(getattr(config, "model_type", "") or type(model).__name__),
        "architectures": sorted(str(value) for value in architectures),
        "vocab_size": vocab_size,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@torch.no_grad()
def private_model_fingerprint(
    model,
    secret: str | bytes,
    spec: PrivateCopyProbeSpec = DEFAULT_PRIVATE_SPEC,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Measure one loaded model once and return a persistable float16 fingerprint."""
    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
    tokens, positions, key_id = private_probe_plan(vocab_size, secret, spec)
    logits = probe_logits(model, tokens, positions).to(dtype=torch.float16).contiguous()
    return logits, {
        "version": spec.version,
        "key_id": key_id,
        "compatibility_key": model_compatibility_key(model),
        "vocab_size": vocab_size,
        "n_distributions": int(logits.shape[0]),
    }


def _fingerprint_bytes(fingerprint: torch.Tensor) -> tuple[bytes, int, int, str]:
    if fingerprint.ndim != 2 or fingerprint.shape[0] < 1 or fingerprint.shape[1] < 2:
        raise ValueError(f"fingerprint must be a non-empty 2D tensor, got {fingerprint.shape}")
    array = np.asarray(fingerprint.detach().cpu().numpy(), dtype="<f2", order="C")
    payload = array.tobytes(order="C")
    return payload, int(array.shape[0]), int(array.shape[1]), hashlib.sha256(payload).hexdigest()


def _fingerprint_tensor(payload: bytes, rows: int, vocab_size: int, digest: str) -> torch.Tensor:
    expected_bytes = rows * vocab_size * np.dtype("<f2").itemsize
    if rows < 1 or vocab_size < 2 or len(payload) != expected_bytes:
        raise RuntimeError(
            f"corrupt copy fingerprint: shape=({rows}, {vocab_size}) bytes={len(payload)}"
        )
    actual_digest = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual_digest, digest):
        raise RuntimeError("corrupt copy fingerprint: SHA-256 mismatch")
    # Copy makes the NumPy storage writable and releases the SQLite blob after
    # this comparison instead of retaining a view of it.
    array = np.frombuffer(payload, dtype="<f2").reshape(rows, vocab_size).copy()
    return torch.from_numpy(array)


class FingerprintBank:
    """SQLite-backed rolling bank of recent private model fingerprints."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        max_entries: int = 100,
        max_similar_models: int = 3,
    ):
        self.path = Path(path)
        self.max_entries = int(max_entries)
        self.max_similar_models = int(max_similar_models)
        if self.max_entries < 1:
            raise ValueError("fingerprint bank max_entries must be >= 1")
        if self.max_similar_models < 1:
            raise ValueError("fingerprint bank max_similar_models must be >= 1")
        if self.max_similar_models > self.max_entries:
            raise ValueError(
                "fingerprint bank max_similar_models must not exceed max_entries"
            )

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except PermissionError:
            pass
        connection = sqlite3.connect(self.path, timeout=60.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        self._create_schema(connection)
        try:
            os.chmod(self.path, 0o600)
        except PermissionError:
            pass
        return connection

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                model_ref TEXT NOT NULL,
                model_digest TEXT NOT NULL,
                submission_id TEXT NOT NULL,
                submission_order TEXT NOT NULL,
                hotkey TEXT NOT NULL,
                version TEXT NOT NULL,
                key_id TEXT NOT NULL,
                compatibility_key TEXT NOT NULL,
                similarity_group TEXT NOT NULL,
                rows INTEGER NOT NULL,
                vocab_size INTEGER NOT NULL,
                logits_sha256 TEXT NOT NULL,
                logits BLOB NOT NULL,
                UNIQUE(version, key_id, model_digest)
            )
            """
        )
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(fingerprints)")
        }
        if "similarity_group" not in columns:
            connection.execute(
                "ALTER TABLE fingerprints "
                "ADD COLUMN similarity_group TEXT NOT NULL DEFAULT ''"
            )
            connection.execute(
                "UPDATE fingerprints SET similarity_group = model_digest "
                "WHERE similarity_group = ''"
            )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS fingerprints_shape_lookup
            ON fingerprints(version, key_id, rows, vocab_size, id DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS fingerprints_similarity_group
            ON fingerprints(version, key_id, similarity_group, id DESC)
            """
        )
        connection.commit()

    def initialize(self) -> None:
        with self._connect():
            pass

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM fingerprints").fetchone()
            return int(row["n"])

    def recent_metadata(self) -> list[dict[str, Any]]:
        """Return non-logit metadata for private diagnostics and tests."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, model_ref, model_digest, submission_id,
                       submission_order, hotkey, version, compatibility_key,
                       similarity_group, rows, vocab_size, logits_sha256
                FROM fingerprints ORDER BY id DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def compare_and_store(
        self,
        fingerprint: torch.Tensor,
        fingerprint_meta: dict[str, Any],
        *,
        model_ref: str,
        model_digest: str,
        submission_id: str,
        submission_order: str,
        hotkey: str,
        js_p95_max: float,
    ) -> dict[str, Any]:
        """Compare a candidate and enforce the similar-model family limit.

        A near-match joins the closest stored model's similarity family until
        ``max_similar_models`` total members have been admitted. The next member
        is rejected and not inserted. Entries with another probe version,
        secret key identifier, tensor shape, or vocabulary are skipped.
        Architecture metadata is intentionally not trusted as a filter.
        """
        payload, rows, vocab_size, logits_sha256 = _fingerprint_bytes(fingerprint)
        version = str(fingerprint_meta["version"])
        key_id = str(fingerprint_meta["key_id"])
        compatibility_key = str(fingerprint_meta["compatibility_key"])
        if int(fingerprint_meta["vocab_size"]) != vocab_size:
            raise ValueError("fingerprint metadata vocabulary does not match its tensor")
        if int(fingerprint_meta["n_distributions"]) != rows:
            raise ValueError("fingerprint metadata row count does not match its tensor")
        if not model_digest:
            raise ValueError("model_digest is required for the fingerprint bank")

        candidate = torch.from_numpy(
            np.frombuffer(payload, dtype="<f2").reshape(rows, vocab_size).copy()
        )
        closest_row: sqlite3.Row | None = None
        closest_metrics: dict[str, float] | None = None
        compared = 0

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT submission_id, similarity_group FROM fingerprints
                WHERE version = ? AND key_id = ? AND model_digest = ?
                """,
                (version, key_id, model_digest),
            ).fetchone()
            if existing is not None and str(existing["submission_id"]) == submission_id:
                family_size = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) AS n FROM fingerprints
                        WHERE version = ? AND key_id = ? AND similarity_group = ?
                        """,
                        (version, key_id, str(existing["similarity_group"])),
                    ).fetchone()["n"]
                )
                bank_size = int(
                    connection.execute("SELECT COUNT(*) AS n FROM fingerprints").fetchone()["n"]
                )
                connection.commit()
                return {
                    "detected": False,
                    "stored": True,
                    "already_stored": True,
                    "too_similar": family_size > 1,
                    "similar_family_size": family_size,
                    "max_similar_models": self.max_similar_models,
                    "compared": 0,
                    "bank_size": bank_size,
                    "version": version,
                    "js_p95_max": float(js_p95_max),
                    "match": None,
                }
            prior_rows = connection.execute(
                """
                SELECT id, model_ref, model_digest, submission_id,
                       submission_order, hotkey, similarity_group, rows, vocab_size,
                       logits_sha256, logits
                FROM fingerprints
                WHERE version = ? AND key_id = ? AND rows = ? AND vocab_size = ?
                ORDER BY id DESC LIMIT ?
                """,
                (version, key_id, rows, vocab_size, self.max_entries),
            )
            matching_groups: set[str] = set()
            for prior in prior_rows:
                stored = _fingerprint_tensor(
                    prior["logits"],
                    int(prior["rows"]),
                    int(prior["vocab_size"]),
                    str(prior["logits_sha256"]),
                )
                metrics = distribution_metrics(stored, candidate)
                compared += 1
                if closest_metrics is None or metrics["js_p95"] < closest_metrics["js_p95"]:
                    closest_row = prior
                    closest_metrics = metrics
                if float(metrics["js_p95"]) <= float(js_p95_max):
                    matching_groups.add(str(prior["similarity_group"]))

            too_similar = bool(matching_groups)
            family_size_before = 0
            similarity_group = model_digest
            if too_similar:
                assert closest_row is not None
                similarity_group = str(closest_row["similarity_group"])
                placeholders = ",".join("?" for _ in matching_groups)
                family_size_before = int(
                    connection.execute(
                        f"""
                        SELECT COUNT(*) AS n FROM fingerprints
                        WHERE version = ? AND key_id = ?
                          AND similarity_group IN ({placeholders})
                        """,
                        (version, key_id, *sorted(matching_groups)),
                    ).fetchone()["n"]
                )
            if len(matching_groups) > 1:
                placeholders = ",".join("?" for _ in matching_groups)
                connection.execute(
                    f"""
                    UPDATE fingerprints SET similarity_group = ?
                    WHERE version = ? AND key_id = ?
                      AND similarity_group IN ({placeholders})
                    """,
                    (
                        similarity_group,
                        version,
                        key_id,
                        *sorted(matching_groups),
                    ),
                )
            detected = too_similar and family_size_before >= self.max_similar_models
            if not detected:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO fingerprints (
                        created_at, model_ref, model_digest, submission_id,
                        submission_order, hotkey, version, key_id,
                        compatibility_key, similarity_group, rows, vocab_size,
                        logits_sha256, logits
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        time.time(),
                        model_ref,
                        model_digest,
                        submission_id,
                        submission_order,
                        hotkey,
                        version,
                        key_id,
                        compatibility_key,
                        similarity_group,
                        rows,
                        vocab_size,
                        logits_sha256,
                        sqlite3.Binary(payload),
                    ),
                )
                connection.execute(
                    """
                    DELETE FROM fingerprints
                    WHERE id NOT IN (
                        SELECT id FROM fingerprints ORDER BY id DESC LIMIT ?
                    )
                    """,
                    (self.max_entries,),
                )
            similar_family_size = family_size_before
            if not detected:
                similar_family_size = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) AS n FROM fingerprints
                        WHERE version = ? AND key_id = ? AND similarity_group = ?
                        """,
                        (version, key_id, similarity_group),
                    ).fetchone()["n"]
                )
            bank_size = int(
                connection.execute("SELECT COUNT(*) AS n FROM fingerprints").fetchone()["n"]
            )
            connection.commit()

        match = None
        if closest_row is not None and closest_metrics is not None:
            match = {
                "model_ref": str(closest_row["model_ref"]),
                "model_digest": str(closest_row["model_digest"]),
                "submission_id": str(closest_row["submission_id"]),
                "submission_order": str(closest_row["submission_order"]),
                "hotkey": str(closest_row["hotkey"]),
                "metrics": closest_metrics,
            }
        return {
            "detected": detected,
            "stored": not detected,
            "too_similar": too_similar,
            "similar_family_size": similar_family_size,
            "max_similar_models": self.max_similar_models,
            "compared": compared,
            "bank_size": bank_size,
            "version": version,
            "js_p95_max": float(js_p95_max),
            "match": match,
        }


def is_behavioral_copy(
    metrics: dict,
    *,
    js_p95_max: float,
) -> bool:
    return float(metrics["js_p95"]) <= float(js_p95_max)


def _public_value(value: Any) -> str:
    return " ".join(str(value or "").split())


def public_rejection_message(
    decision: dict[str, Any],
    *,
    candidate_model_ref: str,
    candidate_model_digest: str,
    candidate_submission_id: str,
    candidate_submission_order: str,
) -> str:
    """Limited aggregate evidence safe to publish only when the gate fires."""
    if not decision.get("detected") or not decision.get("match"):
        raise ValueError("a public rejection message requires a detected bank match")
    match = decision["match"]
    metrics = match["metrics"]
    return (
        "behavioral copy gate rejected queued model: near-identical behavioral "
        "family has reached its configured limit: "
        f"candidate_model={_public_value(candidate_model_ref)} "
        f"candidate_digest={_public_value(candidate_model_digest)} "
        f"candidate_submission={_public_value(candidate_submission_id)} "
        f"candidate_order={_public_value(candidate_submission_order)} "
        f"matched_model={_public_value(match['model_ref'])} "
        f"matched_digest={_public_value(match['model_digest'])} "
        f"matched_submission={_public_value(match['submission_id'])} "
        f"matched_order={_public_value(match['submission_order'])} "
        f"fingerprint_version={_public_value(decision['version'])} "
        f"similar_family_size={int(decision['similar_family_size'])} "
        f"max_similar_models={int(decision['max_similar_models'])} "
        f"js_p95={metrics['js_p95']:.9g} "
        f"js_p95_max={float(decision['js_p95_max']):.9g}"
    )
