"""Delta validation: safetensors loading, architecture match, bounding box."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors

from .config import BoundingBoxConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    valid: bool
    reason: str = ""
    detail: str = ""
    delta_l2_global: float = 0.0
    delta_linf_max: float = 0.0
    n_changed_params: int = 0


def sha256_of_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_directory(directory: str | Path) -> str:
    """SHA256 over all .safetensors files in a directory, sorted by name."""
    directory = Path(directory)
    h = hashlib.sha256()
    for p in sorted(directory.glob("*.safetensors")):
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()


def load_safetensors_from_dir(directory: str | Path) -> dict[str, torch.Tensor]:
    """Load all .safetensors files from a directory into a single state dict."""
    directory = Path(directory)
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {directory}")

    state_dict: dict[str, torch.Tensor] = {}
    for f in files:
        state_dict.update(load_safetensors(str(f)))
    return state_dict


def validate_challenger(
    king_dir: str | Path,
    challenger_dir: str | Path,
    bbox: BoundingBoxConfig,
    expected_hash: str | None = None,
) -> ValidationResult:
    """Validate a challenger model against the king.

    1. Verify safetensors-only (no .bin/.pt/.pkl)
    2. Verify hash if provided
    3. Load both models
    4. Verify architecture match (same keys, shapes, dtypes)
    5. Compute delta, check bounding box
    """
    king_dir = Path(king_dir)
    challenger_dir = Path(challenger_dir)

    # Check for forbidden file types
    for pattern in ("*.bin", "*.pt", "*.pkl", "*.pickle"):
        if list(challenger_dir.glob(pattern)):
            return ValidationResult(
                valid=False,
                reason="forbidden_file_type",
                detail=f"Challenger contains {pattern} files",
            )

    safetensor_files = list(challenger_dir.glob("*.safetensors"))
    if not safetensor_files:
        return ValidationResult(
            valid=False,
            reason="no_safetensors",
            detail="No .safetensors files found in challenger repo",
        )

    # Hash verification
    if expected_hash:
        actual_hash = sha256_of_directory(challenger_dir)
        if actual_hash != expected_hash:
            return ValidationResult(
                valid=False,
                reason="hash_mismatch",
                detail=f"Expected {expected_hash[:16]}..., got {actual_hash[:16]}...",
            )

    # Load state dicts
    try:
        king_sd = load_safetensors_from_dir(king_dir)
    except Exception as e:
        return ValidationResult(valid=False, reason="king_load_failed", detail=str(e))

    try:
        challenger_sd = load_safetensors_from_dir(challenger_dir)
    except Exception as e:
        return ValidationResult(valid=False, reason="challenger_load_failed", detail=str(e))

    # Architecture match: keys
    king_keys = set(king_sd.keys())
    challenger_keys = set(challenger_sd.keys())

    missing = king_keys - challenger_keys
    extra = challenger_keys - king_keys
    if missing:
        return ValidationResult(
            valid=False,
            reason="missing_params",
            detail=f"Missing parameters: {sorted(missing)[:5]}",
        )
    if extra:
        return ValidationResult(
            valid=False,
            reason="extra_params",
            detail=f"Extra parameters: {sorted(extra)[:5]}",
        )

    # Architecture match: shapes and dtypes
    for name in king_keys:
        kt, ct = king_sd[name], challenger_sd[name]
        if kt.shape != ct.shape:
            return ValidationResult(
                valid=False,
                reason="shape_mismatch",
                detail=f"{name}: king {kt.shape} vs challenger {ct.shape}",
            )
        if kt.dtype != ct.dtype:
            return ValidationResult(
                valid=False,
                reason="dtype_mismatch",
                detail=f"{name}: king {kt.dtype} vs challenger {ct.dtype}",
            )

    # Compute delta and check bounding box
    delta_l2_sq = 0.0
    delta_linf = 0.0
    n_changed = 0

    for name in king_keys:
        # Check frozen layers
        if any(name.startswith(prefix) for prefix in bbox.frozen_param_prefixes):
            delta = (challenger_sd[name].float() - king_sd[name].float()).abs()
            if delta.max().item() > 1e-8:
                return ValidationResult(
                    valid=False,
                    reason="frozen_param_modified",
                    detail=f"Frozen parameter {name} was modified",
                )
            continue

        delta = challenger_sd[name].float() - king_sd[name].float()

        # L-inf per element
        linf = delta.abs().max().item()
        delta_linf = max(delta_linf, linf)
        if linf > bbox.max_linf:
            return ValidationResult(
                valid=False,
                reason="linf_violation",
                detail=f"{name}: max |delta|={linf:.6f} > {bbox.max_linf}",
            )

        # L2 per tensor
        l2 = delta.norm(2).item()
        if bbox.max_l2_per_tensor is not None and l2 > bbox.max_l2_per_tensor:
            return ValidationResult(
                valid=False,
                reason="l2_per_tensor_violation",
                detail=f"{name}: ||delta||_2={l2:.4f} > {bbox.max_l2_per_tensor}",
            )

        delta_l2_sq += l2 ** 2

        if linf > 1e-8:
            n_changed += 1

    delta_l2_global = delta_l2_sq ** 0.5

    # Global L2
    if bbox.max_l2_global is not None and delta_l2_global > bbox.max_l2_global:
        return ValidationResult(
            valid=False,
            reason="l2_global_violation",
            detail=f"||delta||_2={delta_l2_global:.4f} > {bbox.max_l2_global}",
        )

    # Sparsity
    total_params = len(king_keys)
    if bbox.max_sparsity_frac is not None:
        frac = n_changed / total_params if total_params > 0 else 0
        if frac > bbox.max_sparsity_frac:
            return ValidationResult(
                valid=False,
                reason="sparsity_violation",
                detail=f"{n_changed}/{total_params} params changed ({frac:.2%}) > {bbox.max_sparsity_frac:.2%}",
            )

    return ValidationResult(
        valid=True,
        delta_l2_global=delta_l2_global,
        delta_linf_max=delta_linf,
        n_changed_params=n_changed,
    )
