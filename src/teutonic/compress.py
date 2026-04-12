"""Top-k gradient compression / decompression.

Simplified from Templar's compress.py.  Keeps the same conceptual pipeline
(top-k sparsification with optional quantisation) but drops the 12-bit
index packing for clarity.  Can be swapped for the full Templar version
in production.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class TopKCompressor:
    """Select the top-k largest-magnitude gradient elements per parameter."""

    def __init__(self, topk: int = 64):
        self.topk = topk

    def compress(
        self, grad: torch.Tensor
    ) -> dict[str, Any]:
        """Return sparse representation of *grad* (flattened)."""
        flat = grad.flatten()
        k = min(self.topk, flat.numel())
        vals, idxs = torch.topk(flat.abs(), k, sorted=False)
        vals = flat[idxs]  # keep sign
        return {
            "idxs": idxs.cpu(),
            "vals": vals.cpu(),
            "shape": tuple(grad.shape),
        }

    def decompress(self, compressed: dict[str, Any], device: torch.device | str = "cpu") -> torch.Tensor:
        """Reconstruct a dense gradient from the sparse representation."""
        shape = compressed["shape"]
        numel = 1
        for s in shape:
            numel *= s

        dense = torch.zeros(
            shape, device=device, dtype=compressed["vals"].dtype
        ).flatten()

        idxs = compressed["idxs"].to(device)
        vals = compressed["vals"].to(device)

        # Bounds check: indices must be within [0, numel)
        if idxs.numel() > 0 and (idxs.max() >= numel or idxs.min() < 0):
            logger.warning("Compressed indices out of bounds (max=%d, numel=%d), skipping",
                          idxs.max().item(), numel)
            return dense.reshape(shape)

        dense.scatter_(0, idxs, vals)
        return dense.reshape(shape)


def compress_model_gradients(
    model: torch.nn.Module, compressor: TopKCompressor
) -> dict[str, dict[str, Any]]:
    """Compress gradients for all parameters that have them."""
    result: dict[str, dict[str, Any]] = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            result[name] = compressor.compress(p.grad)
    return result


def decompress_and_apply(
    model: torch.nn.Module,
    compressed_grads: dict[str, dict[str, Any]],
    compressor: TopKCompressor,
    lr: float,
) -> None:
    """Decompress gradients and apply them to the model via SGD step.

    Skips parameters whose decompressed gradient contains NaN/Inf.
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in compressed_grads:
                grad = compressor.decompress(compressed_grads[name], device=p.device)
                if not torch.isfinite(grad).all():
                    logger.warning("Non-finite gradient for %s, skipping apply", name)
                    continue
                p.data.sub_(grad, alpha=lr)
