"""Miner submission format: compressed gradient + loss ledger + gradient probes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class MinerSubmission:
    """Everything a miner uploads for one window."""

    uid: int
    window: int
    compressed_gradients: dict[str, dict[str, Any]]
    loss_ledger: list[float]
    grad_probes: dict[int, torch.Tensor] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "window": self.window,
            "compressed_gradients": self.compressed_gradients,
            "loss_ledger": self.loss_ledger,
            "grad_probes": {k: v.cpu() for k, v in self.grad_probes.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MinerSubmission:
        return cls(
            uid=d["uid"],
            window=d["window"],
            compressed_gradients=d["compressed_gradients"],
            loss_ledger=d["loss_ledger"],
            grad_probes=d.get("grad_probes", {}),
        )

    def storage_key(self) -> str:
        return self.make_storage_key(self.window, self.uid)

    @staticmethod
    def make_storage_key(window: int, uid: int) -> str:
        return f"gradient/{window}/{uid}"
