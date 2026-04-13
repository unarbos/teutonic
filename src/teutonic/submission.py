"""Miner submission format: compressed gradient + loss ledger + gradient probes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class MinerSubmission:
    """Everything a miner uploads for one window.

    ``n_batches_trained`` records how many batches from the deterministic
    sequence the miner actually completed.  ``loss_ledger`` has exactly
    that many entries.

    ``grad_probes`` maps ``batch_index -> param_name -> gradient_slice``
    so the validator can verify any parameter at any batch index.
    """

    uid: int
    window: int
    compressed_gradients: dict[str, dict[str, Any]]
    loss_ledger: list[float]
    n_batches_trained: int = 0
    grad_probes: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)

    def __post_init__(self):
        if self.n_batches_trained == 0 and self.loss_ledger:
            self.n_batches_trained = len(self.loss_ledger)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "window": self.window,
            "n_batches_trained": self.n_batches_trained,
            "compressed_gradients": self.compressed_gradients,
            "loss_ledger": self.loss_ledger,
            "grad_probes": {
                k: {pname: t.cpu() for pname, t in pdict.items()}
                for k, pdict in self.grad_probes.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MinerSubmission:
        loss_ledger = d["loss_ledger"]
        raw_probes = d.get("grad_probes", {})
        grad_probes: dict[int, dict[str, torch.Tensor]] = {}
        for k, v in raw_probes.items():
            if isinstance(v, dict):
                grad_probes[int(k)] = v
            elif isinstance(v, torch.Tensor):
                grad_probes[int(k)] = {"_legacy": v}
        return cls(
            uid=d["uid"],
            window=d["window"],
            n_batches_trained=d.get("n_batches_trained", len(loss_ledger)),
            compressed_gradients=d["compressed_gradients"],
            loss_ledger=loss_ledger,
            grad_probes=grad_probes,
        )

    def storage_key(self) -> str:
        return self.make_storage_key(self.window, self.uid)

    @staticmethod
    def make_storage_key(window: int, uid: int) -> str:
        return f"gradient/{window}/{uid}"
