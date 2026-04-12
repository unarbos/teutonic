"""Shared hyperparameter configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HParams:
    n_batches: int = 6
    micro_bs: int = 2
    topk: int = 64
    lr: float = 1e-3
    outer_lr: float = 0.4
    max_grad_norm: float = 1.0
    n_loss_spot_checks: int = 8
    n_probes: int = 3
    probe_param_name: str = "layers.0.attention.wq.weight"
    probe_slice_start: int = 0
    probe_slice_end: int = 128
    score_history_len: int = 8
    score_ema_alpha: float = 0.3
    eval_timeout: float = 60.0
    apply_timeout: float = 60.0
