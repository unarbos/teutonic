"""Shared hyperparameter configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HParams:
    max_batches: int = 32
    min_batches: int = 4
    micro_bs: int = 2
    topk: int = 64
    lr: float = 1e-3
    outer_lr: float = 0.4
    max_grad_norm: float = 1.0
    upload_budget_s: float = 10.0
    n_loss_spot_checks: int = 8
    n_probes: int = 3
    probe_slice_size: int = 128
    n_probe_params: int = 3
    score_history_len: int = 8
    score_ema_alpha: float = 0.3
    eval_timeout: float = 60.0
    apply_timeout: float = 60.0
