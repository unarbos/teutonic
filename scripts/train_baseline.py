#!/usr/bin/env python3
"""Centralized training baseline — 8xGPU FSDP training on the Teutonic dataset.

Trains the starting Gemma-3-1B model on the same R2 dataset used by the subnet,
measuring throughput (tokens/sec) and loss for comparison with decentralized training.

Usage:
    torchrun --nproc_per_node=8 train_baseline.py --model-dir /tmp/experiment/king

Env vars: same as eval_torch.py (TEUTONIC_R2_*)
"""
import argparse
import io
import json
import logging
import math
import os
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from botocore.config import Config as BotoConfig
import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoConfig

log = logging.getLogger("train")


# ---------------------------------------------------------------------------
# R2 client (same as eval_torch.py)
# ---------------------------------------------------------------------------

class R2:
    def __init__(self):
        self.client = boto3.client(
            "s3",
            endpoint_url=os.environ["TEUTONIC_R2_ENDPOINT"],
            aws_access_key_id=os.environ["TEUTONIC_R2_ACCESS_KEY"],
            aws_secret_access_key=os.environ["TEUTONIC_R2_SECRET_KEY"],
            region_name="auto",
            config=BotoConfig(retries={"max_attempts": 5, "mode": "adaptive"}),
        )
        self.bucket = os.environ.get("TEUTONIC_R2_BUCKET", "constantinople")

    def get(self, key):
        try:
            return json.loads(
                self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            )
        except Exception:
            return None

    def download_shard(self, shard_key, retries=5):
        for attempt in range(retries):
            try:
                t0 = time.time()
                raw = self.client.get_object(Bucket=self.bucket, Key=shard_key)["Body"].read()
                buf = io.BytesIO(raw)
                buf.read(6)
                ver = struct.unpack("BB", buf.read(2))
                hl = struct.unpack("<H" if ver[0] == 1 else "<I", buf.read(2 if ver[0] == 1 else 4))[0]
                buf.read(hl)
                data_offset = buf.tell()
                elapsed = time.time() - t0
                n_tokens = (len(raw) - data_offset) // 4
                if is_rank0():
                    log.info("shard %s: %.0fM tokens, %.1f MB in %.1fs (%.0f MB/s)",
                             shard_key, n_tokens / 1e6, len(raw) / 1e6, elapsed,
                             len(raw) / 1e6 / max(elapsed, 0.01))
                return np.frombuffer(raw[data_offset:], dtype="<u4").astype(np.int64)
            except Exception as e:
                if is_rank0():
                    log.warning("shard download attempt %d/%d failed: %s", attempt + 1, retries, e)
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"failed to download {shard_key} after {retries} attempts")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShardDataset(torch.utils.data.Dataset):
    """Wraps a single shard's token array into seq_len chunks."""

    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_sequences = len(tokens) // seq_len

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len]
        return torch.from_numpy(chunk.copy())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_rank0():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_val_loss(model, val_tokens, seq_len, batch_size, device):
    """Compute average cross-entropy on validation data."""
    model.eval()
    dataset = ShardDataset(val_tokens, seq_len)
    n_eval = min(len(dataset), 500)
    indices = list(range(n_eval))
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, n_eval, batch_size):
            batch_idx = indices[i : i + batch_size]
            batch = torch.stack([dataset[j] for j in batch_idx]).to(device)
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += labels.numel()

    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    if dist.is_initialized():
        t = torch.tensor([total_loss, total_tokens], device=device, dtype=torch.float64)
        dist.all_reduce(t)
        avg_loss = (t[0] / t[1]).item()
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, step, loss, out_dir):
    if not is_rank0():
        return
    ckpt_dir = Path(out_dir) / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
        state_dict = model.state_dict()

    if is_rank0():
        from safetensors.torch import save_file
        save_file(state_dict, str(ckpt_dir / "model.safetensors"))
        torch.save({
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss": loss,
        }, str(ckpt_dir / "training_state.pt"))
        cfg_src = Path(os.environ.get("MODEL_DIR", "/tmp/experiment/king")) / "config.json"
        if cfg_src.exists():
            import shutil
            shutil.copy(cfg_src, ckpt_dir / "config.json")
        log.info("checkpoint saved: %s", ckpt_dir)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Centralized training baseline")
    parser.add_argument("--model-dir", default="/tmp/experiment/king")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = run for --hours")
    parser.add_argument("--hours", type=float, default=0, help="0 = run forever")
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-every", type=int, default=500, help="Validate every N steps")
    parser.add_argument("--ckpt-every", type=int, default=2000, help="Checkpoint every N steps")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out-dir", default="/tmp/experiment/train_output")
    parser.add_argument("--val-shard-idx", type=int, default=0, help="Shard index for validation")
    parser.add_argument("--start-shard", type=int, default=1, help="First training shard index")
    parser.add_argument("--val-batch-size", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s [%(process)d] %(message)s",
        datefmt="%H:%M:%S",
    )

    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    os.environ["MODEL_DIR"] = args.model_dir

    if is_rank0():
        log.info("=" * 70)
        log.info("CENTRALIZED TRAINING BASELINE")
        log.info("  model:       %s", args.model_dir)
        log.info("  world_size:  %d", world_size)
        log.info("  micro_batch: %d", args.micro_batch)
        log.info("  grad_accum:  %d", args.grad_accum)
        log.info("  eff_batch:   %d seqs = %d tokens",
                 args.micro_batch * args.grad_accum * world_size,
                 args.micro_batch * args.grad_accum * world_size * args.seq_len)
        log.info("  lr:          %s -> %s", args.lr, args.min_lr)
        log.info("  warmup:      %d steps", args.warmup_steps)
        log.info("  duration:    %s", f"{args.max_steps} steps" if args.max_steps else f"{args.hours}h")
        log.info("=" * 70)

    # --- Load model ---
    if is_rank0():
        log.info("loading model from %s ...", args.model_dir)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    if is_rank0():
        log.info("model loaded: %.2fB params", n_params)

    # --- FSDP wrap ---
    from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Gemma3DecoderLayer},
    )
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        use_orig_params=True,
    )

    if is_rank0():
        log.info("FSDP wrapped, device=%s", device)

    # --- Optimizer & scheduler ---
    total_steps = args.max_steps if args.max_steps > 0 else 10_000_000
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.95),
        fused=True,
    )
    scheduler = get_cosine_schedule(optimizer, args.warmup_steps, total_steps,
                                    min_lr_ratio=args.min_lr / args.lr)

    # --- R2 + manifest ---
    r2 = R2()
    manifest = r2.get("dataset/v1/manifest.json")
    if not manifest:
        log.error("could not fetch dataset manifest")
        sys.exit(1)
    n_shards = manifest["total_shards"]
    if is_rank0():
        log.info("dataset: %d shards, val_shard=%d", n_shards, args.val_shard_idx)

    # --- Load validation shard ---
    val_shard_key = manifest["shards"][args.val_shard_idx]["key"]
    if is_rank0():
        log.info("downloading validation shard %s ...", val_shard_key)
    val_tokens = r2.download_shard(val_shard_key)

    # --- Initial validation ---
    if is_rank0():
        log.info("computing initial validation loss...")
    init_val_loss = compute_val_loss(model, val_tokens, args.seq_len, args.val_batch_size, device)
    if is_rank0():
        log.info("initial val loss: %.4f (perplexity: %.2f)", init_val_loss, math.exp(init_val_loss))

    # --- Prepare output ---
    out_dir = Path(args.out_dir)
    if is_rank0():
        out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "train_log.jsonl"

    _last_r2_upload = [0.0]

    def log_step(data):
        if not is_rank0():
            return
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(log_file, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")
        now = time.time()
        if now - _last_r2_upload[0] > 120:
            try:
                r2.client.upload_file(
                    str(log_file), r2.bucket, "experiments/baseline_train_log.jsonl",
                )
                _last_r2_upload[0] = now
            except Exception:
                pass

    log_step({
        "event": "start",
        "config": vars(args),
        "n_params_B": round(n_params, 3),
        "init_val_loss": round(init_val_loss, 6),
        "init_val_ppl": round(math.exp(init_val_loss), 2),
        "world_size": world_size,
    })

    # --- Training loop ---
    model.train()
    step = 0
    total_tokens_processed = 0
    running_loss = 0.0
    loss_count = 0
    t_start = time.time()
    deadline = t_start + args.hours * 3600 if args.hours > 0 else float("inf")

    shard_order = list(range(args.start_shard, n_shards))
    rng = np.random.default_rng(42)
    rng.shuffle(shard_order)
    shard_cursor = 0

    if is_rank0():
        log.info("training starts, shard_order has %d shards", len(shard_order))

    while True:
        if args.max_steps > 0 and step >= args.max_steps:
            break
        if time.time() > deadline:
            break

        # Load next shard
        if shard_cursor >= len(shard_order):
            rng.shuffle(shard_order)
            shard_cursor = 0
            if is_rank0():
                log.info("epoch boundary — reshuffled shards")

        shard_idx = shard_order[shard_cursor]
        shard_key = manifest["shards"][shard_idx]["key"]
        shard_cursor += 1

        try:
            tokens = r2.download_shard(shard_key)
        except Exception as e:
            if is_rank0():
                log.warning("failed to download shard %s: %s, skipping", shard_key, e)
            continue

        dataset = ShardDataset(tokens, args.seq_len)
        if len(dataset) == 0:
            continue

        indices = list(range(len(dataset)))
        rng.shuffle(indices)

        # Per-rank slicing
        rank_indices = indices[local_rank::world_size]

        micro_steps_in_shard = 0
        for batch_start in range(0, len(rank_indices), args.micro_batch):
            if args.max_steps > 0 and step >= args.max_steps:
                break
            if time.time() > deadline:
                break

            batch_idx = rank_indices[batch_start : batch_start + args.micro_batch]
            if len(batch_idx) == 0:
                continue

            batch = torch.stack([dataset[i] for i in batch_idx]).to(device)
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]

            is_accum_step = (micro_steps_in_shard + 1) % args.grad_accum != 0

            if is_accum_step:
                with model.no_sync():
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                    )
                    loss_scaled = loss / args.grad_accum
                    loss_scaled.backward()
            else:
                outputs = model(input_ids=input_ids)
                logits = outputs.logits
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )
                loss_scaled = loss / args.grad_accum
                loss_scaled.backward()

                grad_norm = model.clip_grad_norm_(args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                batch_tokens = args.micro_batch * args.grad_accum * world_size * (args.seq_len - 1)
                total_tokens_processed += batch_tokens
                running_loss += loss.item()
                loss_count += 1

                if step % args.log_every == 0 and is_rank0():
                    elapsed = time.time() - t_start
                    tps = total_tokens_processed / elapsed
                    avg_loss = running_loss / max(loss_count, 1)
                    lr_now = scheduler.get_last_lr()[0]
                    mem = torch.cuda.max_memory_allocated(device) / 1e9

                    log.info(
                        "step %d | loss %.4f | ppl %.2f | lr %.2e | "
                        "%.0f tok/s | %.1fB total | grad_norm %.2f | mem %.1fGB",
                        step, avg_loss, math.exp(avg_loss), lr_now,
                        tps, total_tokens_processed / 1e9,
                        grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        mem,
                    )
                    log_step({
                        "event": "step",
                        "step": step,
                        "loss": round(avg_loss, 6),
                        "perplexity": round(math.exp(min(avg_loss, 20)), 2),
                        "lr": lr_now,
                        "tokens_per_sec": round(tps, 0),
                        "total_tokens_B": round(total_tokens_processed / 1e9, 3),
                        "grad_norm": round(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm, 4),
                        "gpu_mem_GB": round(mem, 1),
                        "elapsed_s": round(elapsed, 0),
                        "shard_idx": shard_idx,
                    })
                    running_loss = 0.0
                    loss_count = 0

                if step % args.val_every == 0:
                    val_loss = compute_val_loss(model, val_tokens, args.seq_len, args.val_batch_size, device)
                    if is_rank0():
                        log.info("VALIDATION step %d: loss %.4f ppl %.2f", step, val_loss, math.exp(val_loss))
                        log_step({
                            "event": "validation",
                            "step": step,
                            "val_loss": round(val_loss, 6),
                            "val_ppl": round(math.exp(val_loss), 2),
                            "total_tokens_B": round(total_tokens_processed / 1e9, 3),
                        })
                    model.train()

                if step % args.ckpt_every == 0:
                    dist.barrier()
                    save_checkpoint(model, optimizer, scheduler, step, loss.item(), args.out_dir)
                    dist.barrier()

            micro_steps_in_shard += 1

        del tokens, dataset
        torch.cuda.empty_cache()

    # --- Final validation and checkpoint ---
    final_val_loss = compute_val_loss(model, val_tokens, args.seq_len, args.val_batch_size, device)
    elapsed = time.time() - t_start

    if is_rank0():
        tps = total_tokens_processed / max(elapsed, 1)
        log.info("=" * 70)
        log.info("TRAINING COMPLETE")
        log.info("  steps:          %d", step)
        log.info("  elapsed:        %.1f hours", elapsed / 3600)
        log.info("  total tokens:   %.2fB", total_tokens_processed / 1e9)
        log.info("  avg tok/s:      %.0f", tps)
        log.info("  init val loss:  %.4f (ppl %.2f)", init_val_loss, math.exp(init_val_loss))
        log.info("  final val loss: %.4f (ppl %.2f)", final_val_loss, math.exp(final_val_loss))
        log.info("  improvement:    %.4f nats/token", init_val_loss - final_val_loss)
        log.info("=" * 70)

        log_step({
            "event": "done",
            "step": step,
            "elapsed_hours": round(elapsed / 3600, 2),
            "total_tokens_B": round(total_tokens_processed / 1e9, 3),
            "avg_tokens_per_sec": round(tps, 0),
            "init_val_loss": round(init_val_loss, 6),
            "init_val_ppl": round(math.exp(init_val_loss), 2),
            "final_val_loss": round(final_val_loss, 6),
            "final_val_ppl": round(math.exp(final_val_loss), 2),
            "improvement_nats": round(init_val_loss - final_val_loss, 6),
        })

    dist.barrier()
    save_checkpoint(model, optimizer, scheduler, step, final_val_loss, args.out_dir)
    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
