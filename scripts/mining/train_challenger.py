#!/usr/bin/env python3
"""Teutonic mining harness — train a challenger that beats the current king.

Runs on a multi-B200 box. Pipeline:
  1. Discover current king from R2 dashboard (repo + revision).
  2. Download king from HF (snapshot_download, pinned revision).
  3. Pull a few dataset shards from Hippius (pretokenized .npy uint32, seq_len=2048).
  4. Score sample sequences with the king (avg next-token loss).
  5. Build a curriculum (general / hard / easy buckets, drop suspicious).
  6. Train a LoRA adapter with torchrun multi-GPU on the chosen training mix.
  7. Merge LoRA into the base weights -> standalone candidate dir.
  8. Offline paired eval candidate vs king on a held-out shard slice
     (mirrors validator's compute_paired_losses + bootstrap LCB > delta).
  9. Emit a JSON verdict file. If accepted, optionally upload to HF.

Designed to be re-run iteratively (--max-iters): if first attempt's mu_hat
falls short, training is re-run with a different seed / more steps until
the budget is spent.

This script is meant to live on the GPU box (e.g. /root/teutonic-mining/)
and be invoked there. It does NOT touch bittensor — that step is handled
by submit_challenger.py on the templar host where the wallet lives.

REQUIRED — coldkey prefix in --upload-repo (since 2026-04-29):
  The Teutonic-XXIV validator rejects any HF repo whose name doesn't
  contain the first 8 ss58 chars of the miner's coldkey (case-insensitive
  substring, in either the HF account or the model basename). This is
  an anti-impersonation gate — only the legit coldkey owner can publish
  a repo whose name embeds *their* coldkey. Imposters who lift somebody
  else's URL end up advertising the victim's coldkey on chain.

  This script doesn't have wallet access so it can't enforce locally —
  the orchestrator (run_pipeline.sh) and the on-chain submitter
  (submit_challenger.py) both check before they burn HF / TAO. Pass an
  --upload-repo whose full id contains your coldkey prefix, e.g.
      myaccount/Teutonic-XXIV-5DhAqMpd-v3
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import logging
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# Mining harness lives at teutonic/scripts/mining/; bootstrap workspace root
# onto sys.path so `import teutonic.quasar` registers the Quasar arch.
_workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)
import teutonic.quasar  # noqa: F401  registers Quasar with AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [train_challenger] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_challenger")

# ---------------------------------------------------------------------------
# Defaults (mirror validator constants where applicable)
# ---------------------------------------------------------------------------
SEQ_LEN = 2048
EVAL_ALPHA = 0.001
LM_HEAD_CHUNK = 256
DASHBOARD_URL = os.environ.get(
    "TEUTONIC_DASHBOARD_URL",
    "https://s3.hippius.com/teutonic-sn3/dashboard.json",
)
HIPPIUS_BASE = "https://s3.hippius.com/teutonic-sn3"


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------
def parse_npy_header(raw: bytes) -> tuple[int, dict]:
    buf = io.BytesIO(raw)
    if buf.read(6) != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack("<H" if ver[0] == 1 else "<I",
                       buf.read(2 if ver[0] == 1 else 4))[0]
    header = eval(buf.read(hl).decode("latin1").strip())
    return buf.tell(), header


def load_shard(path: Path, seq_len: int = SEQ_LEN) -> tuple[np.ndarray, int]:
    """Shards are 1D uint32 arrays (concatenated tokens). Reshape into
    (n_sequences, seq_len) by truncating tail so it divides cleanly.
    Matches validator's slicing semantics in eval_torch.fetch_sequences.
    """
    raw = path.read_bytes()
    data_offset, header = parse_npy_header(raw)
    shape = header["shape"]
    flat = np.frombuffer(raw[data_offset:], dtype="<u4")
    if len(shape) == 1:
        n_total = shape[0]
        n_seq = n_total // seq_len
        arr = flat[: n_seq * seq_len].reshape(n_seq, seq_len)
    elif len(shape) == 2:
        n_seq, seq_len = shape
        arr = flat.reshape(n_seq, seq_len)
    else:
        raise ValueError(f"unexpected shard shape {shape}")
    return arr, seq_len


def download_shard(shard_key: str, out: Path) -> Path:
    if out.exists() and out.stat().st_size > 1024:
        log.info("shard cached: %s (%.1f GB)", out, out.stat().st_size / 1e9)
        return out
    url = f"{HIPPIUS_BASE}/{shard_key}"
    log.info("downloading %s -> %s", url, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["curl", "-fsSL", "-o", str(out), url])
    return out


def fetch_manifest(cache: Path) -> dict:
    p = cache / "manifest.json"
    if not p.exists():
        url = f"{HIPPIUS_BASE}/dataset/v2/manifest.json"
        log.info("downloading manifest from %s", url)
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["curl", "-fsSL", "-o", str(p), url])
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# King discovery
# ---------------------------------------------------------------------------
def fetch_king() -> dict:
    import urllib.request
    log.info("fetching dashboard %s", DASHBOARD_URL)
    with urllib.request.urlopen(DASHBOARD_URL, timeout=30) as r:
        d = json.loads(r.read())
    k = d["king"]
    log.info("king: repo=%s revision=%s reign=%d hotkey=%s",
             k["hf_repo"], (k.get("king_revision") or "HEAD")[:12],
             k.get("reign_number", 0), k.get("hotkey", "?")[:16])
    return k


def sha256_dir(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.glob("*.safetensors")):
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Paired eval (mirrors eval_torch.compute_paired_losses + bootstrap)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_per_seq_loss(model, token_batches, device, chunk=LM_HEAD_CHUNK):
    """Average per-token cross-entropy per sequence (matches eval_torch)."""
    input_ids = torch.tensor(token_batches, dtype=torch.long, device=device)
    # Reset stateful arch (Quasar latent memory) before each batch — see
    # eval_torch.compute_paired_losses for rationale. No-op for stock HF archs.
    if hasattr(model, "reset_state"):
        model.reset_state()
    out = model.model(input_ids)
    hidden = out.last_hidden_state
    lm_head = model.lm_head
    n_pos = input_ids.size(1) - 1
    total = torch.zeros(len(token_batches), device=device)
    for i in range(0, n_pos, chunk):
        end = min(i + chunk, n_pos)
        logits = lm_head(hidden[:, i:end, :])
        labels = input_ids[:, i + 1:end + 1]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        )
        total += loss.reshape(len(token_batches), -1).sum(dim=1)
        del logits, loss
    return (total / n_pos).cpu().tolist()


def paired_eval(king_dir: str, chall_dir: str, shard: np.ndarray,
                indices: list[int], device: str, batch_size: int = 8,
                n_bootstrap: int = 10000, alpha: float = EVAL_ALPHA) -> dict:
    """Mirrors validator's paired bootstrap test on a single GPU.

    Acceptance floor delta = 1/N (N = len(indices)) matches the validator,
    where the floor scales with the bootstrap's own resolution.
    """
    delta = 1.0 / len(indices) if indices else 0.0
    log.info("paired_eval: loading king %s on %s", king_dir, device)
    king = AutoModelForCausalLM.from_pretrained(
        king_dir, torch_dtype=torch.bfloat16, device_map={"": device},
        use_safetensors=True,
    )
    king.eval()
    log.info("paired_eval: loading challenger %s on %s", chall_dir, device)
    chall = AutoModelForCausalLM.from_pretrained(
        chall_dir, torch_dtype=torch.bfloat16, device_map={"": device},
        use_safetensors=True,
    )
    chall.eval()

    diffs = []
    king_sum = chall_sum = 0.0
    n_done = 0
    t0 = time.time()
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        toks = [shard[j].tolist() for j in batch_idx]
        kl = compute_per_seq_loss(king, toks, device)
        cl = compute_per_seq_loss(chall, toks, device)
        for k, c in zip(kl, cl):
            diffs.append(k - c)
            king_sum += k
            chall_sum += c
            n_done += 1
        if (i // batch_size) % 5 == 0:
            mu = float(np.mean(diffs))
            log.info("eval %d/%d | mu_hat=%.6f | king=%.4f chall=%.4f | %.1fs",
                     n_done, len(indices), mu,
                     king_sum / n_done, chall_sum / n_done, time.time() - t0)

    diffs = np.asarray(diffs, dtype=np.float64)
    mu_hat = float(diffs.mean())
    boot = np.empty(n_bootstrap)
    rng = np.random.default_rng(0xB007)
    for b in range(n_bootstrap):
        boot[b] = diffs[rng.integers(0, len(diffs), size=len(diffs))].mean()
    lcb = float(np.quantile(boot, alpha))
    accepted = lcb > delta
    res = {
        "n_eval": n_done,
        "mu_hat": mu_hat,
        "lcb": lcb,
        "delta": delta,
        "alpha": alpha,
        "accepted": accepted,
        "avg_king_loss": king_sum / n_done,
        "avg_chall_loss": chall_sum / n_done,
        "elapsed_s": time.time() - t0,
    }
    log.info("paired_eval: mu_hat=%.6f lcb=%.6f accepted=%s",
             mu_hat, lcb, accepted)
    del king, chall
    torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------------------
# Sample scoring + curriculum (single-GPU; lifted from training_bundle)
# ---------------------------------------------------------------------------
def score_and_curate(king_dir: str, shards: list[np.ndarray],
                     n_score: int, train_per_iter: int, val_size: int,
                     seed: int, device: str, work: Path) -> tuple[Path, Path]:
    """Score `n_score` random samples on the king, bucket, write train/val jsonl."""
    rng = np.random.default_rng(seed)
    cands = []
    for s_idx, shard in enumerate(shards):
        if len(shard) == 0:
            continue
        n_take = max(n_score // len(shards), 32)
        idxs = rng.choice(len(shard), size=min(n_take, len(shard)), replace=False)
        for j in idxs:
            cands.append((s_idx, int(j)))
    rng.shuffle(cands)

    log.info("scoring %d samples with king on %s", len(cands), device)
    model = AutoModelForCausalLM.from_pretrained(
        king_dir, torch_dtype=torch.bfloat16, device_map={"": device},
        use_safetensors=True,
    )
    model.eval()

    rows = []
    BATCH = 8
    for i in range(0, len(cands), BATCH):
        chunk = cands[i:i + BATCH]
        toks = [shards[s][j].tolist() for s, j in chunk]
        losses = compute_per_seq_loss(model, toks, device)
        for (s_idx, j), tok, loss in zip(chunk, toks, losses):
            arr = np.asarray(tok)
            unique_r = float(len(set(tok)) / len(tok))
            rep_r = float(np.mean(arr[1:] == arr[:-1])) if len(arr) > 1 else 0.0
            ngrams = [tuple(tok[k:k + 4]) for k in range(len(tok) - 3)]
            rep_ng = 1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0
            rows.append({
                "shard": s_idx,
                "idx": j,
                "loss": float(loss),
                "unique_r": unique_r,
                "rep_r": rep_r,
                "rep_ng4": rep_ng,
                "tokens": tok,
            })

    del model
    torch.cuda.empty_cache()

    losses = np.asarray([r["loss"] for r in rows])
    p50 = float(np.percentile(losses, 50))
    p85 = float(np.percentile(losses, 85))

    def bucket(r):
        if r["rep_r"] > 0.2 or r["rep_ng4"] > 0.5 or r["unique_r"] < 0.05:
            return "suspicious"
        if r["loss"] >= p85:
            return "hard"
        if r["loss"] >= p50 * 0.8:
            return "general"
        return "easy"

    for r in rows:
        r["bucket"] = bucket(r)
    counts = {b: sum(1 for r in rows if r["bucket"] == b)
              for b in ("general", "hard", "easy", "suspicious")}
    log.info("scoring done: p50=%.3f p85=%.3f buckets=%s", p50, p85, counts)

    clean = [r for r in rows if r["bucket"] != "suspicious"]
    rng2 = np.random.default_rng(seed + 1)
    rng2.shuffle(clean)
    val_rows = clean[:val_size]
    val_keys = {(r["shard"], r["idx"]) for r in val_rows}
    pool = [r for r in clean if (r["shard"], r["idx"]) not in val_keys]

    general = [r for r in pool if r["bucket"] == "general"]
    hard = [r for r in pool if r["bucket"] == "hard"]
    easy = [r for r in pool if r["bucket"] == "easy"]
    n_general = int(train_per_iter * 0.6)
    n_hard = int(train_per_iter * 0.3)
    n_easy = train_per_iter - n_general - n_hard

    train_rows = []
    for src, n in ((general, n_general), (hard, n_hard), (easy, n_easy)):
        if not src:
            continue
        if n >= len(src):
            train_rows.extend(src)
        else:
            sel = rng2.choice(len(src), size=n, replace=False)
            train_rows.extend(src[int(k)] for k in sel)
    rng2.shuffle(train_rows)

    work.mkdir(parents=True, exist_ok=True)
    train_p = work / "train.jsonl"
    val_p = work / "val.jsonl"
    eval_p = work / "eval_indices.json"

    with open(train_p, "w") as f:
        for r in train_rows:
            f.write(json.dumps({"input_ids": r["tokens"]}) + "\n")
    with open(val_p, "w") as f:
        for r in val_rows:
            f.write(json.dumps({"input_ids": r["tokens"]}) + "\n")
    json.dump({"counts": counts, "p50": p50, "p85": p85,
               "train": len(train_rows), "val": len(val_rows)},
              open(work / "scoring.json", "w"), indent=2)
    log.info("wrote train=%d val=%d -> %s", len(train_rows), len(val_rows), work)
    return train_p, val_p


# ---------------------------------------------------------------------------
# Multi-GPU LoRA training (delegated to torchrun)
# ---------------------------------------------------------------------------
def run_lora_training(base_model: str, train_p: Path, val_p: Path,
                      out_dir: Path, n_gpus: int, args: argparse.Namespace,
                      bundle: Path) -> Path:
    """Spawn torchrun on the existing training_bundle script."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "torchrun", f"--nproc_per_node={n_gpus}",
        str(bundle / "train_lora_token_ids.py"),
        "--base-model", base_model,
        "--train-data", str(train_p),
        "--val-data", str(val_p),
        "--output-dir", str(out_dir),
        "--seq-len", "2048",
        "--micro-batch-size", str(args.micro_batch),
        "--grad-accum", str(args.grad_accum),
        "--learning-rate", str(args.lr),
        "--epochs", str(args.epochs),
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", "0.05",
    ]
    log.info("training: %s", " ".join(cmd))
    t0 = time.time()
    subprocess.check_call(cmd)
    log.info("training done in %.1fs", time.time() - t0)
    adapter = out_dir / "best_adapter"
    if not adapter.exists():
        # Trainer.save_model may have only put the adapter in the root output_dir.
        # Look for adapter_model files.
        if (out_dir / "adapter_model.safetensors").exists() or \
           (out_dir / "adapter_model.bin").exists():
            adapter = out_dir
        else:
            raise RuntimeError(f"no adapter found in {out_dir}")
    return adapter


def merge_lora(base_model: str, adapter: Path, out: Path) -> Path:
    log.info("merging LoRA %s into %s -> %s", adapter, base_model, out)
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, use_safetensors=True,
    )
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tok.save_pretrained(str(out))
    # Copy config files for parity with king
    for name in ("config.json",):
        src = Path(snapshot_download(base_model, allow_patterns=[name])) / name
        if src.exists():
            shutil.copy(src, out / name)
    del base, merged
    torch.cuda.empty_cache()
    log.info("merged model saved to %s", out)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/root/teutonic-mining/work",
                    help="Working dir on this box")
    ap.add_argument("--bundle", default="/root/teutonic-mining/bundle",
                    help="Path to training_bundle directory")
    ap.add_argument("--n-shards", type=int, default=2,
                    help="Number of dataset shards to download for training")
    ap.add_argument("--shard-start", type=int, default=0,
                    help="Index of first shard to use (other than eval shard)")
    ap.add_argument("--eval-shard", type=int, default=10,
                    help="Held-out shard index for offline paired eval")
    ap.add_argument("--n-eval", type=int, default=2000,
                    help="Sequences for offline paired eval (validator uses 20k)")
    ap.add_argument("--n-score", type=int, default=4000)
    ap.add_argument("--train-per-iter", type=int, default=4000)
    ap.add_argument("--val-size", type=int, default=400)
    ap.add_argument("--max-iters", type=int, default=3,
                    help="Retry training with new seed if first attempt insufficient")
    ap.add_argument("--target-mu", type=float, default=0.05,
                    help="Stop training as soon as offline mu_hat exceeds this")
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-gpus", type=int, default=8)
    ap.add_argument("--upload-repo", default="",
                    help="If set + accepted, push merged model to this HF repo. "
                         "Must contain the first 8 ss58 chars of your coldkey "
                         "(case-insensitive substring) or the validator will "
                         "reject the eval with `coldkey_required`. See "
                         "submit_challenger.py for the gate that enforces this.")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--report-out", default="",
                    help="Write a final JSON verdict to this path")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    # 1. king
    king = fetch_king()
    king_dir = work / "king"
    if king_dir.exists():
        shutil.rmtree(king_dir)
    log.info("downloading king to %s", king_dir)
    snapshot_download(king["hf_repo"], local_dir=str(king_dir),
                      revision=king.get("king_revision") or None,
                      token=args.hf_token or None)
    king_hash = sha256_dir(king_dir)
    log.info("king sha256[:16]=%s", king_hash[:16])

    # 2. dataset shards
    manifest = fetch_manifest(cache)
    train_shard_idxs = list(range(args.shard_start, args.shard_start + args.n_shards))
    if args.eval_shard in train_shard_idxs:
        raise ValueError("eval_shard cannot overlap training shards")
    shards = []
    for idx in train_shard_idxs:
        key = manifest["shards"][idx]["key"]
        path = cache / Path(key).name
        download_shard(key, path)
        arr, _ = load_shard(path)
        log.info("loaded shard %d: %d sequences", idx, len(arr))
        shards.append(arr)

    eval_key = manifest["shards"][args.eval_shard]["key"]
    eval_path = cache / Path(eval_key).name
    download_shard(eval_key, eval_path)
    eval_arr, _ = load_shard(eval_path)
    rng_eval = np.random.default_rng(0xE1A)
    eval_indices = rng_eval.choice(
        len(eval_arr), size=min(args.n_eval, len(eval_arr)), replace=False,
    ).tolist()
    log.info("held-out eval shard %d: %d sequences (sampling %d)",
             args.eval_shard, len(eval_arr), len(eval_indices))

    best = None
    history = []
    for it in range(args.max_iters):
        log.info("=" * 60)
        log.info("=== iteration %d/%d ===", it + 1, args.max_iters)
        log.info("=" * 60)
        seed = args.seed + 1000 * it

        # 3+4. score+curate
        iter_work = work / f"iter_{it:02d}"
        iter_work.mkdir(exist_ok=True)
        train_p, val_p = score_and_curate(
            str(king_dir), shards, args.n_score,
            args.train_per_iter, args.val_size, seed, "cuda:0", iter_work,
        )

        # 5. LoRA train
        out_dir = iter_work / "lora_out"
        adapter = run_lora_training(
            str(king_dir), train_p, val_p, out_dir, args.n_gpus, args,
            Path(args.bundle),
        )

        # 6. merge
        merged_dir = iter_work / "merged"
        merge_lora(str(king_dir), adapter, merged_dir)

        # 7. paired eval
        verdict = paired_eval(
            str(king_dir), str(merged_dir), eval_arr, eval_indices, "cuda:0",
        )
        verdict["iter"] = it
        verdict["seed"] = seed
        history.append(verdict)
        json.dump(verdict, open(iter_work / "verdict.json", "w"), indent=2)

        if best is None or verdict["mu_hat"] > best["mu_hat"]:
            best = {**verdict, "iter_dir": str(iter_work),
                    "merged_dir": str(merged_dir)}
        if verdict["mu_hat"] >= args.target_mu and verdict["accepted"]:
            log.info("target reached at iter %d", it)
            break

    final = {
        "king_repo": king["hf_repo"],
        "king_revision": king.get("king_revision"),
        "king_hash": king_hash,
        "best": best,
        "history": history,
        "ts": time.time(),
    }
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(final, open(args.report_out, "w"), indent=2)
        log.info("wrote verdict to %s", args.report_out)

    # 8. optional upload
    if args.upload_repo and best and best["accepted"]:
        log.info("uploading %s -> %s", best["merged_dir"], args.upload_repo)
        api = HfApi(token=args.hf_token)
        api.create_repo(args.upload_repo, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=best["merged_dir"],
            repo_id=args.upload_repo,
            commit_message=f"Teutonic challenger (mu_hat={best['mu_hat']:.6f})",
            allow_patterns=["*.safetensors", "config.json", "tokenizer*",
                            "special_tokens*", "generation_config.json"],
        )
        info = api.repo_info(args.upload_repo)
        final["uploaded_repo"] = args.upload_repo
        final["uploaded_revision"] = info.sha
        final["challenger_hash"] = sha256_dir(Path(best["merged_dir"]))
        if args.report_out:
            json.dump(final, open(args.report_out, "w"), indent=2)
        log.info("uploaded -> %s @ %s", args.upload_repo, info.sha[:12])
    elif args.upload_repo:
        log.warning("not uploading: best=%s", best)

    log.info("DONE — best mu_hat=%.6f accepted=%s",
             best["mu_hat"] if best else float("nan"),
             best["accepted"] if best else False)


if __name__ == "__main__":
    main()
