#!/usr/bin/env python3
import argparse
import io
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


LM_HEAD_CHUNK = 256


def parse_npy_header(raw: bytes) -> tuple[int, dict]:
    buf = io.BytesIO(raw)
    magic = buf.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack("<H" if ver[0] == 1 else "<I", buf.read(2 if ver[0] == 1 else 4))[0]
    header = eval(buf.read(hl).decode("latin1").strip())
    return buf.tell(), header


def load_shard(path: str):
    raw = Path(path).read_bytes()
    data_offset, header = parse_npy_header(raw)
    shape = header["shape"]
    if len(shape) != 2:
        raise ValueError(f"expected 2D shard, got shape={shape}")
    n_sequences, seq_len = shape
    arr = np.frombuffer(raw[data_offset:], dtype="<u4").reshape(n_sequences, seq_len)
    return arr, seq_len


@torch.no_grad()
def compute_batch_losses(model, token_batches, device, chunk_size=LM_HEAD_CHUNK):
    input_ids = torch.tensor(token_batches, dtype=torch.long, device=device)
    hidden = model.model(input_ids).last_hidden_state
    lm_head = model.lm_head
    n_positions = input_ids.size(1) - 1
    total_loss = torch.zeros(len(token_batches), device=device)

    for i in range(0, n_positions, chunk_size):
        end_pos = min(i + chunk_size, n_positions)
        chunk_logits = lm_head(hidden[:, i:end_pos, :])
        chunk_labels = input_ids[:, i + 1 : end_pos + 1]
        loss = F.cross_entropy(
            chunk_logits.reshape(-1, chunk_logits.size(-1)),
            chunk_labels.reshape(-1),
            reduction="none",
        )
        total_loss += loss.reshape(len(token_batches), -1).sum(dim=1)
        del chunk_logits, loss

    return (total_loss / n_positions).cpu().tolist()


def repetition_ratio(tokens):
    arr = np.asarray(tokens)
    return float(np.mean(arr[1:] == arr[:-1])) if len(arr) > 1 else 0.0


def unique_ratio(tokens):
    return float(len(set(tokens)) / max(1, len(tokens)))


def repeated_ngram_ratio(tokens, n=4):
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def bucket(loss, unique_r, rep_r, rep_ngram_r, p50, p85):
    if rep_r > 0.2 or rep_ngram_r > 0.5 or unique_r < 0.05:
        return "suspicious"
    if loss >= p85:
        return "hard"
    if loss >= p50 * 0.8 and loss <= p85:
        return "general"
    return "easy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", required=True, help="Local .npy shard path")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    shard, seq_len = load_shard(args.shard)
    rng = np.random.default_rng(args.seed)
    max_samples = min(args.max_samples, len(shard))
    indices = rng.choice(len(shard), size=max_samples, replace=False).tolist()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={"": args.device},
        use_safetensors=True,
    )
    model.eval()

    rows = []
    losses = []
    for i in range(0, len(indices), args.batch_size):
        batch_idx = indices[i:i+args.batch_size]
        batch = [shard[j].tolist() for j in batch_idx]
        batch_losses = compute_batch_losses(model, batch, args.device)
        for idx, tokens, loss in zip(batch_idx, batch, batch_losses):
            row = {
                "sample_index": int(idx),
                "seq_len": seq_len,
                "avg_loss": float(loss),
                "unique_ratio": unique_ratio(tokens),
                "repeat_ratio": repetition_ratio(tokens),
                "repeat_ngram4_ratio": repeated_ngram_ratio(tokens, 4),
                "input_ids": tokens,
            }
            rows.append(row)
            losses.append(loss)

    p50 = float(np.percentile(losses, 50))
    p85 = float(np.percentile(losses, 85))
    for row in rows:
        row["bucket"] = bucket(
            row["avg_loss"],
            row["unique_ratio"],
            row["repeat_ratio"],
            row["repeat_ngram4_ratio"],
            p50,
            p85,
        )

    with open(args.output, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    counts = {}
    for r in rows:
        counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1
    print(json.dumps({"written": len(rows), "bucket_counts": counts, "p50": p50, "p85": p85}, indent=2))


if __name__ == "__main__":
    main()
