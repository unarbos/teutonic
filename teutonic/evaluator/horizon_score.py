"""Isolated BF16 reference scoring process. Layer-sharded correctness baseline.

This uses Accelerate's balanced device map, not tensor parallelism. Benchmark
before production; a faster validated distributed backend can replace it later.
"""

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path


def load_reference(snapshot, *, memory_fraction=0.8):
    import torch
    from transformers import AutoModelForCausalLM

    if torch.cuda.device_count() != 8:
        raise ValueError("reference scoring requires eight visible GPUs")
    if not 0 < memory_fraction < 1:
        raise ValueError("invalid reference memory setting")
    memory = {i: int(torch.cuda.get_device_properties(i).total_memory * memory_fraction) for i in range(8)}
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="balanced", max_memory=memory,
        attn_implementation="sdpa",
    ).eval()
    if model.config.model_type != "k2_horizon":
        raise ValueError("unexpected reference architecture")
    if any(str(d) in {"cpu", "disk"} for d in model.hf_device_map.values()):
        raise ValueError("reference model did not fit on GPUs; CPU/disk offload is disabled")
    return model


def document_nll(model, tokens, *, chunk_size=128):
    import torch
    import torch.nn.functional as F

    if chunk_size < 1 or not 2 <= len(tokens) <= model.config.max_position_embeddings:
        raise ValueError("invalid reference chunk size or document outside context policy")
    input_device = model.get_input_embeddings().weight.device
    head_device = model.get_output_embeddings().weight.device
    with torch.inference_mode():
        ids = torch.tensor([tokens], dtype=torch.long, device=input_device)
        hidden = model.model(ids, use_cache=False).last_hidden_state.to(head_device)
        labels = ids[:, 1:].to(head_device)
        total = torch.zeros((), dtype=torch.float64, device=head_device)
        for start in range(0, len(tokens) - 1, chunk_size):
            end = min(start + chunk_size, len(tokens) - 1)
            logits = model.lm_head(hidden[:, start:end]).float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   labels[:, start:end].reshape(-1), reduction="none")
            total += loss.double().sum()
            del logits, loss
        return total.item()


def score(snapshot, input_path, output_path, *, memory_fraction=0.8, chunk_size=128):
    if chunk_size < 1:
        raise ValueError("invalid reference chunk size")
    model = load_reference(snapshot, memory_fraction=memory_fraction)
    target = Path(output_path)
    temporary = target.with_suffix(".pending")
    with (
        nullcontext(sys.stdin) if str(input_path) == "-" else open(input_path) as source,
        temporary.open("w") as out,
    ):
        for line in source:
            row = json.loads(line)
            tokens = row["tokens"]
            nll = document_nll(model, tokens, chunk_size=chunk_size)
            out.write(json.dumps({"key": row["key"], "nll": nll,
                                  "scored_tokens": len(tokens)-1}) + "\n")
            out.flush()
    temporary.replace(target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    score(args.snapshot, args.input, args.output)


if __name__ == "__main__":
    main()
