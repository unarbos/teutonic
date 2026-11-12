"""Model-free replay of MiMo's fused sigmoid grouped-top-k router."""
import json
import os
from pathlib import Path

os.environ["VLLM_BATCH_INVARIANT"] = "1"

import torch
from vllm import _custom_ops as ops


def main():
    results = []
    for seed in range(3):
        torch.manual_seed(seed)
        logits = torch.randn((8192, 256), device="cuda", dtype=torch.float32)
        bias = torch.randn(256, device="cuda", dtype=torch.float32) * 0.01
        reference = None
        for size in (8192, 127, 128, 512, 721, 932, 1023, 1024, 1025, 2048, 8192):
            weights, ids = ops.grouped_topk(logits[:size], 1, 1, 8, True, 1.0, bias, 1)
            weights, ids = weights[:127].cpu(), ids[:127].cpu()
            if reference is None:
                reference = weights.clone(), ids.clone()
            delta = (weights - reference[0]).abs()
            results.append({"seed": seed, "rows": size,
                            "different_weights": int((delta != 0).sum()),
                            "max_weight_delta": float(delta.max()),
                            "different_ids": int((ids != reference[1]).sum())})
    root = Path(os.environ["TEUTONIC_ROUTER_DIAGNOSTIC_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
